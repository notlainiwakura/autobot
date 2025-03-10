import re
import logging
import numpy as np
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from atlassian import Confluence
import html2text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import os
from dotenv import load_dotenv
import nltk

# Download NLTK data for tokenization
nltk.download('punkt_tab')
nltk.download('punkt', quiet=True)

# Load variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
CONFLUENCE_URL = os.environ["CONFLUENCE_URL"]
CONFLUENCE_USERNAME = os.environ["CONFLUENCE_USERNAME"]
CONFLUENCE_API_TOKEN = os.environ["CONFLUENCE_API_TOKEN"]
SPACE_KEY = os.environ.get("CONFLUENCE_SPACE_KEY", None)  # Optional, can specify multiple spaces

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize Confluence client
confluence = Confluence(
    url=CONFLUENCE_URL,
    username=CONFLUENCE_USERNAME,
    password=CONFLUENCE_API_TOKEN
)

# Initialize HTML to text converter
html_converter = html2text.HTML2Text()
html_converter.ignore_links = False
html_converter.ignore_images = True

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Storage for document chunks and embeddings
document_chunks = []
chunk_embeddings = []
document_metadata = []


def fetch_confluence_content():
    """Fetch content from Confluence and prepare it for chunking"""
    all_pages = []

    if SPACE_KEY:
        # If specific space is provided, get pages from that space
        space_keys = [s.strip() for s in SPACE_KEY.split(',')]
        for space in space_keys:
            logger.info(f"Fetching pages from space: {space}")
            space_pages = confluence.get_all_pages_from_space(space, limit=100)
            all_pages.extend(space_pages)
    else:
        # Otherwise, get all pages from all spaces
        logger.info("Fetching all pages from all spaces")
        all_spaces = confluence.get_all_spaces()
        for space in all_spaces:
            space_pages = confluence.get_all_pages_from_space(space['key'])
            all_pages.extend(space_pages)

    logger.info(f"Total pages fetched: {len(all_pages)}")

    documents = []
    for page in all_pages:
        try:
            # Get page content
            page_content = confluence.get_page_by_id(page['id'], expand='body.storage')
            html_content = page_content['body']['storage']['value']
            text_content = html_converter.handle(html_content)

            # Create document with metadata
            doc = {
                'content': text_content,
                'metadata': {
                    'title': page['title'],
                    'id': page['id'],
                    'url': f"{CONFLUENCE_URL}/pages/viewpage.action?pageId={page['id']}",
                    'space': page.get('space', {}).get('name', 'Unknown')
                }
            }
            documents.append(doc)
        except Exception as e:
            logger.error(f"Error processing page {page.get('title', 'Unknown')}: {str(e)}")

    return documents


def chunk_documents(documents, chunk_size=500, overlap=100):
    """Split documents into overlapping chunks"""
    global document_chunks, document_metadata

    document_chunks = []
    document_metadata = []

    for doc in documents:
        # Split content into sentences
        sentences = sent_tokenize(doc['content'])

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size and we have content,
            # save current chunk and start a new one with overlap
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save the current chunk
                chunk_text = ' '.join(current_chunk)
                document_chunks.append(chunk_text)
                document_metadata.append(doc['metadata'])

                # Calculate how many sentences to keep for overlap
                overlap_length = 0
                overlap_sentences = []

                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                # Start new chunk with overlap sentences
                current_chunk = overlap_sentences
                current_length = overlap_length

            # Add current sentence to chunk
            current_chunk.append(sentence)
            current_length += sentence_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            document_chunks.append(chunk_text)
            document_metadata.append(doc['metadata'])

    logger.info(f"Created {len(document_chunks)} chunks from {len(documents)} documents")
    return document_chunks, document_metadata


def create_embeddings():
    """Create embeddings for all document chunks"""
    global chunk_embeddings

    logger.info("Creating embeddings for document chunks...")
    chunk_embeddings = model.encode(document_chunks)
    logger.info(f"Created {len(chunk_embeddings)} embeddings")


def search_documents(query, top_k=1):
    """Search for the most relevant document chunks"""
    # Encode the query
    query_embedding = model.encode([query])[0]

    # Calculate similarity scores
    similarity_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]

    # Get top-k results
    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'chunk': document_chunks[idx],
            'metadata': document_metadata[idx],
            'score': similarity_scores[idx]
        })

    return results


def extract_answer(query, relevant_chunks, max_length=40000):
    """Extract a simple answer from relevant chunks"""
    # Combine chunks, starting with the most relevant
    combined_text = ' '.join([chunk['chunk'] for chunk in relevant_chunks])

    # Split into sentences
    sentences = sent_tokenize(combined_text)

    # Get query embedding
    query_embedding = model.encode([query])[0]

    # Calculate sentence similarity to query
    sentence_embeddings = model.encode(sentences)
    similarities = cosine_similarity([query_embedding], sentence_embeddings)[0]

    # Sort sentences by relevance
    sorted_idx = np.argsort(similarities)[::-1]

    # Build answer using most relevant sentences until max_length
    answer = ""
    current_length = 0

    for idx in sorted_idx:
        sentence = sentences[idx]
        if current_length + len(sentence) <= max_length:
            answer += sentence + " "
            current_length += len(sentence)
        else:
            break

    return answer.strip()


def initialize_knowledge_base():
    """Initialize the knowledge base from Confluence"""
    logger.info("Initializing knowledge base from Confluence...")
    documents = fetch_confluence_content()
    chunk_documents(documents)
    create_embeddings()
    logger.info("Knowledge base initialized successfully!")


@app.event("app_mention")
def handle_app_mentions(body, say):
    """Handle app mentions in channels"""
    event = body["event"]
    text = event["text"]

    # Extract the question (remove the app mention)
    question = re.sub(r'<@[A-Z0-9]+>\s*', '', text).strip()

    if not question:
        say("Please ask me a question about Confluence documents! 📚")
        return

    if question.lower() == "refresh":
        say("Refreshing knowledge base from Confluence... This may take a few minutes.")
        initialize_knowledge_base()
        say("Knowledge base refreshed successfully! 🎉")
        return

    try:
        # Check if knowledge base is initialized
        if len(document_chunks) == 0:
            say("Initializing knowledge base for the first time. This may take a few minutes...")
            initialize_knowledge_base()
            say("Knowledge base initialized! Now processing your question...")

        # Search for relevant documents
        relevant_chunks = search_documents(question, top_k=3)

        if not relevant_chunks:
            say("I couldn't find any relevant information in the Confluence documents. Please try rephrasing your question.")
            return

        # Extract answer
        answer = extract_answer(question, relevant_chunks)

        # Format the response with sources
        formatted_response = f"*Answer:*\n{answer}\n\n*Sources:*"

        # Add unique sources
        seen_urls = set()
        for chunk in relevant_chunks:
            url = chunk['metadata'].get('url')
            title = chunk['metadata'].get('title')

            if url and url not in seen_urls:
                seen_urls.add(url)
                formatted_response += f"\n• <{url}|{title}>"

        say(formatted_response)

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        say("Sorry, I encountered an error while processing your question. Please try again later.")


@app.message("help")
def help_message(message, say):
    """Provide help information"""
    help_text = """
*Confluence Knowledge Bot Help*

I'm your secure, local Confluence knowledge assistant. I can help you find information from your Confluence workspace without sending your data to external services!

*Commands:*
• `@ConfluenceBot [your question]` - Ask me anything about your Confluence documents
• `@ConfluenceBot refresh` - Refresh my knowledge base with the latest Confluence content
• `help` - Show this help message

*Examples:*
• `@ConfluenceBot What is our return policy?`
• `@ConfluenceBot Who is responsible for the marketing campaigns?`
• `@ConfluenceBot When is the next product release?`
    """
    say(help_text)


@app.event("app_home_opened")
def update_home_tab(client, event, logger):
    """Publish a custom Home tab view when the App Home is opened."""
    try:
        client.views_publish(
            user_id=event["user"],
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Welcome to *Confluence Knowledge Bot*!\nI'm here to help you find information in your Confluence documents."
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Commands:*\n• `@ConfluenceBot [your question]` - Ask a question\n• `@ConfluenceBot refresh` - Refresh the knowledge base\n• `help` - Get help information"
                        }
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Ask a Question"},
                                "action_id": "ask_question_button"
                            }
                        ]
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.action("ask_question_button")
def handle_ask_question_button(ack, body, client, logger):
    """Handle the Ask a Question button click by opening a modal."""
    ack()
    try:
        trigger_id = body["trigger_id"]
        client.views_open(
            trigger_id=trigger_id,
            view={
                "type": "modal",
                "callback_id": "question_modal",
                "title": {"type": "plain_text", "text": "Ask a Question"},
                "submit": {"type": "plain_text", "text": "Submit"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "question_input",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "question_value",
                            "multiline": True,
                            "placeholder": {"type": "plain_text", "text": "Type your question here..."}
                        },
                        "label": {"type": "plain_text", "text": "Your Question"}
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error opening modal: {e}")


@app.view("question_modal")
def handle_question_modal_submission(ack, body, client, logger):
    """Handle modal submission from the Ask a Question modal."""
    ack()
    try:
        # Extract the submitted question
        submitted_data = body["view"]["state"]["values"]
        question = submitted_data["question_input"]["question_value"]["value"]

        # Process the question using your existing functions
        if len(document_chunks) == 0:
            # If the knowledge base hasn't been initialized, initialize it
            initialize_knowledge_base()

        relevant_chunks = search_documents(question, top_k=3)
        if not relevant_chunks:
            answer = "I couldn't find any relevant information. Please try a different question."
        else:
            answer = extract_answer(question, relevant_chunks)

        # Send the answer as a message to the user (you can also update the Home tab if desired)
        user_id = body["user"]["id"]
        client.chat_postMessage(
            channel=user_id,
            text=f"*Answer:*\n{answer}"
        )
    except Exception as e:
        logger.error(f"Error handling modal submission: {e}")


# Initialize the app
if __name__ == "__main__":
    # Initialize knowledge base on startup
    initialize_knowledge_base()

    # Start the app
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
