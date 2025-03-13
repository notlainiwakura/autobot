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

# Hardcode model for embeddings:
model = SentenceTransformer("msmarco-distilbert-base-v4")

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

# Storage for document chunks and embeddings
document_chunks = []
chunk_embeddings = []
document_metadata = []

def send_long_message_generic(send_func, **kwargs):
    """
    Helper function to send messages that may exceed Slack's 4000 character limit.
    Splits the text on newline boundaries and sends multiple messages.
    """
    text = kwargs.get("text", "")
    max_chars = 4000
    if len(text) <= max_chars:
        send_func(**kwargs)
    else:
        lines = text.split("\n")
        current_chunk = ""
        for line in lines:
            # +1 accounts for the newline character
            if len(current_chunk) + len(line) + 1 > max_chars:
                new_kwargs = kwargs.copy()
                new_kwargs["text"] = current_chunk
                send_func(**new_kwargs)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line
        if current_chunk:
            new_kwargs = kwargs.copy()
            new_kwargs["text"] = current_chunk
            send_func(**new_kwargs)

def fetch_confluence_content():
    """Fetch content from Confluence and prepare it for chunking"""
    all_pages = []

    if SPACE_KEY:
        # If a specific space is provided, get pages from that space
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

def chunk_documents(documents, chunk_size=512, overlap=128):
    """
    Split documents into overlapping chunks.
    Note: This implementation uses sentence boundaries and counts characters.
    For true token counts, integrate a tokenizer that returns token lengths.
    """
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
            # save current chunk and start a new one with overlap.
            if current_length + sentence_length > chunk_size and current_chunk:
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

def search_documents(query, top_k=3):
    """
    Search for the most relevant document chunks.
    Default top_k is set to 3; you can adjust dynamically if needed.
    """
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
    """
    Extract a simple answer from relevant chunks.
    Combines chunks and selects sentences most similar to the query.
    """
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
    chunk_documents(documents)  # Uses default chunk_size=512 and overlap=128
    create_embeddings()
    logger.info("Knowledge base initialized successfully!")

@app.event("app_mention")
def handle_app_mentions(body, say):
    """
    Process messages where the bot is mentioned in channels.
    (This listener is kept so that the bot can still be addressed in channels.)
    """
    event = body["event"]
    text = event["text"]

    # Extract the question (remove the app mention)
    question = re.sub(r'<@[A-Z0-9]+>\s*', '', text).strip()

    if not question:
        say(text="Please ask me a question about Confluence documents! 📚")
        return

    if question.lower() == "refresh":
        say(text="Refreshing knowledge base from Confluence... This may take a few minutes.")
        initialize_knowledge_base()
        say(text="Knowledge base refreshed successfully! 🎉")
        return

    try:
        if len(document_chunks) == 0:
            say(text="Initializing knowledge base for the first time. This may take a few minutes...")
            initialize_knowledge_base()
            say(text="Knowledge base initialized! Now processing your question...")

        relevant_chunks = search_documents(question, top_k=3)
        if not relevant_chunks:
            say(text="I couldn't find any relevant information in the Confluence documents. Please try rephrasing your question.")
            return

        answer = extract_answer(question, relevant_chunks)
        answer_sentences = sent_tokenize(answer)
        bullet_answer = "\n".join([f"- {sentence}" for sentence in answer_sentences])

        seen_sources = {}
        for chunk in relevant_chunks:
            url = chunk['metadata'].get('url')
            title = chunk['metadata'].get('title')
            if url and title and url not in seen_sources:
                seen_sources[url] = title

        related_wikis = ""
        for idx, (url, title) in enumerate(seen_sources.items()):
            related_wikis += f"{idx+1}. [{title}]({url})\n"

        formatted_response = f"*Your Query:* {question}\n"
        formatted_response += "*Answer:*\n" + bullet_answer + "\n\n"
        formatted_response += "*Related Wikis:*\n" + related_wikis.strip()

        send_long_message_generic(say, text=formatted_response)

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        say(text="Sorry, I encountered an error while processing your question. Please try again later.")

@app.message("help")
def help_message(message, say):
    """Provide help information in any channel or DM."""
    help_text = """
*Confluence Knowledge Bot Help*

I'm your secure, local Confluence knowledge assistant. I can help you find information from your Confluence workspace without sending your data to external services!

*Commands:*
• Send me a DM with your question about Confluence documents.
• In channels, mention me with your question (e.g., `@ConfluenceBot What is our return policy?`).
• Send `refresh` to update my knowledge base.
• `help` - Show this help message.
    """
    say(text=help_text)

@app.event("message")
def handle_dm_messages(body, say):
    """
    Process direct messages (DMs) sent to the bot.
    This listener checks if the message is in a DM channel (channel_type "im") and handles it.
    """
    event = body.get("event", {})
    if event.get("channel_type") != "im":
        return  # Only process DMs here

    question = event.get("text", "").strip()

    if not question:
        say(text="Please ask me a question about Confluence documents! 📚")
        return

    if question.lower() == "refresh":
        say(text="Refreshing knowledge base from Confluence... This may take a few minutes.")
        initialize_knowledge_base()
        say(text="Knowledge base refreshed successfully! 🎉")
        return

    try:
        if len(document_chunks) == 0:
            say(text="Initializing knowledge base for the first time. This may take a few minutes...")
            initialize_knowledge_base()
            say(text="Knowledge base initialized! Now processing your question...")

        relevant_chunks = search_documents(question, top_k=3)
        if not relevant_chunks:
            say(text="I couldn't find any relevant information in the Confluence documents. Please try rephrasing your question.")
            return

        answer = extract_answer(question, relevant_chunks)
        answer_sentences = sent_tokenize(answer)
        bullet_answer = "\n".join([f"- {sentence}" for sentence in answer_sentences])

        seen_sources = {}
        for chunk in relevant_chunks:
            url = chunk['metadata'].get('url')
            title = chunk['metadata'].get('title')
            if url and title and url not in seen_sources:
                seen_sources[url] = title

        related_wikis = ""
        for idx, (url, title) in enumerate(seen_sources.items()):
            related_wikis += f"{idx+1}. [{title}]({url})\n"

        formatted_response = f"*Your Query:* {question}\n"
        formatted_response += "*Answer:*\n" + bullet_answer + "\n\n"
        formatted_response += "*Related Wikis:*\n" + related_wikis.strip()

        send_long_message_generic(say, text=formatted_response)

    except Exception as e:
        logger.error(f"Error processing DM question: {str(e)}")
        say(text="Sorry, I encountered an error while processing your question. Please try again later.")

# Initialize the app and start it using Socket Mode
if __name__ == "__main__":
    initialize_knowledge_base()
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
