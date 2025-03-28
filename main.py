
import re
import logging
import numpy as np
import pickle
import time
import hashlib
import math
from datetime import datetime, timedelta
from pathlib import Path
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
from concurrent.futures import ThreadPoolExecutor
import threading
import json
import tiktoken
from typing import List, Dict, Any, Tuple, Optional

# At the top of your script after imports
print("Checking Vertex AI dependencies...")
try:
    from langchain_google_vertexai import ChatVertexAI
    from langchain.schema import Document
    from langchain.vectorstores import FAISS
    from langchain.embeddings import SentenceTransformerEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    print("Successfully imported Vertex AI libraries")
except ImportError as e:
    print(f"Failed to import Vertex AI dependencies: {e}")

# Check for Vertex AI integration
vertex_ai_available = False
vertex_connector = None
langchain_vector_store = None

if os.environ.get("USE_VERTEX_AI", "false").lower() == "true":
    try:
        print("Attempting to import Vertex AI dependencies...")
        # Import required packages
        from langchain_google_vertexai import ChatVertexAI
        from langchain.schema import Document
        from langchain.vectorstores import FAISS
        from langchain.embeddings import SentenceTransformerEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.prompts import PromptTemplate
        
        # Define the VertexAI connector class inline
        class VertexAIConnector:
            def __init__(self, project_id, location="us-central1", model_name="gemini-1.5-ultra-001",
                         temperature=0.1, max_output_tokens=8000):
                self.project_id = project_id
                self.location = location
                self.model_name = model_name
                self.temperature = temperature
                self.max_output_tokens = max_output_tokens
                
                # Initialize the LLM
                self.llm = ChatVertexAI(
                    project=project_id,
                    location=location,
                    model_name=model_name,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    convert_system_message_to_human=True
                )

                self.prompt_template = PromptTemplate(
                    template="""You are a precise information assistant that provides only the most relevant details from Confluence.

                Context information:
                --------------------------
                {context}
                --------------------------

                Based on the above context, answer this question concisely:
                {question}

                IMPORTANT GUIDELINES:
                - Focus on answering the specific question ONLY
                - Omit any information not directly relevant to the question
                - Use simple, direct language with no unnecessary words
                - Organize your response as a single unified answer, not a collection of excerpts
                - If you don't have enough information, simply say "This specific information isn't available in our documentation."
                - Don't include any source references or mention where the information comes from
                - Format your answer for instant clarity and actionability
                - Prioritize step-by-step instructions when applicable

                Your goal is to provide the single most helpful, focused answer possible.
                """,
                    input_variables=["context", "question"]
                )
            
            def generate_response(self, query, relevant_chunks):
                # Format the context
                context = self._format_context(relevant_chunks)
                
                if not context:
                    return "I couldn't find relevant information to answer your question."
                
                try:
                    # Create input
                    prompt_input = {
                        "context": context,
                        "question": query
                    }
                    
                    # Generate response
                    response = self.llm.invoke(self.prompt_template.format(**prompt_input))
                    
                    # Extract content from the AIMessage object
                    if hasattr(response, 'content'):
                        return response.content.strip()
                    elif isinstance(response, str):
                        return response.strip()
                    else:
                        logger.warning(f"Unexpected response type: {type(response)}")
                        return str(response)
                except Exception as e:
                    logger.error(f"Error generating Vertex AI response: {str(e)}")
                    return self._generate_fallback_response(query, relevant_chunks)
            
            def _format_context(self, relevant_chunks):
                if not relevant_chunks:
                    return ""
                
                # Sort chunks by score
                sorted_chunks = sorted(relevant_chunks, key=lambda x: x.get('score', 0), reverse=True)
                
                # Format chunks
                formatted_chunks = []
                for i, chunk in enumerate(sorted_chunks):
                    metadata = chunk.get('metadata', {})
                    title = metadata.get('title', 'Unknown document')
                    content = chunk.get('chunk', '').strip()
                    
                    formatted_chunk = f"Document {i+1}: {title}\n{content}\n"
                    formatted_chunks.append(formatted_chunk)
                
                return "\n\n".join(formatted_chunks)
            
            def _generate_fallback_response(self, query, relevant_chunks):
                if not relevant_chunks:
                    return "I couldn't find relevant information to answer your question."
                
                # Use highest scoring chunk
                top_chunk = max(relevant_chunks, key=lambda x: x.get('score', 0))
                content = top_chunk.get('chunk', '')
                
                return f"Based on the available information:\n\n{content}\n\n(Note: This is a fallback response due to an issue with the AI processing.)"
        
        # Define function to create QA chain with Vertex AI
        def create_qa_chain_with_vertex(project_id, documents, cache_dir=None,
                                      location="us-central1", model_name="gemini-1.5-pro"):
            """Create a question-answering chain using Vertex AI and LangChain."""
            try:
                # Initialize embedding function
                embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
                
                # Create directory if provided
                if cache_dir:
                    Path(cache_dir).mkdir(exist_ok=True)
                
                # Convert documents to LangChain format
                langchain_docs = []
                for doc in documents:
                    langchain_docs.append(
                        Document(
                            page_content=doc['content'],
                            metadata=doc['metadata']
                        )
                    )
                
                # Create vector store
                vector_store = FAISS.from_documents(
                    documents=langchain_docs,
                    embedding=embeddings
                )
                
                # Initialize Vertex AI connector
                connector = VertexAIConnector(
                    project_id=project_id,
                    location=location,
                    model_name=model_name
                )
                
                return vector_store, connector
            except Exception as e:
                logger.error(f"Error creating QA chain with Vertex AI: {str(e)}")
                return None, None
        
        # Mark as available
        vertex_ai_available = True
        print("Successfully imported Vertex AI dependencies!")
        
    except ImportError as e:
        print(f"Failed to import Vertex AI dependencies: {e}")
        logging.warning(f"Vertex AI integration not available: {str(e)}. Install required packages.")
    except Exception as e:
        print(f"Unexpected error setting up Vertex AI: {e}")
        logging.error(f"Unexpected error setting up Vertex AI: {str(e)}")

# Download NLTK data for tokenization
nltk.download('punkt', quiet=True)

# Load variables from .env file
load_dotenv()

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("confluence_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
CONFLUENCE_URL = os.environ["CONFLUENCE_URL"]
CONFLUENCE_USERNAME = os.environ["CONFLUENCE_USERNAME"]
CONFLUENCE_API_TOKEN = os.environ["CONFLUENCE_API_TOKEN"]
SPACE_KEY = os.environ.get("CONFLUENCE_SPACE_KEY", None)  # Optional, can specify multiple spaces
CACHE_DIR = os.environ.get("CACHE_DIR", "cache")
REFRESH_INTERVAL = int(os.environ.get("REFRESH_INTERVAL_HOURS", 24))  # Default refresh every 24 hours
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))  # Thread pool size for parallel processing
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))   # Default chunk size
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 128))  # Default chunk overlap
TOP_K_RESULTS = int(os.environ.get("TOP_K_RESULTS", 5))  # Default number of results to return
MAX_PAGES_PER_SPACE = int(os.environ.get("MAX_PAGES_PER_SPACE", 500))  # Maximum pages to fetch per space


# Google Cloud & Vertex AI settings
USE_VERTEX_AI = os.environ.get("USE_VERTEX_AI", "false").lower() == "true"
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-1.5-pro")
LANGCHAIN_CACHE_DIR = os.environ.get("LANGCHAIN_CACHE_DIR", "langchain_cache")

# Message deduplication settings
MESSAGE_CACHE_TTL = int(os.environ.get("MESSAGE_CACHE_TTL", 60))  # Time in seconds to keep messages in deduplication cache
THREAD_TIMEOUT = int(os.environ.get("THREAD_TIMEOUT", 300))  # 5 minutes in seconds

# Create cache directory if it doesn't exist
Path(CACHE_DIR).mkdir(exist_ok=True)

# Hardcode model for embeddings:
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "msmarco-distilbert-base-v4")
model = SentenceTransformer(EMBEDDING_MODEL)

# Initialize tokenizer for token counting (used for chunking)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize Confluence client
confluence = Confluence(
    url=CONFLUENCE_URL,
    username=CONFLUENCE_USERNAME,
    password=CONFLUENCE_API_TOKEN,
    timeout=60  # Increased timeout for larger Confluence instances
)

# Initialize HTML to text converter
html_converter = html2text.HTML2Text()
html_converter.ignore_links = False
html_converter.ignore_images = True
html_converter.body_width = 0  # Don't wrap text at a specific width

# Storage for document chunks and embeddings
document_chunks = []
chunk_embeddings = []
document_metadata = []
knowledge_base_lock = threading.Lock()  # Thread safety for knowledge base access
last_refresh_time = None
auto_refresh_thread = None
refresh_event = threading.Event()

# Message deduplication cache
processed_messages = {}

# Vertex AI and Langchain components
vertex_connector = None
langchain_vector_store = None

# Mapping of Slack channel IDs to in-progress threads
active_threads = {}  # Format: {"channel_id:ts": (start_time, user_id)}
active_threads_lock = threading.Lock()

class ConfluenceCache:
    """Class to handle caching of Confluence content and embeddings"""
    
    @staticmethod
    def get_cache_path(cache_type):
        """Get the path to a specific cache file"""
        return os.path.join(CACHE_DIR, f"{cache_type}.pkl")
    
    @staticmethod
    def save_to_cache(data, cache_type):
        """Save data to cache file"""
        try:
            with open(ConfluenceCache.get_cache_path(cache_type), 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {cache_type} to cache")
            return True
        except Exception as e:
            logger.error(f"Error saving {cache_type} to cache: {str(e)}")
            return False
    
    @staticmethod
    def load_from_cache(cache_type):
        """Load data from cache file"""
        try:
            cache_path = ConfluenceCache.get_cache_path(cache_type)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded {cache_type} from cache")
                return data
            else:
                logger.info(f"No cache file found for {cache_type}")
                return None
        except Exception as e:
            logger.error(f"Error loading {cache_type} from cache: {str(e)}")
            return None
    
    @staticmethod
    def is_cache_valid():
        """Check if cache exists and is not too old"""
        try:
            metadata_path = ConfluenceCache.get_cache_path("metadata")
            if not os.path.exists(metadata_path):
                return False
                
            # Check when the cache was last modified
            mtime = os.path.getmtime(metadata_path)
            cache_time = datetime.fromtimestamp(mtime)
            now = datetime.now()
            
            # If cache is older than refresh interval, it's invalid
            return (now - cache_time) < timedelta(hours=REFRESH_INTERVAL)
        except Exception as e:
            logger.error(f"Error checking cache validity: {str(e)}")
            return False

def should_process_message(event):
    """
    Check if a message should be processed by creating a unique hash and
    checking if we've seen it recently.
    """
    # Create a unique message signature
    message_text = event.get("text", "")
    channel = event.get("channel", "")
    ts = event.get("ts", "")
    
    # Create a unique hash for this message
    message_hash = hashlib.md5(f"{channel}:{ts}:{message_text}".encode()).hexdigest()
    
    current_time = datetime.now()
    
    # Clean up old entries from cache
    expired_keys = []
    for key, (timestamp, _) in processed_messages.items():
        if current_time - timestamp > timedelta(seconds=MESSAGE_CACHE_TTL):
            expired_keys.append(key)
    
    for key in expired_keys:
        del processed_messages[key]
    
    # Check if we've processed this message recently
    if message_hash in processed_messages:
        logger.info(f"Skipping duplicate message: {message_text[:30]}...")
        return False
    
    # Add to cache
    processed_messages[message_hash] = (current_time, message_text)
    return True

def manage_active_thread(channel_id, thread_ts, user_id, action="add"):
    """
    Add, check, or remove a thread from the active threads tracking.
    Returns True if the thread can be processed, False if it's already being processed.
    """
    thread_key = f"{channel_id}:{thread_ts}"
    
    with active_threads_lock:
        current_time = time.time()
        
        # Clean up any expired threads
        expired_keys = []
        for key, (start_time, _) in active_threads.items():
            if current_time - start_time > THREAD_TIMEOUT:
                expired_keys.append(key)
                
        for key in expired_keys:
            logger.info(f"Removing expired thread: {key}")
            del active_threads[key]
        
        if action == "check":
            return thread_key in active_threads
            
        elif action == "add":
            if thread_key in active_threads:
                return False
            active_threads[thread_key] = (current_time, user_id)
            return True
            
        elif action == "remove":
            if thread_key in active_threads:
                del active_threads[thread_key]
            return True
    
    return False

def count_tokens(text):
    """Count the number of tokens in a text using tiktoken"""
    return len(tokenizer.encode(text))

def send_long_message(say, text, thread_ts=None, blocks=None):
    """
    Send messages that may exceed Slack's character limit by splitting intelligently.
    Uses blocks when available for better formatting.
    """
    max_chars = 3000  # Setting a bit lower than the 4000 limit to be safe
    
    # If message is short enough, send it directly
    if len(text) <= max_chars:
        say(text=text, thread_ts=thread_ts, blocks=blocks)
        return
    
    # If we have blocks, we need a different approach
    if blocks:
        # First send just the text portion in chunks
        send_long_message(say, text, thread_ts)
        
        # Then send blocks in separate messages if needed
        for i in range(0, len(blocks), 50):  # Slack has a limit of 50 blocks per message
            block_subset = blocks[i:i+50]
            say(text="", thread_ts=thread_ts, blocks=block_subset)
        return
        
    # Split on sentence boundaries for more natural breaks
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            # Current chunk would be too large, store it and start a new one
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            # Add to current chunk with a space if not empty
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Send each chunk in order
    for i, chunk in enumerate(chunks):
        # Add continuation indicator for multi-part messages
        if len(chunks) > 1:
            if i == 0:
                chunk += "\n\n_(message continued in thread...)_"
            else:
                chunk = f"_(part {i+1}/{len(chunks)})_\n\n" + chunk
        
        say(text=chunk, thread_ts=thread_ts)


def fetch_confluence_content():
    """Fetch content from Confluence and prepare it for chunking with proper pagination handling"""
    all_pages = []

    if SPACE_KEY:
        # If specific spaces are provided, get pages from those spaces
        space_keys = [s.strip() for s in SPACE_KEY.split(',')]
        for space in space_keys:
            logger.info(f"Fetching pages from space: {space}")
            try:
                # Initialize variables for pagination
                start = 0
                page_limit = 100  # This is the API's internal limit per request
                all_space_pages = []
                has_more = True

                # Keep fetching pages until we get all of them or reach our maximum limit
                while has_more and start < MAX_PAGES_PER_SPACE:
                    # Get a batch of pages with pagination parameters
                    logger.info(f"Fetching pages {start} to {start + page_limit} from space {space}")
                    batch = confluence.get_all_pages_from_space(
                        space,
                        start=start,
                        limit=page_limit,
                        expand="body.storage"
                    )

                    # If we got fewer results than the limit, we've reached the end
                    if len(batch) < page_limit:
                        has_more = False

                    # Add pages to our collection
                    all_space_pages.extend(batch)

                    # Update start for next batch
                    start += len(batch)

                    # If no results were returned, we're done
                    if not batch:
                        has_more = False

                logger.info(f"Fetched {len(all_space_pages)} pages from space {space}")
                all_pages.extend(all_space_pages)
            except Exception as e:
                logger.error(f"Error fetching pages from space {space}: {str(e)}")
    else:
        # Otherwise, get all pages from all spaces
        logger.info("Fetching all spaces")
        try:
            all_spaces = confluence.get_all_spaces()
            logger.info(f"Found {len(all_spaces)} spaces")

            for space in all_spaces:
                try:
                    logger.info(f"Fetching pages from space: {space['key']}")

                    # Initialize variables for pagination
                    start = 0
                    page_limit = 100  # This is the API's internal limit per request
                    all_space_pages = []
                    has_more = True

                    # Keep fetching pages until we get all of them or reach our maximum limit
                    while has_more and start < MAX_PAGES_PER_SPACE:
                        # Get a batch of pages with pagination parameters
                        logger.info(f"Fetching pages {start} to {start + page_limit} from space {space['key']}")
                        batch = confluence.get_all_pages_from_space(
                            space['key'],
                            start=start,
                            limit=page_limit,
                            expand="body.storage"
                        )

                        # If we got fewer results than the limit, we've reached the end
                        if len(batch) < page_limit:
                            has_more = False

                        # Add pages to our collection
                        all_space_pages.extend(batch)

                        # Update start for next batch
                        start += len(batch)

                        # If no results were returned, we're done
                        if not batch:
                            has_more = False

                    logger.info(f"Fetched {len(all_space_pages)} pages from space {space['key']}")
                    all_pages.extend(all_space_pages)
                except Exception as e:
                    logger.error(f"Error fetching pages from space {space['key']}: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching all spaces: {str(e)}")

    logger.info(f"Total pages fetched: {len(all_pages)}")

    # Process pages in parallel
    documents = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all page processing tasks
        future_to_page = {executor.submit(process_page, page): page for page in all_pages}

        # Collect results as they complete
        for future in future_to_page:
            try:
                doc = future.result()
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing page: {str(e)}")

    logger.info(f"Successfully processed {len(documents)} pages")
    return documents

def process_page(page):
    """Process a single Confluence page and return a document"""
    try:
        # Get the page ID
        page_id = page['id']
        
        # Get page content with expanded body and space information
        page_content = confluence.get_page_by_id(page_id, expand='body.storage,version,history,metadata.labels,space')
        
        # Extract HTML content
        html_content = page_content['body']['storage']['value']
        
        # Convert HTML to text
        text_content = html_converter.handle(html_content)
        
        # Extract labels if available
        labels = []
        if 'metadata' in page_content and 'labels' in page_content['metadata']:
            for label in page_content['metadata']['labels'].get('results', []):
                labels.append(label.get('name', ''))
        
        # Get last updated info
        updated_date = page_content.get('version', {}).get('when', '')
        last_updater = page_content.get('version', {}).get('by', {}).get('displayName', 'Unknown')
        
        # Get space key from expanded page content
        space_key = page_content.get('space', {}).get('key', page.get('space', {}).get('key', 'Unknown'))
        
        # Create document with enhanced metadata
        doc = {
            'content': text_content,
            'metadata': {
                'title': page_content.get('title', page.get('title', 'Untitled')),
                'id': page_id,
                'url': f"{CONFLUENCE_URL}/wiki/spaces/{space_key}/pages/{page_id}",
                'space': page_content.get('space', {}).get('name', page.get('space', {}).get('name', 'Unknown')),
                'space_key': space_key,
                'labels': labels,
                'last_updated': updated_date,
                'last_updater': last_updater
            }
        }
        
        # Add debugging log
        logger.info(f"Created URL for page {page_id} in space {space_key}: {doc['metadata']['url']}")
        
        return doc
    except Exception as e:
        logger.error(f"Error processing page {page.get('title', 'Unknown')}: {str(e)}")
        return None

def chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split documents into overlapping chunks using token-based approach
    with sentence boundaries preserved.
    """
    global document_chunks, document_metadata
    
    document_chunks = []
    document_metadata = []
    
    for doc in documents:
        # Split content into sentences
        sentences = sent_tokenize(doc['content'])
        
        # Get token counts for each sentence
        sentence_token_counts = [count_tokens(sentence) for sentence in sentences]
        
        current_chunk_sentences = []
        current_chunk_token_count = 0
        
        for i, (sentence, token_count) in enumerate(zip(sentences, sentence_token_counts)):
            # If a single sentence exceeds chunk size, we need to split it further
            if token_count > chunk_size:
                # Just include it as its own chunk to avoid complexity
                # This is a rare case that would require word-level chunking
                document_chunks.append(sentence)
                document_metadata.append({
                    **doc['metadata'],
                    'chunk_index': len(document_chunks) - 1,
                    'is_large_sentence': True
                })
                continue
            
            # If adding this sentence would exceed the chunk size and we have content
            if current_chunk_token_count + token_count > chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = ' '.join(current_chunk_sentences)
                document_chunks.append(chunk_text)
                document_metadata.append({
                    **doc['metadata'],
                    'chunk_index': len(document_chunks) - 1
                })
                
                # Calculate overlap using tokens
                overlap_sentences = []
                overlap_token_count = 0
                
                # Start from the end and work backwards for overlap
                for s, s_token_count in reversed(list(zip(current_chunk_sentences,
                                                         [count_tokens(s) for s in current_chunk_sentences]))):
                    if overlap_token_count + s_token_count <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_token_count += s_token_count
                    else:
                        break
                
                # Start new chunk with overlap sentences
                current_chunk_sentences = overlap_sentences
                current_chunk_token_count = overlap_token_count
            
            # Add current sentence to the chunk
            current_chunk_sentences.append(sentence)
            current_chunk_token_count += token_count
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            document_chunks.append(chunk_text)
            document_metadata.append({
                **doc['metadata'],
                'chunk_index': len(document_chunks) - 1
            })
    
    logger.info(f"Created {len(document_chunks)} chunks from {len(documents)} documents")
    return document_chunks, document_metadata

def create_embeddings():
    """Create embeddings for all document chunks with batching"""
    global chunk_embeddings
    
    logger.info("Creating embeddings for document chunks...")
    
    # Process in batches to avoid memory issues with large knowledge bases
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(document_chunks), batch_size):
        batch = document_chunks[i:i + batch_size]
        try:
            batch_embeddings = model.encode(batch, show_progress_bar=(len(batch) > 10))
            all_embeddings.append(batch_embeddings)
        except Exception as e:
            logger.error(f"Error creating embeddings for batch {i//batch_size}: {str(e)}")
            # Create zero embeddings as placeholders for failed batches
            batch_embeddings = np.zeros((len(batch), model.get_sentence_embedding_dimension()))
            all_embeddings.append(batch_embeddings)
    
    # Combine all batches
    if all_embeddings:
        chunk_embeddings = np.vstack(all_embeddings)
    else:
        chunk_embeddings = np.array([])
    
    logger.info(f"Created {len(chunk_embeddings)} embeddings")


def search_documents(query, top_k=5, min_score=0.4):
    """Search for document chunks and prioritize documents with multiple relevant chunks."""
    # Encode the query
    query_embedding = model.encode([query])[0]

    # Calculate similarity scores
    similarity_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]

    # Get results above minimum score
    candidate_indices = [idx for idx in range(len(similarity_scores)) if similarity_scores[idx] >= min_score]

    # Group by document ID
    doc_id_to_chunks = {}
    for idx in candidate_indices:
        doc_id = document_metadata[idx]['id']
        if doc_id not in doc_id_to_chunks:
            doc_id_to_chunks[doc_id] = []
        doc_id_to_chunks[doc_id].append({
            'idx': idx,
            'score': float(similarity_scores[idx])
        })

    # Calculate document scores by combining chunk scores
    # Documents with multiple relevant chunks get boosted
    doc_scores = {}
    for doc_id, chunks in doc_id_to_chunks.items():
        # Calculate average score
        avg_score = sum(chunk['score'] for chunk in chunks) / len(chunks)
        # Boost score based on number of relevant chunks (logarithmic scaling)
        boost = 1 + (math.log10(len(chunks)) * 0.5)
        doc_scores[doc_id] = avg_score * boost

    # Sort documents by combined score
    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    # Take top documents and their chunks
    results = []
    docs_selected = 0

    # First try to get everything from a single document if it has enough content
    best_doc_id = sorted_doc_ids[0] if sorted_doc_ids else None
    if best_doc_id and len(doc_id_to_chunks[best_doc_id]) >= 2:
        # Sort chunks by score
        sorted_chunks = sorted(doc_id_to_chunks[best_doc_id], key=lambda x: x['score'], reverse=True)
        # Take top chunks from best document
        for chunk in sorted_chunks[:3]:  # Limit to 3 chunks from same document
            idx = chunk['idx']
            results.append({
                'chunk': document_chunks[idx],
                'metadata': document_metadata[idx],
                'score': chunk['score']
            })
        docs_selected = 1
    else:
        # If single best document doesn't have enough content, take top chunks from top 2 docs
        for doc_id in sorted_doc_ids[:2]:
            docs_selected += 1
            # Sort chunks by score
            sorted_chunks = sorted(doc_id_to_chunks[doc_id], key=lambda x: x['score'], reverse=True)
            # Take top chunks from this document
            for chunk in sorted_chunks[:2]:  # Limit to 2 chunks per document
                idx = chunk['idx']
                results.append({
                    'chunk': document_chunks[idx],
                    'metadata': document_metadata[idx],
                    'score': chunk['score']
                })

    logger.info(f"Selected {len(results)} chunks from {docs_selected} documents")
    return results


def extract_answer(query, relevant_chunks, max_tokens=1200):
    """Extract a coherent answer from relevant chunks, prioritizing single-source information."""
    if not relevant_chunks:
        return "I couldn't find relevant information to answer your question."

    # If Vertex AI integration is enabled and available, use it
    global vertex_connector
    if USE_VERTEX_AI and vertex_ai_available and vertex_connector:
        try:
            logger.info("Using Vertex AI for response generation")
            return vertex_connector.generate_response(query, relevant_chunks)
        except Exception as e:
            logger.error(f"Error using Vertex AI for response: {str(e)}")
            logger.info("Falling back to default extraction method")

    # Group chunks by document ID
    doc_id_to_chunks = {}
    for chunk in relevant_chunks:
        doc_id = chunk['metadata']['id']
        if doc_id not in doc_id_to_chunks:
            doc_id_to_chunks[doc_id] = []
        doc_id_to_chunks[doc_id].append(chunk)

    # Find document with most chunks
    best_doc_id = max(doc_id_to_chunks.keys(), key=lambda x: len(doc_id_to_chunks[x]))
    best_doc_chunks = doc_id_to_chunks[best_doc_id]

    # Sort best document chunks by score
    best_doc_chunks.sort(key=lambda x: x['score'], reverse=True)

    # Combine chunks from best document into single context
    context = "\n".join([chunk['chunk'] for chunk in best_doc_chunks])

    # If context is too long, we'll need to be selective
    if count_tokens(context) > max_tokens:
        # Get query embedding for sentence-level relevance
        query_embedding = model.encode([query])[0]

        # Split into sentences and compute relevance
        sentences = sent_tokenize(context)
        sentence_embeddings = model.encode(sentences)
        sentence_scores = cosine_similarity([query_embedding], sentence_embeddings)[0]

        # Sort sentences by relevance
        sorted_sentences = [(s, score) for s, score in zip(sentences, sentence_scores)]
        sorted_sentences.sort(key=lambda x: x[1], reverse=True)

        # Take most relevant sentences up to token limit
        selected_sentences = []
        token_count = 0

        for sentence, _ in sorted_sentences:
            sentence_tokens = count_tokens(sentence)
            if token_count + sentence_tokens <= max_tokens:
                selected_sentences.append(sentence)
                token_count += sentence_tokens
            else:
                break

        context = " ".join(selected_sentences)

    # Prepare the prompt for Vertex AI model
    system_prompt = f"""
    Answer the following question precisely using ONLY the information in the context below.
    If the context doesn't contain relevant information, say "I don't have specific information on this topic."
    Focus only on information directly relevant to the question.
    Use clear, natural language, and organize information logically.

    CONTEXT:
    {context}

    QUESTION:
    {query}
    """

    if USE_VERTEX_AI and vertex_ai_available:
        try:
            # Direct call to Vertex AI with focused prompt
            response = vertex_connector.llm.invoke(system_prompt)
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
        except Exception as e:
            logger.error(f"Error calling Vertex AI directly: {str(e)}")

    # Fallback method if Vertex AI isn't available or fails
    # Simple approach: return the most relevant chunk with a clean intro
    top_chunk = best_doc_chunks[0]['chunk']

    # Clean up by removing section headings and formatting
    cleaned_chunk = re.sub(r'#{1,6}\s+.*?\n', '', top_chunk)  # Remove markdown headings
    cleaned_chunk = re.sub(r'\*\*.*?\*\*', '', cleaned_chunk)  # Remove bold text markers

    return f"Based on our documentation: {cleaned_chunk}"


def format_response(query, answer, relevant_chunks):
    """Format the response with only the most relevant source."""
    # Format answer text
    formatted_answer = answer.replace("\n\n", "\n").strip()

    # Find the most relevant document (one with highest scoring chunk)
    if relevant_chunks:
        top_chunk = max(relevant_chunks, key=lambda x: x['score'])
        url = top_chunk['metadata'].get('url')
        title = top_chunk['metadata'].get('title')
        space = top_chunk['metadata'].get('space')
        last_updated = top_chunk['metadata'].get('last_updated', '')

        # Format the date if available
        date_str = ""
        if last_updated:
            try:
                # Parse ISO format date if possible
                date_obj = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                date_str = f", last updated {date_obj.strftime('%b %d, %Y')}"
            except:
                # If parsing fails, use the raw string
                date_str = f", last updated {last_updated}"

        source_info = f"*{title}* ({space}{date_str})\n{url}" if url and title else ""
    else:
        source_info = ""

    # Build the final response
    formatted_response = f"*Your Query:* {query}\n\n"
    formatted_response += "*Answer:*\n" + formatted_answer + "\n\n"

    if source_info:
        formatted_response += "*Source:*\n" + source_info

    return formatted_response

def initialize_knowledge_base(force=False):
    """Initialize the knowledge base from Confluence or load from cache"""
    global document_chunks, chunk_embeddings, document_metadata, last_refresh_time
    global vertex_connector, langchain_vector_store
    
    with knowledge_base_lock:
        # If we have a valid cache and aren't forcing a refresh, use it
        if not force and ConfluenceCache.is_cache_valid():
            logger.info("Loading knowledge base from cache...")
            document_chunks = ConfluenceCache.load_from_cache("chunks")
            chunk_embeddings = ConfluenceCache.load_from_cache("embeddings")
            document_metadata = ConfluenceCache.load_from_cache("metadata")
            
            if (document_chunks is not None and
                isinstance(chunk_embeddings, np.ndarray) and len(chunk_embeddings) > 0 and
                document_metadata is not None):
                logger.info(f"Successfully loaded knowledge base from cache: {len(document_chunks)} chunks")
                last_refresh_time = datetime.now()
                
                # Initialize Vertex AI if enabled
                if USE_VERTEX_AI and vertex_ai_available and GCP_PROJECT_ID:
                    try:
                        logger.info("Initializing Vertex AI integration...")
                        # Convert document chunks to format expected by Langchain
                        documents = []
                        for i, chunk in enumerate(document_chunks):
                            documents.append({
                                'content': chunk,
                                'metadata': document_metadata[i]
                            })
                        
                        # Create Langchain components
                        langchain_vector_store, vertex_connector = create_qa_chain_with_vertex(
                            project_id=GCP_PROJECT_ID,
                            documents=documents,
                            cache_dir=LANGCHAIN_CACHE_DIR,
                            location=GCP_LOCATION,
                            model_name=VERTEX_MODEL
                        )
                        logger.info("Vertex AI integration initialized successfully")
                    except Exception as e:
                        logger.error(f"Error initializing Vertex AI: {str(e)}")
                        vertex_connector = None
                
                return True
            else:
                logger.warning("Cache seems corrupt or incomplete, rebuilding knowledge base")
                # Fall through to rebuild
        
        # If cache is invalid or we're forcing a refresh, build from scratch
        logger.info("Building knowledge base from Confluence...")
        try:
            start_time = time.time()
            documents = fetch_confluence_content()
            chunk_documents(documents)
            create_embeddings()
            last_refresh_time = datetime.now()
            
            # Save to cache
            ConfluenceCache.save_to_cache(document_chunks, "chunks")
            ConfluenceCache.save_to_cache(chunk_embeddings, "embeddings")
            ConfluenceCache.save_to_cache(document_metadata, "metadata")
            
            # Initialize Vertex AI if enabled
            if USE_VERTEX_AI and vertex_ai_available and GCP_PROJECT_ID:
                try:
                    logger.info("Initializing Vertex AI integration...")
                    # Convert document chunks to format expected by Langchain
                    langchain_documents = []
                    for i, chunk in enumerate(document_chunks):
                        langchain_documents.append({
                            'content': chunk,
                            'metadata': document_metadata[i]
                        })
                    
                    # Create Langchain components
                    langchain_vector_store, vertex_connector = create_qa_chain_with_vertex(
                        project_id=GCP_PROJECT_ID,
                        documents=langchain_documents,
                        cache_dir=LANGCHAIN_CACHE_DIR,
                        location=GCP_LOCATION,
                        model_name=VERTEX_MODEL
                    )
                    logger.info("Vertex AI integration initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing Vertex AI: {str(e)}")
                    vertex_connector = None
            
            elapsed_time = time.time() - start_time
            logger.info(f"Knowledge base initialized in {elapsed_time:.2f} seconds with {len(document_chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            return False

def auto_refresh_knowledge_base():
    """Background thread to automatically refresh the knowledge base"""
    global last_refresh_time
    
    logger.info("Starting automatic knowledge base refresh thread")
    while not refresh_event.is_set():
        # Sleep for a bit, checking periodically if we should exit
        for _ in range(60):  # Check every minute if we should exit
            if refresh_event.is_set():
                break
            time.sleep(60)
        
        # If it's time to refresh (or it's never been refreshed), do it
        with knowledge_base_lock:
            if (not last_refresh_time or
                (datetime.now() - last_refresh_time > timedelta(hours=REFRESH_INTERVAL))):
                logger.info("Auto-refreshing knowledge base...")
                initialize_knowledge_base(force=True)
    
    logger.info("Automatic knowledge base refresh thread stopped")

def ensure_knowledge_base(say=None):
    """Ensure the knowledge base is initialized"""
    global document_chunks, auto_refresh_thread
    
    # Start the auto-refresh thread if it's not running
    if auto_refresh_thread is None or not auto_refresh_thread.is_alive():
        refresh_event.clear()
        auto_refresh_thread = threading.Thread(target=auto_refresh_knowledge_base, daemon=True)
        auto_refresh_thread.start()
    
    # Initialize if needed
    with knowledge_base_lock:
        if len(document_chunks) == 0 or not isinstance(chunk_embeddings, np.ndarray) or len(chunk_embeddings) == 0:
            if say:
                say(text="Initializing knowledge base for the first time. This may take a few minutes...")
            success = initialize_knowledge_base()
            if success and say:
                say(text="Knowledge base initialized! Now processing your question...")
            return success
        return True

@app.event("app_mention")
def handle_app_mentions(body, say, client):
    """Process messages where the bot is mentioned in channels."""
    event = body["event"]
    
    # Skip if we've processed this message recently
    if not should_process_message(event):
        return
        
    channel_id = event["channel"]
    thread_ts = event.get("thread_ts", event.get("ts"))
    user_id = event["user"]
    text = event["text"]
    
    # Extract the question (remove the app mention)
    question = re.sub(r'<@[A-Z0-9]+>\s*', '', text).strip()
    
    if not question:
        say(text="Please ask me a question about Confluence documents! 📚")
        return
    
    # Check for admin commands
    if question.lower() == "refresh":
        user_info = client.users_info(user=user_id)
        user_is_admin = user_info.get("user", {}).get("is_admin", False)
        
        if user_is_admin:
            say(text="Refreshing knowledge base from Confluence... This may take a few minutes.")
            success = initialize_knowledge_base(force=True)
            if success:
                say(text="Knowledge base refreshed successfully! 🎉")
            else:
                say(text="There was an error refreshing the knowledge base. Please check the logs.")
        else:
            say(text="Sorry, only workspace admins can refresh the knowledge base.")
        return
    
    # Check if this thread is already being processed
    if not manage_active_thread(channel_id, thread_ts, user_id, "add"):
        say(text="I'm already processing a request in this thread. Please wait until it's complete before asking another question.")
        return
    
    try:
        # Show typing indicator
        try:
            client.reactions_add(
                channel=channel_id,
                timestamp=thread_ts,
                name="thinking_face"
            )
        except Exception as e:
            # If we can't add reactions, just log it and continue
            logger.warning(f"Could not add reaction: {str(e)}. This is non-critical.")
            # Use typing indicator as alternative if reactions fail
            try:
                client.conversations_mark(channel=channel_id, ts=thread_ts)
            except:
                pass
        
        # Ensure knowledge base is initialized
        if not ensure_knowledge_base(say):
            say(text="I'm having trouble accessing the knowledge base. Please try again later.")
            return
        
        # Log the query
        logger.info(f"Processing question from <@{user_id}>: {question}")
        
        # Search for relevant documents
        relevant_chunks = search_documents(question, top_k=TOP_K_RESULTS)
        
        if not relevant_chunks:
            say(text="I couldn't find any relevant information in the Confluence documents. Please try rephrasing your question or check that the documents you're looking for are in the indexed spaces.")
            return
        
        # Extract answer
        answer = extract_answer(question, relevant_chunks)
        
        # Format response
        formatted_response = format_response(question, answer, relevant_chunks)
        
        # Remove typing indicator
        try:
            client.reactions_remove(
                channel=channel_id,
                timestamp=thread_ts,
                name="thinking_face"
            )
        except Exception as e:
            # If we can't remove reactions, just log it and continue
            logger.warning(f"Could not remove reaction: {str(e)}. This is non-critical.")
        
        # Send the response
        send_long_message(say, formatted_response, thread_ts=thread_ts)
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        say(text="Sorry, I encountered an error while processing your question. Please try again later.")
    
    finally:
        # Remove the thread from active tracking
        manage_active_thread(channel_id, thread_ts, user_id, "remove")

@app.message("help")
def help_message(message, say):
    """Provide help information in any channel or DM."""
    help_text = """
*Confluence Knowledge Bot Help*

I'm your secure, local Confluence knowledge assistant. I can help you find information from your Confluence workspace without sending your data to external services!

*Commands:*
• Send me a DM with your question about Confluence documents.
• In channels, mention me with your question (e.g., `@ConfluenceBot What is our return policy?`).
• Admins can send `refresh` to update my knowledge base.
• `help` - Show this help message.

*Tips for Good Questions:*
• Be specific in your questions
• Include keywords that might appear in the documents
• If you don't get a good answer, try rephrasing your question

*My Knowledge:*
• I only know what's in your Confluence workspace
• My knowledge updates automatically every 24 hours
• I can search across all spaces or specific ones (configured by your admin)
    """
    say(text=help_text)

@app.event("message")
def handle_dm_messages(body, say, client):
    """Process direct messages (DMs) sent to the bot."""
    event = body.get("event", {})
    
    # Only process DMs
    if event.get("channel_type") != "im":
        return
    
    # Skip if we've processed this message recently
    if not should_process_message(event):
        return
        
    channel_id = event.get("channel")
    user_id = event.get("user")
    thread_ts = event.get("thread_ts", event.get("ts"))
    question = event.get("text", "").strip()
    
    # Ignore messages from bots (including ourselves)
    if event.get("bot_id") or user_id == "USLACKBOT":
        return
    
    if not question:
        say(text="Please ask me a question about Confluence documents! 📚")
        return
    
    # Check for admin commands
    if question.lower() == "refresh":
        user_info = client.users_info(user=user_id)
        user_is_admin = user_info.get("user", {}).get("is_admin", False)
        
        if user_is_admin:
            say(text="Refreshing knowledge base from Confluence... This may take a few minutes.")
            success = initialize_knowledge_base(force=True)
            if success:
                say(text="Knowledge base refreshed successfully! 🎉")
            else:
                say(text="There was an error refreshing the knowledge base. Please check the logs.")
        else:
            say(text="Sorry, only workspace admins can refresh the knowledge base.")
        return
    
    if question.lower() == "help":
        help_message(event, say)
        return
    
    # Check if this thread is already being processed
    if not manage_active_thread(channel_id, thread_ts, user_id, "add"):
        say(text="I'm already processing a request in this thread. Please wait until it's complete before asking another question.")
        return
    
    try:
        # Show typing indicator
        try:
            client.reactions_add(
                channel=channel_id,
                timestamp=thread_ts,
                name="thinking_face"
            )
        except Exception as e:
            # If we can't add reactions, just log it and continue
            logger.warning(f"Could not add reaction: {str(e)}. This is non-critical.")
        
        # Ensure knowledge base is initialized
        if not ensure_knowledge_base(say):
            say(text="I'm having trouble accessing the knowledge base. Please try again later.")
            return
        
        # Log the query
        logger.info(f"Processing DM question from <@{user_id}>: {question}")
        
        # Search for relevant documents
        relevant_chunks = search_documents(question, top_k=TOP_K_RESULTS)
        
        if not relevant_chunks:
            say(text="I couldn't find any relevant information in the Confluence documents. Please try rephrasing your question or check that the documents you're looking for are in the indexed spaces.")
            return
        
        # Extract answer
        answer = extract_answer(question, relevant_chunks)
        
        # Format response
        formatted_response = format_response(question, answer, relevant_chunks)
        
        # Remove typing indicator
        try:
            client.reactions_remove(
                channel=channel_id,
                timestamp=thread_ts,
                name="thinking_face"
            )
        except Exception as e:
            # If we can't remove reactions, just log it and continue
            logger.warning(f"Could not remove reaction: {str(e)}. This is non-critical.")
        
        # Send the response
        send_long_message(say, formatted_response, thread_ts=thread_ts)
    
    except Exception as e:
        logger.error(f"Error processing DM question: {str(e)}")
        say(text="Sorry, I encountered an error while processing your question. Please try again later.")
    
    finally:
        # Remove the thread from active tracking
        manage_active_thread(channel_id, thread_ts, user_id, "remove")

@app.event("app_home_opened")
def update_home_tab(client, event, logger):
    """Update the app home tab when a user opens it"""
    user_id = event["user"]
    
    try:
        # Publish view to Home tab
        client.views_publish(
            user_id=user_id,
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": "Confluence Knowledge Bot",
                            "emoji": True
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Welcome to the Confluence Knowledge Bot! I can help you find information from your Confluence workspace."
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*How to use me:*"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "• Send me a direct message with your question\n• Mention me in a channel with your question\n• Try to be specific in your questions"
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Last knowledge base refresh: {last_refresh_time.strftime('%Y-%m-%d %H:%M:%S') if last_refresh_time else 'Never'}"
                            }
                        ]
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {str(e)}")

# Function to save and restore state (for graceful shutdowns)
def save_state():
    """Save the current state to disk for recovery"""
    with knowledge_base_lock:
        if document_chunks and isinstance(chunk_embeddings, np.ndarray) and len(chunk_embeddings) > 0 and document_metadata:
            try:
                state = {
                    "last_refresh_time": last_refresh_time.isoformat() if last_refresh_time else None,
                }
                with open(os.path.join(CACHE_DIR, "state.json"), 'w') as f:
                    json.dump(state, f)
                logger.info("Saved state to disk")
            except Exception as e:
                logger.error(f"Error saving state: {str(e)}")

def load_state():
    """Load saved state from disk"""
    global last_refresh_time
    
    try:
        state_path = os.path.join(CACHE_DIR, "state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            if state.get("last_refresh_time"):
                last_refresh_time = datetime.fromisoformat(state["last_refresh_time"])
                logger.info(f"Loaded state: last refresh {last_refresh_time}")
    except Exception as e:
        logger.error(f"Error loading state: {str(e)}")

# Add a signal handler for graceful shutdown
def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info("Received shutdown signal, cleaning up...")
    refresh_event.set()  # Signal the background thread to exit
    save_state()
    logger.info("Cleanup complete, exiting")
    sys.exit(0)

def _update_vertex_connector_prompt():
    if vertex_connector:
        vertex_connector.prompt_template = PromptTemplate(
            template="""You are a precise knowledge assistant that delivers focused answers based on Confluence documentation.

Context information:
--------------------------
{context}
--------------------------

Question: {question}

Instructions:
1. Answer ONLY the specific question asked based on the context provided
2. Present information from a SINGLE document perspective when possible
3. Use natural, clear language with a professional tone
4. Focus only on the most relevant details, omitting tangential information
5. If multiple documents are referenced in the context, prioritize the one with the most relevant information
6. If the answer isn't found in the context, say "I don't have specific information on this topic."
7. Format your response as a cohesive, properly structured answer
8. Never reference document names or sources within your answer

Your response should read as a single, authoritative answer from the company's official documentation.
""",
            input_variables=["context", "question"]
        )

# Initialize the app and start it using Socket Mode
if __name__ == "__main__":
    import signal
    import sys
    
    print("\n=== Starting Confluence Knowledge Bot ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    # Check for required environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "CONFLUENCE_URL",
                      "CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    print(f"Vertex AI integration enabled: {USE_VERTEX_AI}")
    if USE_VERTEX_AI:
        print(f"GCP Project ID: {GCP_PROJECT_ID or 'Not set'}")
        print(f"Vertex AI model: {VERTEX_MODEL}")
        
        if not GCP_PROJECT_ID:
            print("WARNING: GCP_PROJECT_ID is not set but USE_VERTEX_AI is enabled")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load previous state
    load_state()
    
    print("Initializing knowledge base...")
    # Ensure the knowledge base is initialized
    initialize_knowledge_base()
    
    if USE_VERTEX_AI and vertex_ai_available:
        if vertex_connector:
            print("✅ Vertex AI integration successfully initialized")
        else:
            print("❌ Vertex AI integration failed to initialize")
    
    # Start the automatic refresh thread
    refresh_event.clear()
    auto_refresh_thread = threading.Thread(target=auto_refresh_knowledge_base, daemon=True)
    auto_refresh_thread.start()
    
    print("Starting Slack bot...")
    logger.info("Starting Confluence Knowledge Bot...")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
