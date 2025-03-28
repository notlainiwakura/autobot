
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
    """Extract a coherent answer with complete context and improved formatting."""
    if not relevant_chunks:
        return "I couldn't find relevant information to answer your question."

    # If Vertex AI integration is enabled and available, use it with improved prompting
    global vertex_connector
    if USE_VERTEX_AI and vertex_ai_available and vertex_connector:
        try:
            logger.info("Using Vertex AI for response generation")
            # Update the prompt template to emphasize completeness
            _update_vertex_connector_prompt()

            # Instead of sending individual chunks, send complete context for more coherence
            enhanced_chunks = enhance_context_completeness(relevant_chunks, query, max_tokens)
            return vertex_connector.generate_response(query, enhanced_chunks)
        except Exception as e:
            logger.error(f"Error using Vertex AI for response: {str(e)}")
            logger.info("Falling back to improved extraction method")

    # Create enhanced context with better completeness
    enhanced_chunks = enhance_context_completeness(relevant_chunks, query, max_tokens)

    # Group chunks by document ID to maintain coherence
    doc_id_to_chunks = {}
    for chunk in enhanced_chunks:
        doc_id = chunk['metadata']['id']
        if doc_id not in doc_id_to_chunks:
            doc_id_to_chunks[doc_id] = []
        doc_id_to_chunks[doc_id].append(chunk)

    # Find document with most chunks
    if not doc_id_to_chunks:
        return "I couldn't find relevant information to answer your question."

    best_doc_id = max(doc_id_to_chunks.keys(), key=lambda x: sum(c['score'] for c in doc_id_to_chunks[x]))
    best_doc_chunks = doc_id_to_chunks[best_doc_id]

    # Sort best document chunks by score
    best_doc_chunks.sort(key=lambda x: x['score'], reverse=True)

    # Combine chunks from best document into single context
    context = "\n\n".join([process_text_formatting(chunk['chunk']) for chunk in best_doc_chunks])

    # Prepare the prompt for Vertex AI model
    system_prompt = f"""
    Answer the following question with complete, detailed instructions. Focus on providing ALL necessary context.
    Use ONLY the information in the context below.
    If the context doesn't contain complete information, acknowledge any gaps rather than skipping details.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    COMPLETENESS GUIDELINES:
    1. Include ALL steps in any procedure, with no missing actions
    2. Explain any technical terms, UI elements, or buttons mentioned (e.g., what exact menu to click)
    3. Make sure instructions can be followed without prior knowledge
    4. If instructions reference specific values, parameters, or settings, include them exactly
    5. If different platforms (Android, iOS, etc.) have different instructions, clearly label each set
    6. Avoid placeholder text like "..." or "[something]" - if you don't have the exact text, say so
    7. Spell out complete URLs, paths, or commands
    8. If steps seem ambiguous or incomplete in the source material, note this explicitly
    """

    if USE_VERTEX_AI and vertex_ai_available:
        try:
            # Direct call to Vertex AI with focused prompt
            response = vertex_connector.llm.invoke(system_prompt)
            if hasattr(response, 'content'):
                return process_text_formatting(response.content.strip())
            elif isinstance(response, str):
                return process_text_formatting(response.strip())
        except Exception as e:
            logger.error(f"Error calling Vertex AI directly: {str(e)}")

    # If Vertex AI isn't available or fails, use improved fallback
    return process_text_formatting(context)


def enhance_context_completeness(chunks, query, max_tokens=1200):
    """Enhance context by fetching additional chunks to ensure completeness."""
    enhanced_chunks = list(chunks)  # Start with existing chunks

    # Extract key entities and concepts from query and existing chunks
    entities = extract_key_entities(query, enhanced_chunks)

    # Track document IDs we already have
    existing_doc_ids = set(chunk['metadata']['id'] for chunk in enhanced_chunks)

    # For each document we already have, try to get more complete context
    for doc_id in existing_doc_ids:
        # Get all chunks from this document, sorted by chunk_index to maintain order
        doc_chunks = [c for c in enhanced_chunks if c['metadata']['id'] == doc_id]
        chunk_indices = [c['metadata'].get('chunk_index', 0) for c in doc_chunks]

        # If we have gaps in the indices, try to fill them
        if len(chunk_indices) > 1:
            min_idx = min(chunk_indices)
            max_idx = max(chunk_indices)

            if max_idx - min_idx + 1 > len(chunk_indices):
                # We have gaps - find missing indices
                existing_indices = set(chunk_indices)
                missing_indices = [i for i in range(min_idx, max_idx + 1) if i not in existing_indices]

                logger.info(f"Found {len(missing_indices)} missing chunks in document {doc_id}")

                # Try to fetch missing chunks for better completeness
                for missing_idx in missing_indices:
                    # Find chunks with this document ID and chunk index
                    for i, metadata in enumerate(document_metadata):
                        if (metadata['id'] == doc_id and
                                metadata.get('chunk_index', 0) == missing_idx):
                            # Add this chunk to enhance context
                            enhanced_chunks.append({
                                'chunk': document_chunks[i],
                                'metadata': metadata,
                                'score': 0.5  # Assign a reasonable score
                            })
                            break

    # Check if we need more context for any detected procedures
    procedure_keywords = ['step', 'guide', 'instruction', 'tutorial', 'how to', 'setup', 'install']
    is_procedure_query = any(keyword in query.lower() for keyword in procedure_keywords)

    if is_procedure_query:
        # This is likely asking for a procedure - ensure we have complete steps
        enhanced_chunks = ensure_complete_procedure(enhanced_chunks, max_tokens)

    # Handle missing context for UI elements and technical terms
    enhanced_chunks = add_missing_term_definitions(enhanced_chunks, entities, max_tokens)

    return enhanced_chunks


def extract_key_entities(query, chunks):
    """Extract key entities and technical terms from the query and chunks."""
    # Simple extraction based on capitalized terms and terms in quotes
    entities = set()

    # Extract capitalized terms and quoted terms from query
    cap_pattern = r'\b[A-Z][a-zA-Z0-9_]+\b'
    quote_pattern = r'["\'](.*?)["\']'

    for match in re.finditer(cap_pattern, query):
        entities.add(match.group(0))

    for match in re.finditer(quote_pattern, query):
        entities.add(match.group(1))

    # Also extract terms that appear in the chunks
    for chunk in chunks:
        for match in re.finditer(cap_pattern, chunk['chunk']):
            entities.add(match.group(0))

        for match in re.finditer(quote_pattern, chunk['chunk']):
            entities.add(match.group(1))

    # Add common technical terms that might need explanation
    tech_terms = ['proxy', 'port', 'IP', 'settings', 'config', 'setup', 'API', 'token', 'auth']
    for term in tech_terms:
        if term.lower() in query.lower():
            entities.add(term)

    return entities


def ensure_complete_procedure(chunks, max_tokens):
    """Ensure procedure steps are complete by checking for numerical sequences."""
    enhanced_chunks = list(chunks)

    # Group chunks by document ID
    doc_chunks = {}
    for chunk in enhanced_chunks:
        doc_id = chunk['metadata']['id']
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append(chunk)

    # For each document, check if we have complete numbered steps
    for doc_id, chunks in doc_chunks.items():
        # Extract all numbered steps from chunks
        step_pattern = r'(?:^|\n)(\s*)(\d+)\.(?:\s+)(.+?)(?:\n|$)'
        steps = []

        for chunk in chunks:
            for match in re.finditer(step_pattern, chunk['chunk']):
                indent, step_num, step_text = match.groups()
                steps.append((int(step_num), step_text.strip()))

        # Sort steps by number
        steps.sort(key=lambda x: x[0])

        # Check for missing steps
        if steps:
            expected_steps = list(range(1, max(step[0] for step in steps) + 1))
            existing_steps = [step[0] for step in steps]
            missing_steps = [step for step in expected_steps if step not in existing_steps]

            if missing_steps:
                logger.info(f"Found missing steps {missing_steps} in document {doc_id}")

                # If we have missing steps, try to fetch more chunks from this document
                for i, metadata in enumerate(document_metadata):
                    if metadata['id'] == doc_id and i < len(document_chunks):
                        # Check if this chunk has any of the missing steps
                        chunk_text = document_chunks[i]
                        has_missing_step = False

                        for step in missing_steps:
                            step_regex = r'(?:^|\n)\s*' + str(step) + r'\.(?:\s+).+?(?:\n|$)'
                            if re.search(step_regex, chunk_text):
                                has_missing_step = True
                                break

                        if has_missing_step:
                            # Add this chunk to enhance context
                            enhanced_chunks.append({
                                'chunk': chunk_text,
                                'metadata': metadata,
                                'score': 0.5  # Assign a reasonable score
                            })

    return enhanced_chunks


def add_missing_term_definitions(chunks, entities, max_tokens):
    """Add chunks that might define technical terms found in the query and chunks."""
    enhanced_chunks = list(chunks)
    current_tokens = sum(count_tokens(chunk['chunk']) for chunk in enhanced_chunks)

    # Track document IDs we already have
    existing_doc_ids = set(chunk['metadata']['id'] for chunk in enhanced_chunks)

    # For each technical term, try to find definitions
    for entity in entities:
        if current_tokens >= max_tokens:
            break

        # Skip very short terms to avoid noise
        if len(entity) < 3:
            continue

        # Create a search pattern for definitions
        definition_patterns = [
            rf"\b{re.escape(entity)}\b.*?\bis\b",
            rf"\b{re.escape(entity)}\b.*?\bmeans\b",
            rf"\b{re.escape(entity)}\b.*?\brefers to\b",
            rf"\b{re.escape(entity)}\b.*?:"
        ]

        # Search through document chunks for definitions
        for i, chunk_text in enumerate(document_chunks):
            # Skip chunks from documents we already have
            if document_metadata[i]['id'] in existing_doc_ids:
                continue

            # Check if chunk might contain a definition
            has_definition = False
            for pattern in definition_patterns:
                if re.search(pattern, chunk_text, re.IGNORECASE):
                    has_definition = True
                    break

            # If term appears multiple times, it might be important context
            term_count = len(re.findall(rf"\b{re.escape(entity)}\b", chunk_text, re.IGNORECASE))
            if term_count >= 3:
                has_definition = True

            if has_definition:
                # Add this chunk to enhance context
                enhanced_chunks.append({
                    'chunk': chunk_text,
                    'metadata': document_metadata[i],
                    'score': 0.4  # Slightly lower score for definitions
                })

                current_tokens += count_tokens(chunk_text)
                if current_tokens >= max_tokens:
                    break

    return enhanced_chunks


def process_text_formatting(text):
    """Clean up formatting issues in text."""
    # Replace multiple newlines with double newlines
    processed = re.sub(r'\n{3,}', '\n\n', text)

    # Fix list formatting
    lines = processed.split('\n')
    processed_lines = []
    in_list = False
    list_indent = 0
    current_section = None

    for i, line in enumerate(lines):
        # Check for section headers and maintain them
        header_match = re.match(r'^(#+)\s+(.+)$', line)
        if header_match:
            current_section = line
            processed_lines.append(line)
            in_list = False
            continue

        # Check if line is a list item
        list_match = re.match(r'^(\s*)(\d+)\.(\s*)(.+)$', line)
        if list_match:
            # This is a list item, ensure proper spacing
            spaces, number, existing_space, content = list_match.groups()
            proper_space = ' ' if not existing_space else existing_space
            processed_lines.append(f"{spaces}{number}.{proper_space}{content}")
            in_list = True
            list_indent = len(spaces)
        elif in_list and line.strip() and i > 0:
            # This might be a continuation of a list item
            if not re.search(r'[.:]$', lines[i - 1]) and not line.startswith(' ' * (list_indent + 2)):
                # Add indentation for continuation text
                processed_lines.append(' ' * (list_indent + 2) + line)
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)
            if not line.strip():
                in_list = False

    # Join lines back together
    processed = '\n'.join(processed_lines)

    # Fix markdown formatting
    processed = re.sub(r'\*\*\s+', '**', processed)  # Fix bold text spacing
    processed = re.sub(r'\s+\*\*', '**', processed)

    # Fix broken links
    processed = re.sub(r'\[([^\]]+)\]\s+\(([^)]+)\)', r'[\1](\2)', processed)

    # Fix incomplete placeholders (empty quotes or brackets)
    processed = re.sub(r'["\']\s*["\']', '[unknown value]', processed)
    processed = re.sub(r'\[\s*\]', '[unknown value]', processed)

    # Add clarification for potential UI element abbreviations
    ui_element_pattern = r'\b([A-Z][A-Z0-9_]{1,5})\b'

    def replace_ui_abbreviation(match):
        abbr = match.group(1)
        # Don't replace common acronyms
        if abbr in ['UI', 'API', 'URL', 'IP', 'ID', 'OS', 'UI', 'PC']:
            return abbr
        return f"{abbr} button"  # Assume it's a UI element

    # Find potential UI element abbreviations that might be unclear
    processed = re.sub(ui_element_pattern, replace_ui_abbreviation, processed)

    return processed

def _update_vertex_connector_prompt():
    """Update the Vertex AI prompt template to emphasize completeness and clarity."""
    if vertex_connector:
        vertex_connector.prompt_template = PromptTemplate(
            template="""You are a detailed technical guide that provides complete, clear instructions from Confluence documentation.

Context information:
--------------------------
{context}
--------------------------

Question: {question}

INSTRUCTIONS:
1. Answer the question using ONLY information from the context
2. Provide COMPREHENSIVE, STEP-BY-STEP instructions that leave nothing to interpretation:
   - Include EVERY step needed to complete the task
   - When mentioning buttons, menus, or UI elements, describe EXACTLY where they are located
   - Include all specific values, IP addresses, port numbers, and parameters exactly as given
   - For each platform (Android, iOS, Windows, etc.), provide the COMPLETE set of instructions
3. Format your answer professionally:
   - Use proper numbered lists for steps
   - Group related steps under clear headings
   - Maintain consistent indentation and formatting
4. If specific details are missing from the context, explicitly state what information is unavailable
5. If steps seem ambiguous in the source material, provide the most likely interpretation and note any uncertainty
6. Never omit details or use placeholder text (like "..." or "[something]")
7. Do not reference document names or sources within your answer

Your response must be a self-contained guide that someone could follow successfully without additional information.
""",
            input_variables=["context", "question"]
        )

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
        answer = extract_answer_with_structure(question, relevant_chunks)
        
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


def extract_answer_with_structure(query, relevant_chunks, max_tokens=1200):
    """
    Extract an answer with improved content structure, clear section headers,
    and smooth transitions between different parts.
    """
    if not relevant_chunks:
        return "I couldn't find relevant information to answer your question."

    # Process and categorize the content for better structure
    structured_content = structure_content_by_category(query, relevant_chunks)

    # If Vertex AI integration is enabled and available, use it with structured prompting
    global vertex_connector
    if USE_VERTEX_AI and vertex_ai_available and vertex_connector:
        try:
            logger.info("Using Vertex AI for response generation with structured prompting")

            # Update prompt template to emphasize structure
            _update_vertex_connector_prompt_for_structure()

            # Generate response with structured content
            response = vertex_connector.generate_response(query, structured_content['chunks'])

            # Apply post-processing to ensure structural integrity
            final_response = enhance_response_structure(response, structured_content['outline'])

            return final_response

        except Exception as e:
            logger.error(f"Error using Vertex AI with structured prompting: {str(e)}")
            logger.info("Falling back to manual structure generation")

            # Fallback to manual generation of structured response
            return generate_structured_response(query, structured_content)

    # Fallback to manual generation if Vertex AI isn't available
    return generate_structured_response(query, structured_content)


def structure_content_by_category(query, chunks):
    """
    Analyze chunks and categorize them for structured presentation.
    Returns a dictionary with categorized content and a suggested outline.
    """
    # Initialize structure
    structured_content = {
        'chunks': [],  # Will store processed chunks
        'categories': {},  # Will store content by category
        'outline': {},  # Will store suggested section outline
        'sources': set(),  # Will track unique source documents
        'platforms': set(),  # Will track platform-specific content (iOS, Android, etc.)
        'has_procedure': False,  # Flag if content contains procedural steps
        'query_type': determine_query_type(query)
    }

    # Identify the type of content needed based on query
    query_keywords = extract_query_keywords(query)

    # First pass: identify categories and platforms
    categorize_content(chunks, structured_content, query_keywords)

    # Second pass: arrange chunks based on categories and create logical structure
    create_content_structure(chunks, structured_content)

    # Create transition text between sections
    add_section_transitions(structured_content)

    return structured_content


def determine_query_type(query):
    """Determine the type of query to guide response structure."""
    query_lower = query.lower()

    if any(kw in query_lower for kw in ['how to', 'steps', 'procedure', 'guide', 'tutorial', 'setup', 'configure']):
        return 'procedure'
    elif any(kw in query_lower for kw in ['what is', 'definition', 'explain', 'describe', 'tell me about']):
        return 'explanation'
    elif any(kw in query_lower for kw in ['compare', 'difference', 'versus', 'vs', 'pros and cons']):
        return 'comparison'
    elif any(kw in query_lower for kw in ['troubleshoot', 'fix', 'solve', 'issue', 'error', 'problem']):
        return 'troubleshooting'
    else:
        return 'general'


def extract_query_keywords(query):
    """Extract important keywords from the query."""
    # Remove common stop words
    stop_words = set(['a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'for', 'to', 'with', 'by'])

    # Tokenize and extract important words
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    # Extract phrases that might be important (e.g., "Charles Proxy")
    phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', query)

    return set(keywords + phrases)


def categorize_content(chunks, structured_content, query_keywords):
    """Categorize content chunks based on their content."""
    # Platform detection patterns
    platform_patterns = {
        'android': r'\bandroid\b',
        'ios': r'\bios\b|\biphone\b|\bipad\b|\bapple\s+device',
        'windows': r'\bwindows\b',
        'mac': r'\bmac\b|\bmacos\b|\bosx\b',
        'linux': r'\blinux\b|\bubuntu\b|\bdebian\b',
        'web': r'\bweb\b|\bbrowser\b|\bchrome\b|\bfirefox\b|\bsafari\b'
    }

    # Content category patterns
    category_patterns = {
        'prerequisites': r'\bprerequisite|\brequirement|\bbefore\s+you\s+begin|\bneeded\b',
        'installation': r'\binstall|\bdownload|\bsetup\b|\bconfiguration\b',
        'basic_usage': r'\bbasic\s+usage|\bget\s+started|\bquick\s+start|\bintroduction\b',
        'advanced_usage': r'\badvanced|\bexpert|\bcustom\b',
        'troubleshooting': r'\btroubleshoot|\bissue|\berror|\bproblem|\bdebug\b|\bfix\b',
        'reference': r'\breference|\bapi|\bcommand|\bparameter\b|\boption\b'
    }

    # Check each chunk for categories and platforms
    for chunk in chunks:
        chunk_text = chunk['chunk'].lower()
        chunk_categories = set()
        chunk_platforms = set()

        # Check for platforms
        for platform, pattern in platform_patterns.items():
            if re.search(pattern, chunk_text, re.IGNORECASE):
                chunk_platforms.add(platform)
                structured_content['platforms'].add(platform)

        # Check for content categories
        for category, pattern in category_patterns.items():
            if re.search(pattern, chunk_text, re.IGNORECASE):
                chunk_categories.add(category)

        # Check for procedural content (numbered steps)
        if re.search(r'(?:^|\n)\s*\d+\.\s+', chunk['chunk']):
            chunk_categories.add('procedure')
            structured_content['has_procedure'] = True

        # Add source document
        source_doc = chunk['metadata'].get('title', 'Unknown')
        structured_content['sources'].add(source_doc)

        # Store categorization with the chunk
        chunk_info = chunk.copy()
        chunk_info['categories'] = chunk_categories
        chunk_info['platforms'] = chunk_platforms

        # Calculate relevance boost for chunks that match query keywords
        keyword_matches = sum(1 for kw in query_keywords if kw in chunk_text)
        chunk_info['keyword_relevance'] = keyword_matches / len(query_keywords) if query_keywords else 0

        structured_content['chunks'].append(chunk_info)

        # Add to category dictionary
        for category in chunk_categories:
            if category not in structured_content['categories']:
                structured_content['categories'][category] = []
            structured_content['categories'][category].append(chunk_info)


def create_content_structure(chunks, structured_content):
    """Create a logical content structure based on categories and query type."""
    query_type = structured_content['query_type']
    categories = structured_content['categories']
    platforms = structured_content['platforms']

    # Initialize outline with an introduction
    outline = {'introduction': 'Introduction'}

    # Build appropriate structure based on query type
    if query_type == 'procedure':
        # For procedures, organize by step sequence
        if 'prerequisites' in categories:
            outline['prerequisites'] = 'Prerequisites'

        if 'installation' in categories:
            outline['installation'] = 'Installation and Setup'

        outline['procedure'] = 'Step-by-Step Guide'

        # If multiple platforms, create subsections for each
        if len(platforms) > 1:
            for platform in platforms:
                platform_key = f"procedure_{platform}"
                platform_name = platform.capitalize()
                outline[platform_key] = f"Steps for {platform_name}"

        if 'troubleshooting' in categories:
            outline['troubleshooting'] = 'Troubleshooting Common Issues'

        outline['verification'] = 'Verification and Next Steps'

    elif query_type == 'explanation':
        # For explanations, organize by concept
        outline['overview'] = 'Overview'

        if 'basic_usage' in categories:
            outline['basic_usage'] = 'Basic Usage'

        if 'advanced_usage' in categories:
            outline['advanced_usage'] = 'Advanced Features'

        if 'reference' in categories:
            outline['reference'] = 'Technical Reference'

    elif query_type == 'comparison':
        # For comparisons, organize by comparison points
        outline['overview'] = 'Overview of Options'

        # If comparing platforms, create sections for each
        if len(platforms) > 1:
            for platform in platforms:
                platform_key = f"comparison_{platform}"
                platform_name = platform.capitalize()
                outline[platform_key] = f"{platform_name} Features"

        outline['summary'] = 'Summary and Recommendations'

    elif query_type == 'troubleshooting':
        # For troubleshooting, organize by issue and solution
        outline['problem_overview'] = 'Problem Overview'

        if 'basic_usage' in categories:
            outline['basic_checks'] = 'Basic Checks'

        outline['solutions'] = 'Solutions'

        # If platform-specific issues, create subsections
        if len(platforms) > 1:
            for platform in platforms:
                platform_key = f"solutions_{platform}"
                platform_name = platform.capitalize()
                outline[platform_key] = f"Solutions for {platform_name}"

        outline['prevention'] = 'Prevention Tips'

    else:  # general query
        # For general queries, use a simple structure
        outline['overview'] = 'Overview'

        if structured_content['has_procedure']:
            outline['procedure'] = 'Procedures'

        if len(platforms) > 1:
            outline['platforms'] = 'Platform-Specific Information'

        if 'reference' in categories:
            outline['reference'] = 'Reference Information'

    # Add conclusion
    outline['conclusion'] = 'Conclusion'

    # Store the outline
    structured_content['outline'] = outline


def add_section_transitions(structured_content):
    """Add transition text between sections for smoother flow."""
    outline = structured_content['outline']
    query_type = structured_content['query_type']

    transitions = {}

    # Create appropriate transitions based on query type and outline
    if query_type == 'procedure':
        # Transitions for procedural content
        if 'prerequisites' in outline and 'installation' in outline:
            transitions[
                'prerequisites_to_installation'] = "Once you have all the prerequisites ready, let's proceed with the installation process."

        if 'installation' in outline and 'procedure' in outline:
            transitions[
                'installation_to_procedure'] = "After completing the installation, follow these steps to use the software:"

        # For multi-platform content
        platforms = structured_content['platforms']
        if len(platforms) > 1:
            for i, platform in enumerate(platforms):
                next_platform = list(platforms)[i + 1] if i < len(platforms) - 1 else None
                if next_platform:
                    transitions[
                        f"procedure_{platform}_to_procedure_{next_platform}"] = f"If you're using {next_platform.capitalize()} instead, follow these steps:"

        if 'procedure' in outline and 'troubleshooting' in outline:
            transitions[
                'procedure_to_troubleshooting'] = "If you encounter any issues during this process, refer to these troubleshooting tips:"

        if 'troubleshooting' in outline and 'verification' in outline:
            transitions[
                'troubleshooting_to_verification'] = "After resolving any issues, here's how to verify everything is working correctly:"

    elif query_type == 'explanation':
        # Transitions for explanatory content
        if 'overview' in outline and 'basic_usage' in outline:
            transitions['overview_to_basic_usage'] = "Let's look at how to use this in basic scenarios:"

        if 'basic_usage' in outline and 'advanced_usage' in outline:
            transitions[
                'basic_usage_to_advanced_usage'] = "Once you're comfortable with the basics, you can explore these advanced features:"

        if 'advanced_usage' in outline and 'reference' in outline:
            transitions[
                'advanced_usage_to_reference'] = "For complete technical details, refer to this reference information:"

    elif query_type == 'comparison':
        # Transitions for comparison content
        if 'overview' in outline:
            platforms = structured_content['platforms']
            if len(platforms) > 1:
                transitions['overview_to_comparison'] = "Let's compare the key features across different platforms:"

        # Add transitions between platform comparisons
        platforms = structured_content['platforms']
        for i, platform in enumerate(platforms):
            next_platform = list(platforms)[i + 1] if i < len(platforms) - 1 else None
            if next_platform:
                transitions[
                    f"comparison_{platform}_to_comparison_{next_platform}"] = f"Now, let's examine the features for {next_platform.capitalize()}:"

        if any(k.startswith('comparison_') for k in outline) and 'summary' in outline:
            transitions['comparison_to_summary'] = "To summarize the key differences and make recommendations:"

    # Add a final transition to conclusion for all types
    last_section = list(outline.keys())[-2]  # -2 because the last one is conclusion
    transitions[f"{last_section}_to_conclusion"] = "In conclusion:"

    # Store the transitions
    structured_content['transitions'] = transitions


def generate_structured_response(query, structured_content):
    """
    Generate a structured response based on the content structure.
    This is used as a fallback when Vertex AI is not available.
    """
    outline = structured_content['outline']
    transitions = structured_content['transitions']
    chunks = structured_content['chunks']

    # Sort chunks by relevance score
    sorted_chunks = sorted(chunks, key=lambda x: (x['score'] + x.get('keyword_relevance', 0)), reverse=True)

    # Initialize response sections
    sections = {}
    for section_key, section_title in outline.items():
        sections[section_key] = {
            'title': section_title,
            'content': []
        }

    # Fill in content for each section based on relevance and categories
    for section_key in outline.keys():
        # Skip introduction and conclusion - we'll generate these separately
        if section_key in ['introduction', 'conclusion']:
            continue

        # Find chunks relevant to this section
        relevant_chunks = []

        # For procedure sections
        if section_key == 'procedure' or section_key.startswith('procedure_'):
            relevant_chunks = [c for c in sorted_chunks if 'procedure' in c['categories']]

            # For platform-specific procedures
            if section_key.startswith('procedure_'):
                platform = section_key.split('_')[1]
                relevant_chunks = [c for c in relevant_chunks if platform in c['platforms']]

        # For other specific sections
        elif section_key in structured_content['categories']:
            relevant_chunks = structured_content['categories'][section_key]

        # For platform-specific sections
        elif '_' in section_key:
            category, platform = section_key.split('_')
            if category in structured_content['categories']:
                category_chunks = structured_content['categories'][category]
                relevant_chunks = [c for c in category_chunks if platform in c['platforms']]

        # For general sections, use most relevant chunks not yet assigned
        else:
            # Get chunks that haven't been assigned to other sections
            assigned_chunks = set()
            for s_key, s_data in sections.items():
                if s_key != section_key:
                    assigned_chunks.update(id(c) for c in s_data['content'])

            relevant_chunks = [c for c in sorted_chunks if id(c) not in assigned_chunks][:3]

        # Add chunks to section
        sections[section_key]['content'] = relevant_chunks

    # Generate introduction
    query_type = structured_content['query_type']
    sources = structured_content['sources']
    source_text = list(sources)[0] if sources else "our documentation"

    intro_text = ""
    if query_type == 'procedure':
        intro_text = f"Here's a complete guide on {query} based on {source_text}. Follow these instructions carefully for best results."
    elif query_type == 'explanation':
        intro_text = f"Let me explain {query} based on information from {source_text}."
    elif query_type == 'comparison':
        intro_text = f"Here's a detailed comparison for {query} based on {source_text}."
    elif query_type == 'troubleshooting':
        intro_text = f"Here are solutions for {query} based on {source_text}."
    else:
        intro_text = f"Here's the information about {query} from {source_text}."

    sections['introduction']['content'] = [{'chunk': intro_text, 'score': 1.0}]

    # Generate conclusion
    conclusion_text = "This information should help you with " + query + ". "
    if query_type == 'procedure':
        conclusion_text += "By following these steps carefully, you should be able to complete the process successfully."
    elif query_type == 'troubleshooting':
        conclusion_text += "If you continue to experience issues after trying these solutions, please contact technical support for further assistance."
    else:
        conclusion_text += "For more detailed information, refer to the complete documentation."

    sections['conclusion']['content'] = [{'chunk': conclusion_text, 'score': 1.0}]

    # Build final response with sections and transitions
    response = ""
    section_keys = list(outline.keys())

    for i, section_key in enumerate(section_keys):
        section = sections[section_key]

        # Add section header
        if section_key != 'introduction':  # Skip header for introduction
            response += f"\n\n## {section['title']}\n\n"

        # Add section content
        if section_key in ['introduction', 'conclusion']:
            # For intro and conclusion, use our generated text
            response += section['content'][0]['chunk']
        else:
            # For other sections, use relevant chunks
            if section['content']:
                # Combine chunks into coherent text
                combined_text = combine_chunks_to_coherent_text(section['content'], section_key)
                response += combined_text
            else:
                # If no content, add placeholder
                response += f"Information about {section['title']} is not available in the current documentation."

        # Add transition to next section if available
        if i < len(section_keys) - 1:
            next_section_key = section_keys[i + 1]
            transition_key = f"{section_key}_to_{next_section_key}"

            if transition_key in transitions:
                response += f"\n\n{transitions[transition_key]}"

    # Clean up formatting
    return process_text_formatting(response)


def combine_chunks_to_coherent_text(chunks, section_key):
    """Combine chunks into coherent text for a section."""
    # Sort chunks by score
    sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)

    # For procedural sections, try to maintain numbered steps
    if section_key == 'procedure' or section_key.startswith('procedure_'):
        # Extract and organize numbered steps
        steps = extract_numbered_steps(sorted_chunks)

        if steps:
            # Format the steps in order
            step_text = "\n\n"
            current_step = 1

            for step_num, step_content in sorted(steps.items()):
                # If there's a gap in numbering, add a note
                if step_num > current_step:
                    step_text += f"_(Note: Steps {current_step} to {step_num - 1} are not available in the documentation)_\n\n"

                step_text += f"{step_num}. {step_content}\n\n"
                current_step = step_num + 1

            return step_text

    # For other sections, combine chunks with proper transitions
    combined_text = ""

    for i, chunk in enumerate(sorted_chunks):
        chunk_text = chunk['chunk'].strip()

        # Skip if this chunk is too similar to already included content
        if i > 0 and is_content_redundant(chunk_text, combined_text):
            continue

        # Add appropriate separator based on content
        if combined_text:
            # Check if this chunk appears to continue from previous content
            if is_continuation(sorted_chunks[i - 1]['chunk'], chunk_text):
                combined_text += " " + chunk_text
            else:
                combined_text += "\n\n" + chunk_text
        else:
            combined_text = chunk_text

    return combined_text


def extract_numbered_steps(chunks):
    """Extract numbered steps from chunks, organizing them by step number."""
    steps = {}

    # Step pattern matching
    step_pattern = r'(?:^|\n)\s*(\d+)\.(?:\s+)(.+?)(?=(?:\n\s*\d+\.)|$)'

    for chunk in chunks:
        chunk_text = chunk['chunk']

        # Find all numbered steps in this chunk
        for match in re.finditer(step_pattern, chunk_text, re.DOTALL):
            step_num = int(match.group(1))
            step_content = match.group(2).strip()

            # Only add if we don't already have this step or if the new content is better
            if step_num not in steps or len(step_content) > len(steps[step_num]):
                steps[step_num] = step_content

    return steps


def is_content_redundant(new_text, existing_text):
    """Check if new text is redundant with existing content."""
    # Simple redundancy check - if 70% of sentences are already present
    new_sentences = set(s.strip() for s in re.split(r'[.!?]', new_text) if s.strip())
    existing_sentences = set(s.strip() for s in re.split(r'[.!?]', existing_text) if s.strip())

    overlap_count = len(new_sentences.intersection(existing_sentences))
    redundancy_ratio = overlap_count / len(new_sentences) if new_sentences else 0

    return redundancy_ratio > 0.7


def is_continuation(previous_text, current_text):
    """Check if current text appears to be a continuation of previous text."""
    # Check if previous text ends mid-sentence
    if not previous_text.strip().endswith(('.', '!', '?', ':', ';')):
        return True

    # Check if current text starts with lowercase or connecting words
    if re.match(r'^\s*[a-z]', current_text) or re.match(
            r'^\s*(and|or|but|however|therefore|thus|moreover|furthermore|additionally)', current_text, re.IGNORECASE):
        return True

    return False


def _update_vertex_connector_prompt_for_structure():
    """Update the Vertex AI prompt template to emphasize structure and transitions."""
    if vertex_connector:
        vertex_connector.prompt_template = PromptTemplate(
            template="""You are an expert technical writer creating highly structured documentation from Confluence sources.

Context information:
--------------------------
{context}
--------------------------

Question: {question}

I NEED YOU TO CREATE A PERFECTLY STRUCTURED RESPONSE:

CONTENT STRUCTURE REQUIREMENTS:
1. Organize your response with CLEAR SECTION HEADERS (##) for each main section
2. Begin with a brief introduction (no header needed for this section)
3. Arrange content in a logical, progressive order based on complexity or sequence
4. When combining information from different sources:
   * Use section headers to clearly delineate different topics
   * Add smooth transitions between sections to maintain flow
   * Ensure ideas connect logically from one section to the next
5. End with a concise conclusion section that summarizes key points

SECTION TRANSITIONS:
1. Add explicit transition sentences between major sections
2. Use connective phrases like "Now that we've covered X, let's look at Y"
3. Show the relationship between sections (e.g., "Building on these concepts...")
4. For sequential procedures, use clear sequence indicators (First, Next, Finally)
5. When switching between platforms or approaches, clearly signal the change

FORMATTING AND CLARITY:
1. Use consistent formatting throughout the entire response
2. Maintain proper list formatting and numbering across sections
3. Use subsections (###) for complex topics that need further organization
4. Format technical elements consistently (code, commands, parameters)
5. For multi-platform instructions, clearly label each platform section

Your response should read as ONE COHESIVE DOCUMENT with perfect flow between sections, not as fragments from different sources.
""",
            input_variables=["context", "question"]
        )


def enhance_response_structure(response, outline):
    """
    Enhance the structure of a response from Vertex AI by adding missing
    section headers and transitions if needed.
    """
    if not response:
        return "I couldn't generate a proper response based on the available information."

    # Check if the response already has section headers
    has_headers = bool(re.search(r'^\s*#{2,3}\s+.+$', response, re.MULTILINE))

    # If no headers and we have multiple sections in our outline, add them
    if not has_headers and len(outline) > 3:  # More than intro, one section, conclusion
        structured_response = ""
        section_keys = list(outline.keys())

        # Extract paragraphs from response
        paragraphs = re.split(r'\n{2,}', response)

        # Assign paragraphs to sections based on content analysis
        current_section_idx = 0
        for i, paragraph in enumerate(paragraphs):
            # Skip empty paragraphs
            if not paragraph.strip():
                continue

            # First paragraph is introduction (no header)
            if i == 0:
                structured_response += paragraph + "\n\n"
                current_section_idx = 1
                continue

            # Last paragraph is conclusion
            if i == len(paragraphs) - 1 and len(paragraphs) > 3:
                structured_response += f"\n\n## {outline['conclusion']}\n\n{paragraph}"
                break

            # Middle paragraphs get appropriate section headers
            if current_section_idx < len(section_keys) - 1:  # Skip intro and conclusion
                section_key = section_keys[current_section_idx]
                if section_key not in ['introduction', 'conclusion']:
                    structured_response += f"\n\n## {outline[section_key]}\n\n{paragraph}"
                    current_section_idx += 1
                else:
                    # Skip to the next non-intro/conclusion section
                    while (current_section_idx < len(section_keys) and
                           section_keys[current_section_idx] in ['introduction', 'conclusion']):
                        current_section_idx += 1

                    if current_section_idx < len(section_keys):
                        section_key = section_keys[current_section_idx]
                        structured_response += f"\n\n## {outline[section_key]}\n\n{paragraph}"
                        current_section_idx += 1
                    else:
                        # If we've run out of sections, add without header
                        structured_response += "\n\n" + paragraph
            else:
                # If we've run out of sections, add without header
                structured_response += "\n\n" + paragraph

        return structured_response
    else:
        # Response already has good structure or is too short to need it
        return response

def extract_answer_with_enhanced_prompting(query, relevant_chunks, max_tokens=1200):
    """
    Extract an answer with enhanced prompting strategies for Vertex AI.
    Includes pre-processing context and post-processing responses.
    """
    if not relevant_chunks:
        return "I couldn't find relevant information to answer your question."

    # Pre-process context to improve prompt effectiveness
    processed_chunks = preprocess_context_for_prompting(relevant_chunks)

    # If Vertex AI integration is enabled and available, use it with enhanced prompting
    global vertex_connector
    if USE_VERTEX_AI and vertex_ai_available and vertex_connector:
        try:
            logger.info("Using Vertex AI for response generation with enhanced prompting")

            # Apply advanced prompt engineering techniques
            _update_vertex_connector_prompt()

            # For highly technical or procedural questions, add specific meta-prompts
            if is_procedural_question(query):
                add_procedural_meta_instructions()
            elif is_comparison_question(query):
                add_comparison_meta_instructions()
            elif is_troubleshooting_question(query):
                add_troubleshooting_meta_instructions()

            # Generate the response
            raw_response = vertex_connector.generate_response(query, processed_chunks)

            # Post-process the response to fix any remaining formatting issues
            final_response = postprocess_vertex_response(raw_response)

            return final_response

        except Exception as e:
            logger.error(f"Error using Vertex AI with enhanced prompting: {str(e)}")
            logger.info("Falling back to standard extraction method")

    # Fallback to standard extraction if Vertex AI isn't available
    return extract_answer(query, relevant_chunks, max_tokens)


def preprocess_context_for_prompting(chunks):
    """
    Preprocess context chunks to optimize them for the prompt.
    Formats the context to emphasize important information and structure.
    """
    processed_chunks = []

    # Group chunks by document for coherence
    doc_id_to_chunks = {}
    for chunk in chunks:
        doc_id = chunk['metadata']['id']
        if doc_id not in doc_id_to_chunks:
            doc_id_to_chunks[doc_id] = []
        doc_id_to_chunks[doc_id].append(chunk)

    # Process each document's chunks
    for doc_id, doc_chunks in doc_id_to_chunks.items():
        # Sort chunks by relevance score
        sorted_chunks = sorted(doc_chunks, key=lambda x: x['score'], reverse=True)

        # Get document title from metadata
        title = sorted_chunks[0]['metadata'].get('title', 'Document')

        # Create a document section with relevant metadata
        doc_section = {
            'metadata': sorted_chunks[0]['metadata'].copy(),
            'chunk': f"## {title}\n\n"
        }

        # Combine chunks with proper formatting
        for chunk in sorted_chunks:
            # Clean up the chunk text
            chunk_text = process_chunk_for_prompting(chunk['chunk'])
            doc_section['chunk'] += chunk_text + "\n\n"

        # Add the score from the highest-scoring chunk
        doc_section['score'] = max(chunk['score'] for chunk in sorted_chunks)

        processed_chunks.append(doc_section)

    # Sort the processed document sections by score
    return sorted(processed_chunks, key=lambda x: x['score'], reverse=True)


def process_chunk_for_prompting(text):
    """
    Process a chunk of text to enhance its formatting for prompting.
    Highlights key elements that should be preserved.
    """
    # Highlight code blocks, commands, and technical syntax
    processed = text

    # Ensure code blocks are properly formatted
    code_block_pattern = r'```(?:\w+)?\n(.*?)\n```'
    processed = re.sub(code_block_pattern, r'```\n\1\n```', processed, flags=re.DOTALL)

    # Highlight UI navigation with bold
    ui_patterns = [
        (r'click (?:on )?(?:the )?(["\']?(?:[A-Z][a-z]+ )*(?:button|tab|menu|icon|link|option)["\']?)',
         r'click on the **\1**'),
        (r'select (?:the )?(["\']?(?:[A-Z][a-z]+ )*(?:option|item|tab|menu)["\']?)', r'select the **\1**'),
        (r'go to (?:the )?(["\']?(?:[A-Z][a-z]+ )*(?:tab|section|page|screen|menu)["\']?)', r'go to the **\1**')
    ]

    for pattern, replacement in ui_patterns:
        processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)

    # Ensure proper list formatting
    lines = processed.split('\n')
    for i in range(len(lines)):
        # Fix numbered list items
        list_match = re.match(r'^(\s*)(\d+)\.(\s*)(.+)$', lines[i])
        if list_match:
            spaces, number, existing_space, content = list_match.groups()
            proper_space = ' ' if not existing_space else existing_space
            lines[i] = f"{spaces}{number}.{proper_space}{content}"

    return '\n'.join(lines)


def postprocess_vertex_response(response):
    """
    Post-process the Vertex AI response to fix any remaining formatting issues.
    """
    if not response:
        return "I couldn't generate a proper response based on the available information."

    # Fix list formatting issues
    processed = process_text_formatting(response)

    # Ensure headings have proper spacing
    processed = re.sub(r'(?<!\n)\n(#+\s+)', r'\n\n\1', processed)
    processed = re.sub(r'(#+\s+.+)\n(?!\n)', r'\1\n\n', processed)

    # Fix any broken markdown links
    processed = re.sub(r'\[([^\]]+)\]\s+\(([^)]+)\)', r'[\1](\2)', processed)

    # Fix code block formatting
    processed = re.sub(r'```\s+([a-zA-Z0-9]+)', r'```\1', processed)

    # Final check for overall structure
    if not re.search(r'^\s*#+\s+', processed) and len(processed.split('\n\n')) > 3:
        # Long response without headings - add a title based on the question
        processed = f"# Response\n\n{processed}"

    return processed


def is_procedural_question(query):
    """Check if the query is asking for a procedure or how-to."""
    procedural_patterns = [
        r'how\s+to\s+',
        r'steps?\s+to\s+',
        r'guide\s+for\s+',
        r'process\s+for\s+',
        r'instructions?\s+for\s+',
        r'set\s*up',
        r'configure',
        r'install',
        r'implement'
    ]

    return any(re.search(pattern, query, re.IGNORECASE) for pattern in procedural_patterns)


def is_comparison_question(query):
    """Check if the query is asking for a comparison."""
    comparison_patterns = [
        r'compare',
        r'difference\s+between',
        r'vs\.?',
        r'versus',
        r'similarities',
        r'pros\s+and\s+cons'
    ]

    return any(re.search(pattern, query, re.IGNORECASE) for pattern in comparison_patterns)


def is_troubleshooting_question(query):
    """Check if the query is about troubleshooting or error resolution."""
    troubleshooting_patterns = [
        r'troubleshoot',
        r'debug',
        r'fix',
        r'solve',
        r'resolve',
        r'error',
        r'issue',
        r'problem',
        r'not\s+working',
        r'fails?',
        r'broken'
    ]

    return any(re.search(pattern, query, re.IGNORECASE) for pattern in troubleshooting_patterns)


def add_procedural_meta_instructions():
    """Add meta-instructions optimized for procedural questions."""
    if vertex_connector:
        current_template = vertex_connector.prompt_template.template

        procedural_instructions = """
PROCEDURAL RESPONSE REQUIREMENTS:
1. Format the response as a clear step-by-step guide
2. Number all steps sequentially and consistently
3. For each step:
   - Begin with a specific action verb (Click, Enter, Navigate, etc.)
   - Include only ONE primary action per numbered step
   - Add substeps with bullets for related actions within a step
4. Include setup prerequisites before the main steps
5. Add verification steps after critical actions
6. End with a "Verification" section explaining how to confirm success
"""

        # Insert procedural instructions before "Your answer should be"
        updated_template = current_template.replace(
            "Your answer should be complete and ready",
            f"{procedural_instructions}\n\nYour answer should be complete and ready"
        )

        vertex_connector.prompt_template = PromptTemplate(
            template=updated_template,
            input_variables=["context", "question"]
        )


def add_comparison_meta_instructions():
    """Add meta-instructions optimized for comparison questions."""
    if vertex_connector:
        current_template = vertex_connector.prompt_template.template

        comparison_instructions = """
COMPARISON RESPONSE REQUIREMENTS:
1. Structure the response with clear categories for comparison
2. Use a consistent format throughout (either feature-by-feature OR option-by-option)
3. Include a brief introduction explaining what's being compared
4. For feature-by-feature comparison:
   - Use level 2 headings (##) for each feature category
   - Describe how each option implements that feature
5. For option-by-option comparison:
   - Use level 2 headings (##) for each option
   - List the key features and characteristics under each
6. Use tables for side-by-side comparisons when appropriate
7. End with a "Summary" section highlighting key differences
"""

        # Insert comparison instructions before "Your answer should be"
        updated_template = current_template.replace(
            "Your answer should be complete and ready",
            f"{comparison_instructions}\n\nYour answer should be complete and ready"
        )

        vertex_connector.prompt_template = PromptTemplate(
            template=updated_template,
            input_variables=["context", "question"]
        )


def add_troubleshooting_meta_instructions():
    """Add meta-instructions optimized for troubleshooting questions."""
    if vertex_connector:
        current_template = vertex_connector.prompt_template.template

        troubleshooting_instructions = """
TROUBLESHOOTING RESPONSE REQUIREMENTS:
1. Begin with a brief description of the issue being addressed
2. Structure the response in a diagnostic approach:
   - Start with common/simple solutions before complex ones
   - Group related troubleshooting steps under clear headings
3. For each troubleshooting step:
   - Explain what the step addresses and why it might solve the issue
   - Provide specific actions with exact commands or UI paths
   - Include how to verify if the step resolved the issue
4. Add indicators of problem severity or solution complexity when available
5. End with a "Prevention" section with tips to avoid future occurrences
"""

        # Insert troubleshooting instructions before "Your answer should be"
        updated_template = current_template.replace(
            "Your answer should be complete and ready",
            f"{troubleshooting_instructions}\n\nYour answer should be complete and ready"
        )

        vertex_connector.prompt_template = PromptTemplate(
            template=updated_template,
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
