import os
import math
import logging
import numpy as np
import threading
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from nltk.tokenize import sent_tokenize
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

from cache_manager import ConfluenceCache
from utils import count_tokens

# Download NLTK data for tokenization
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Manages the knowledge base of Confluence documents"""
    
    def __init__(self, config, confluence_client):
        """Initialize the knowledge base"""
        self.config = config
        self.confluence_client = confluence_client
        
        # Storage for document chunks and embeddings
        self.document_chunks = []
        self.chunk_embeddings = np.array([])
        self.document_metadata = []
        
        # Thread safety and auto-refresh
        self.knowledge_base_lock = threading.Lock()
        self.last_refresh_time = None
        self.auto_refresh_thread = None
        self.refresh_event = threading.Event()
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize embedding model
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        # Create cache directory if it doesn't exist
        Path(self.config.CACHE_DIR).mkdir(exist_ok=True)
    
    def initialize(self, force=False):
        """Initialize the knowledge base from Confluence or load from cache"""
        with self.knowledge_base_lock:
            # If we have a valid cache and aren't forcing a refresh, use it
            if not force and ConfluenceCache.is_cache_valid(self.config.CACHE_DIR, self.config.REFRESH_INTERVAL):
                logger.info("Loading knowledge base from cache...")
                self.document_chunks = ConfluenceCache.load_from_cache("chunks", self.config.CACHE_DIR)
                self.chunk_embeddings = ConfluenceCache.load_from_cache("embeddings", self.config.CACHE_DIR)
                self.document_metadata = ConfluenceCache.load_from_cache("metadata", self.config.CACHE_DIR)
                
                if (self.document_chunks is not None and
                    isinstance(self.chunk_embeddings, np.ndarray) and len(self.chunk_embeddings) > 0 and
                    self.document_metadata is not None):
                    logger.info(f"Successfully loaded knowledge base from cache: {len(self.document_chunks)} chunks")
                    self.last_refresh_time = datetime.now()
                    
                    # Start the auto-refresh thread if it's not running
                    self._ensure_auto_refresh_thread()
                    return True
                else:
                    logger.warning("Cache seems corrupt or incomplete, rebuilding knowledge base")
                    # Fall through to rebuild
            
            # If cache is invalid or we're forcing a refresh, build from scratch
            logger.info("Building knowledge base from Confluence...")
            try:
                start_time = time.time()
                documents = self.confluence_client.fetch_content(
                    space_key=self.config.SPACE_KEY,
                    max_pages_per_space=self.config.MAX_PAGES_PER_SPACE,
                    max_workers=self.config.MAX_WORKERS
                )
                self._chunk_documents(documents)
                self._create_embeddings()
                self.last_refresh_time = datetime.now()
                
                # Save to cache
                ConfluenceCache.save_to_cache(self.document_chunks, "chunks", self.config.CACHE_DIR)
                ConfluenceCache.save_to_cache(self.chunk_embeddings, "embeddings", self.config.CACHE_DIR)
                ConfluenceCache.save_to_cache(self.document_metadata, "metadata", self.config.CACHE_DIR)
                
                # Start the auto-refresh thread if it's not running
                self._ensure_auto_refresh_thread()
                
                elapsed_time = time.time() - start_time
                logger.info(f"Knowledge base initialized in {elapsed_time:.2f} seconds with {len(self.document_chunks)} chunks")
                return True
            except Exception as e:
                logger.error(f"Error initializing knowledge base: {str(e)}")
                return False
    
    def _chunk_documents(self, documents):
        """Split documents into overlapping chunks using token-based approach with sentence boundaries preserved."""
        self.document_chunks = []
        self.document_metadata = []
        
        for doc in documents:
            # Split content into sentences
            sentences = sent_tokenize(doc['content'])
            
            # Get token counts for each sentence
            sentence_token_counts = [count_tokens(sentence, self.tokenizer) for sentence in sentences]
            
            current_chunk_sentences = []
            current_chunk_token_count = 0
            
            for i, (sentence, token_count) in enumerate(zip(sentences, sentence_token_counts)):
                # If a single sentence exceeds chunk size, we need to split it further
                if token_count > self.config.CHUNK_SIZE:
                    # Just include it as its own chunk to avoid complexity
                    # This is a rare case that would require word-level chunking
                    self.document_chunks.append(sentence)
                    self.document_metadata.append({
                        **doc['metadata'],
                        'chunk_index': len(self.document_chunks) - 1,
                        'is_large_sentence': True
                    })
                    continue
                
                # If adding this sentence would exceed the chunk size and we have content
                if current_chunk_token_count + token_count > self.config.CHUNK_SIZE and current_chunk_sentences:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk_sentences)
                    self.document_chunks.append(chunk_text)
                    self.document_metadata.append({
                        **doc['metadata'],
                        'chunk_index': len(self.document_chunks) - 1
                    })
                    
                    # Calculate overlap using tokens
                    overlap_sentences = []
                    overlap_token_count = 0
                    
                    # Start from the end and work backwards for overlap
                    for s, s_token_count in reversed(list(zip(current_chunk_sentences,
                                                             [count_tokens(s, self.tokenizer) for s in current_chunk_sentences]))):
                        if overlap_token_count + s_token_count <= self.config.CHUNK_OVERLAP:
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
                self.document_chunks.append(chunk_text)
                self.document_metadata.append({
                    **doc['metadata'],
                    'chunk_index': len(self.document_chunks) - 1
                })
        
        logger.info(f"Created {len(self.document_chunks)} chunks from {len(documents)} documents")
    
    def _create_embeddings(self):
        """Create embeddings for all document chunks with batching"""
        logger.info("Creating embeddings for document chunks...")
        
        # Process in batches to avoid memory issues with large knowledge bases
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(self.document_chunks), batch_size):
            batch = self.document_chunks[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(batch, show_progress_bar=(len(batch) > 10))
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i//batch_size}: {str(e)}")
                # Create zero embeddings as placeholders for failed batches
                batch_embeddings = np.zeros((len(batch), self.model.get_sentence_embedding_dimension()))
                all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        if all_embeddings:
            self.chunk_embeddings = np.vstack(all_embeddings)
        else:
            self.chunk_embeddings = np.array([])
        
        logger.info(f"Created {len(self.chunk_embeddings)} embeddings")
    
    def search_documents(self, query, top_k=5, min_score=0.4):
        """Search for document chunks and prioritize documents with multiple relevant chunks."""
        # Ensure we have data to search
        if len(self.document_chunks) == 0 or not isinstance(self.chunk_embeddings, np.ndarray) or len(self.chunk_embeddings) == 0:
            logger.warning("Knowledge base is empty, cannot search")
            return []
            
        # Encode the query
        query_embedding = self.model.encode([query])[0]

        # Calculate similarity scores
        similarity_scores = cosine_similarity([query_embedding], self.chunk_embeddings)[0]

        # Get results above minimum score
        candidate_indices = [idx for idx in range(len(similarity_scores)) if similarity_scores[idx] >= min_score]

        # Group by document ID
        doc_id_to_chunks = {}
        for idx in candidate_indices:
            doc_id = self.document_metadata[idx]['id']
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
                    'chunk': self.document_chunks[idx],
                    'metadata': self.document_metadata[idx],
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
                        'chunk': self.document_chunks[idx],
                        'metadata': self.document_metadata[idx],
                        'score': chunk['score']
                    })

        logger.info(f"Selected {len(results)} chunks from {docs_selected} documents")
        return results
    
    def _ensure_auto_refresh_thread(self):
        """Ensure the auto-refresh thread is running"""
        if self.auto_refresh_thread is None or not self.auto_refresh_thread.is_alive():
            self.refresh_event.clear()
            self.auto_refresh_thread = threading.Thread(target=self._auto_refresh_knowledge_base, daemon=True)
            self.auto_refresh_thread.start()
    
    def _auto_refresh_knowledge_base(self):
        """Background thread to automatically refresh the knowledge base"""
        logger.info("Starting automatic knowledge base refresh thread")
        while not self.refresh_event.is_set():
            # Sleep for a bit, checking periodically if we should exit
            for _ in range(60):  # Check every minute if we should exit
                if self.refresh_event.is_set():
                    break
                time.sleep(60)
            
            # If it's time to refresh (or it's never been refreshed), do it
            with self.knowledge_base_lock:
                if (not self.last_refresh_time or
                    (datetime.now() - self.last_refresh_time > timedelta(hours=self.config.REFRESH_INTERVAL))):
                    logger.info("Auto-refreshing knowledge base...")
                    self.initialize(force=True)
        
        logger.info("Automatic knowledge base refresh thread stopped")
    
    def stop_auto_refresh(self):
        """Stop the auto-refresh thread"""
        self.refresh_event.set()
        if self.auto_refresh_thread and self.auto_refresh_thread.is_alive():
            self.auto_refresh_thread.join(timeout=5)
    
    def save_state(self):
        """Save the current state to disk for recovery"""
        with self.knowledge_base_lock:
            if self.document_chunks and isinstance(self.chunk_embeddings, np.ndarray) and len(self.chunk_embeddings) > 0 and self.document_metadata:
                try:
                    state = {
                        "last_refresh_time": self.last_refresh_time.isoformat() if self.last_refresh_time else None,
                    }
                    with open(os.path.join(self.config.CACHE_DIR, "state.json"), 'w') as f:
                        json.dump(state, f)
                    logger.info("Saved state to disk")
                except Exception as e:
                    logger.error(f"Error saving state: {str(e)}")
    
    def load_state(self):
        """Load saved state from disk"""
        try:
            state_path = os.path.join(self.config.CACHE_DIR, "state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    
                if state.get("last_refresh_time"):
                    self.last_refresh_time = datetime.fromisoformat(state["last_refresh_time"])
                    logger.info(f"Loaded state: last refresh {self.last_refresh_time}")
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
