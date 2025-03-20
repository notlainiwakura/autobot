

class LangchainVectorStore:
    """A vector store implementation using Langchain and FAISS"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            embedding_model_name: The name of the sentence transformer model to use
        """
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        self.vector_store = None
        self.documents = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store.
        
        Args:
            documents: A list of dictionaries with 'content' and 'metadata' keys
        """
        # Convert to Langchain Document format
        langchain_docs = []
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            langchain_docs.append(Document(page_content=content, metadata=metadata))
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        split_docs = text_splitter.split_documents(langchain_docs)
        
        # Store the documents
        self.documents = split_docs
        
        # Create the vector store
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: The query string
            k: The number of documents to return
            
        Returns:
            A list of Document objects
        """
        if not self.vector_store:
            return []
        
        return self.vector_store.similarity_search(query, k=k)
    
    def save(self, folder_path: str):
        """Save the vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(folder_path)
    
    def load(self, folder_path: str):
        """Load the vector store from disk"""
        if os.path.exists(folder_path):
            self.vector_store = FAISS.load_local(folder_path, self.embeddings)
            return True
        return False


def create_qa_chain_with_vertex(
    project_id: str,
    documents: List[Dict[str, Any]], 
    embedding_model: str = "all-MiniLM-L6-v2",
    cache_dir: str = "langchain_cache",
    location: str = "us-central1",
    model_name: str = "gemini-1.5-pro"
) -> Tuple:
    """
    Create a QA chain using Langchain and Vertex AI.
    
    Args:
        project_id: Google Cloud project ID
        documents: List of document dictionaries with content and metadata
        embedding_model: The embedding model to use
        cache_dir: Directory to cache the vector store
        location: Google Cloud region
        model_name: Vertex AI model name
        
    Returns:
        A tuple of (vector_store, vertex_connector)
    """
    # Initialize the vector store
    vector_store = LangchainVectorStore(embedding_model_name=embedding_model)
    
    # Try to load from cache first
    if not vector_store.load(cache_dir):
        # If loading fails, add documents and save
        vector_store.add_documents(documents)
        os.makedirs(cache_dir, exist_ok=True)
        vector_store.save(cache_dir)
    
    # Initialize the Vertex AI connector
    vertex_connector = VertexAIConnector(
        project_id=project_id,
        location=location,
        model_name=model_name
    )
    
    return vector_store, vertex_connectorimport os
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_google_vertexai import VertexAI
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Configure logging
logger = logging.getLogger(__name__)

class VertexAIConnector:
    """A connector to Google Vertex AI for enhanced response generation"""
    
    def __init__(self, 
                 project_id: str, 
                 location: str = "us-central1", 
                 model_name: str = "gemini-1.5-pro",
                 temperature: float = 0.1,
                 max_output_tokens: int = 1024):
        """
        Initialize the Vertex AI connector.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            model_name: Vertex AI model name to use
            temperature: Temperature for response generation (0.0-1.0)
            max_output_tokens: Maximum number of tokens to generate
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Initialize the LLM
        self.llm = VertexAI(
            project=project_id,
            location=location,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        
        # Default prompt template for QA
        self.prompt_template = PromptTemplate(
            template="""You are a helpful assistant that answers questions based on Confluence documentation.
            
Context information is below:
--------------------------
{context}
--------------------------

Given the context information and not prior knowledge, answer the following question:
{question}

If the answer is not explicitly contained in the provided context, say "I don't have enough information to answer this question confidently."
Answer in a professional, clear, and concise manner. Format the response with markdown when appropriate.
Include key facts and details from the context, but keep the overall answer concise.
""",
            input_variables=["context", "question"]
        )
    
    def generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a response to a query using the relevant chunks and Vertex AI.
        
        Args:
            query: The user's question
            relevant_chunks: A list of dictionaries with 'chunk' and 'metadata' keys
            
        Returns:
            A string containing the generated answer
        """
        # Format the context by combining chunks
        context = self._format_context(relevant_chunks)
        
        # If we have no context, return a default message
        if not context:
            return "I couldn't find relevant information in the knowledge base to answer your question."
        
        try:
            # Create input for the LLM
            prompt_input = {
                "context": context,
                "question": query
            }
            
            # Generate the response
            response = self.llm.invoke(self.prompt_template.format(**prompt_input))
            
            # Clean up the response if needed
            return response.strip()
            
        except Exception as e:
            # Log the error and return a fallback response
            logger.error(f"Error generating response with Vertex AI: {str(e)}")
            return self._generate_fallback_response(query, relevant_chunks)
    
    def _format_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Format the relevant chunks into a single context string"""
        if not relevant_chunks:
            return ""
        
        # Sort chunks by score (highest first)
        sorted_chunks = sorted(relevant_chunks, key=lambda x: x.get('score', 0), reverse=True)
        
        # Format each chunk with source information
        formatted_chunks = []
        for i, chunk in enumerate(sorted_chunks):
            metadata = chunk.get('metadata', {})
            title = metadata.get('title', 'Unknown document')
            content = chunk.get('chunk', '').strip()
            
            formatted_chunk = f"Document {i+1}: {title}\n{content}\n"
            formatted_chunks.append(formatted_chunk)
        
        return "\n\n".join(formatted_chunks)
    
    def _generate_fallback_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate a fallback response if Vertex AI fails"""
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Use the highest-scoring chunk as the basis for the response
        top_chunk = max(relevant_chunks, key=lambda x: x.get('score', 0))
        content = top_chunk.get('chunk', '')
        
        # Just return a simple response with the content
        return f"Based on the available information:\n\n{content}\n\n(Note: This is a fallback response due to an issue with the AI processing.)"