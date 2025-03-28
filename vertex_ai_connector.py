import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if Vertex AI libraries are available
vertex_ai_available = False
try:
    from langchain_google_vertexai import ChatVertexAI
    from langchain.schema import Document
    from langchain.vectorstores import FAISS
    from langchain.embeddings import SentenceTransformerEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    
    vertex_ai_available = True
    logger.info("Successfully imported Vertex AI libraries")
except ImportError as e:
    logger.warning(f"Vertex AI integration not available: {str(e)}. Install required packages.")
    vertex_ai_available = False
except Exception as e:
    logger.error(f"Unexpected error setting up Vertex AI: {str(e)}")
    vertex_ai_available = False

class VertexAIConnector:
    """Connector for Google Vertex AI integration"""
    
    def __init__(self, project_id, location="us-central1", model_name="gemini-2.5-pro",
                temperature=0.1, max_output_tokens=8000):
        """Initialize the Vertex AI connector"""
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
            template="""You are a precise information assistant that delivers well-formatted, structured answers based on Confluence documentation.

Context information:
--------------------------
{context}
--------------------------

Question: {question}

Instructions:
1. Answer the specific question asked using ONLY information from the context
2. Present a SINGLE, coherent response that flows naturally
3. Maintain proper formatting for all lists, steps, and instructions:
   - Ensure numbered lists have proper spacing and indentation
   - Preserve paragraph breaks and section headings
   - Format code snippets, commands, and technical details properly
4. If a procedure has steps, ensure they are clearly numbered and complete
5. If multiple procedures exist (e.g., for different platforms), separate them with clear headings
6. Use consistent terminology throughout your response
7. Prefer information from a single document when possible for coherence
8. If the answer isn't in the context, say "I don't have specific information on this topic."
9. Never reference document names or sources within your answer text

Your response should be a polished, properly formatted answer that could appear in an official guide.
""",
            input_variables=["context", "question"]
        )
    
    def generate_response(self, query, relevant_chunks):
        """Generate a response using Vertex AI"""
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
        """Format chunks into a context string for the prompt"""
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
        """Generate a fallback response when the API call fails"""
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Use highest scoring chunk
        top_chunk = max(relevant_chunks, key=lambda x: x.get('score', 0))
        content = top_chunk.get('chunk', '')
        
        return f"Based on the available information:\n\n{content}\n\n(Note: This is a fallback response due to an issue with the AI processing.)"
    
    def update_prompt_template(self, new_template):
        """Update the prompt template"""
        if not new_template:
            return
            
        self.prompt_template = PromptTemplate(
            template=new_template,
            input_variables=["context", "question"]
        )
        logger.info("Updated Vertex AI prompt template")

def setup_vertex_ai(config):
    """Set up Vertex AI integration if enabled and available"""
    if not config.USE_VERTEX_AI or not vertex_ai_available:
        return None
        
    if not config.GCP_PROJECT_ID:
        logger.error("GCP_PROJECT_ID is required for Vertex AI integration")
        return None
        
    try:
        logger.info(f"Initializing Vertex AI with model: {config.VERTEX_MODEL}")
        connector = VertexAIConnector(
            project_id=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION,
            model_name=config.VERTEX_MODEL
        )
        return connector
    except Exception as e:
        logger.error(f"Error initializing Vertex AI: {str(e)}")
        return None
