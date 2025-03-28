import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    # Slack configuration
    SLACK_BOT_TOKEN: str 
    SLACK_APP_TOKEN: str
    
    # Confluence configuration
    CONFLUENCE_URL: str
    CONFLUENCE_USERNAME: str 
    CONFLUENCE_API_TOKEN: str
    SPACE_KEY: str = None  # Optional, can specify multiple spaces
    
    # Cache and refresh settings
    CACHE_DIR: str = "cache"
    REFRESH_INTERVAL: int = 24  # Default refresh every 24 hours
    MAX_WORKERS: int = 4  # Thread pool size for parallel processing
    
    # Chunking settings
    CHUNK_SIZE: int = 512  # Default chunk size
    CHUNK_OVERLAP: int = 128  # Default chunk overlap
    TOP_K_RESULTS: int = 5  # Default number of results to return
    MAX_PAGES_PER_SPACE: int = 500  # Maximum pages to fetch per space
    
    # Google Cloud & Vertex AI settings
    USE_VERTEX_AI: bool = False
    GCP_PROJECT_ID: str = ""
    GCP_LOCATION: str = "us-central1"
    VERTEX_MODEL: str = "gemini-2.5-pro"
    LANGCHAIN_CACHE_DIR: str = "langchain_cache"
    
    # Message deduplication settings
    MESSAGE_CACHE_TTL: int = 60  # Time in seconds to keep messages in deduplication cache
    THREAD_TIMEOUT: int = 300  # 5 minutes in seconds
    
    # Embedding model
    EMBEDDING_MODEL: str = "msmarco-distilbert-base-v4"

def load_config():
    """Load configuration from environment variables"""
    
    # Parse boolean values
    def parse_bool(value, default=False):
        if value is None:
            return default
        return value.lower() == "true"
    
    config = Config(
        # Required settings with no defaults
        SLACK_BOT_TOKEN=os.environ["SLACK_BOT_TOKEN"],
        SLACK_APP_TOKEN=os.environ["SLACK_APP_TOKEN"],
        CONFLUENCE_URL=os.environ["CONFLUENCE_URL"],
        CONFLUENCE_USERNAME=os.environ["CONFLUENCE_USERNAME"],
        CONFLUENCE_API_TOKEN=os.environ["CONFLUENCE_API_TOKEN"],
        
        # Optional settings with defaults
        SPACE_KEY=os.environ.get("CONFLUENCE_SPACE_KEY"),
        CACHE_DIR=os.environ.get("CACHE_DIR", "cache"),
        REFRESH_INTERVAL=int(os.environ.get("REFRESH_INTERVAL_HOURS", 24)),
        MAX_WORKERS=int(os.environ.get("MAX_WORKERS", 4)),
        CHUNK_SIZE=int(os.environ.get("CHUNK_SIZE", 512)),
        CHUNK_OVERLAP=int(os.environ.get("CHUNK_OVERLAP", 128)),
        TOP_K_RESULTS=int(os.environ.get("TOP_K_RESULTS", 5)),
        MAX_PAGES_PER_SPACE=int(os.environ.get("MAX_PAGES_PER_SPACE", 500)),
        
        # Vertex AI settings
        USE_VERTEX_AI=parse_bool(os.environ.get("USE_VERTEX_AI"), False),
        GCP_PROJECT_ID=os.environ.get("GCP_PROJECT_ID", ""),
        GCP_LOCATION=os.environ.get("GCP_LOCATION", "us-central1"),
        VERTEX_MODEL=os.environ.get("VERTEX_MODEL", "gemini-2.5-pro"),
        LANGCHAIN_CACHE_DIR=os.environ.get("LANGCHAIN_CACHE_DIR", "langchain_cache"),
        
        # Deduplication settings
        MESSAGE_CACHE_TTL=int(os.environ.get("MESSAGE_CACHE_TTL", 60)),
        THREAD_TIMEOUT=int(os.environ.get("THREAD_TIMEOUT", 300)),
        
        # Embedding model
        EMBEDDING_MODEL=os.environ.get("EMBEDDING_MODEL", "msmarco-distilbert-base-v4")
    )
    
    # Create cache directory if it doesn't exist
    Path(config.CACHE_DIR).mkdir(exist_ok=True)
    
    return config
