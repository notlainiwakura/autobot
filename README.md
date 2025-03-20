# AutoBot

A Slack bot that connects to our Confluence workspace and uses advanced retrieval techniques to answer questions about our documentation. The bot runs entirely within our infrastructure, keeping our data secure and private.

## Features

### AI-Enhanced Responses
- **Vertex AI Integration**: Optional integration with Google Vertex AI for more coherent, context-aware responses
- **LangChain Framework**: Uses LangChain for advanced document processing and generation
- **Fallback Mechanisms**: Gracefully degrades to basic extraction if AI services are unavailable

### Core Capabilities
- **Semantic Search**: Uses advanced embedding models to understand the meaning of questions, not just keywords
- **Intelligent Context Awareness**: Organizes related content from multiple documents to provide comprehensive answers
- **Smart Chunking**: Splits documents into meaningful segments while preserving context for better retrieval
- **Token-Based Processing**: Uses tiktoken for accurate token counting, ensuring optimal chunk sizes

### Performance Enhancements
- **Persistent Caching**: Saves processed documents and embeddings to disk for faster startup
- **Automatic Refreshes**: Periodically updates the knowledge base in the background
- **Parallel Processing**: Uses multithreading to speed up document processing
- **Graceful Shutdowns**: Preserves state to recover quickly after restarts

### User Experience Improvements
- **Improved Message Formatting**: Better structured responses with clear source citations
- **Smart Message Splitting**: Handles long responses by splitting on natural sentence boundaries
- **Thread Management**: Prevents overlapping requests that could cause confusion
- **Typing Indicators**: Shows when the bot is processing a question
- **Home Tab**: Provides usage information and status directly in Slack
- **Admin Controls**: Allows workspace admins to refresh the knowledge base

## Setup Instructions

### Requirements
- Python 3.8+
- Slack App with Bot Token and Socket Mode enabled
- Confluence API access

### Environment Variables
Create a `.env` file with the following variables:

```
# Slack Credentials
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token

# Confluence Credentials
CONFLUENCE_URL=https://your-workspace.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACE_KEY=SPACE1,SPACE2  # Optional, comma-separated list of spaces to index

# Bot Configuration
CACHE_DIR=cache  # Directory for caching (will be created if it doesn't exist)
REFRESH_INTERVAL_HOURS=24  # How often to refresh the knowledge base
MAX_WORKERS=4  # Number of worker threads for parallel processing
CHUNK_SIZE=512  # Size of document chunks (in tokens)
CHUNK_OVERLAP=128  # Overlap between chunks (in tokens)
TOP_K_RESULTS=5  # Number of results to return per query
EMBEDDING_MODEL=msmarco-distilbert-base-v4  # Model to use for embeddings

# Optional Vertex AI Integration
USE_VERTEX_AI=false  # Set to true to enable Vertex AI integration
GCP_PROJECT_ID=your-gcp-project-id  # Your Google Cloud project ID
GCP_LOCATION=us-central1  # Google Cloud region
VERTEX_MODEL=gemini-1.5-pro  # Vertex AI model to use
LANGCHAIN_CACHE_DIR=langchain_cache  # Directory for LangChain cache
```

### Installation
1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create your `.env` file with the required credentials
6. Run the bot: `python main.py`

### Slack App Permissions

Your Slack app needs the following scopes:
- `app_mentions:read` - To receive mentions in channels
- `chat:write` - To send messages
- `im:history` - To read direct messages
- `im:read` - To read direct message events
- `im:write` - To send direct messages
- `reactions:write` - (Optional) To show typing indicators via reactions

### Required Packages
```
slack-bolt
atlassian-python-api
html2text
sentence-transformers
scikit-learn
nltk
python-dotenv
numpy
tiktoken
```

### Optional Packages (for Vertex AI integration)
```
google-cloud-aiplatform
langchain
langchain-google-vertexai
faiss-cpu
```

## Usage

### Basic Commands
- **Ask a question**: Mention the bot in a channel or send a direct message with your question
- **Get help**: Send `help` to the bot
- **Refresh the knowledge base**: Send `refresh` to the bot (admin only)

### Tips for Good Questions
- Be specific in your questions
- Include keywords that might appear in the documents
- If you don't get a good answer, try rephrasing your question

## Architecture

The bot follows a Retrieval-Augmented Generation (RAG) approach:

1. **Indexing Phase**:
   - Fetches all pages from Confluence
   - Splits content into semantically meaningful chunks
   - Creates vector embeddings for each chunk
   - Stores chunks, embeddings, and metadata in memory and on disk

2. **Query Phase**:
   - Converts user question to a vector embedding
   - Finds the most similar document chunks
   - Extracts relevant sentences from these chunks
   - Organizes information by source document
   - Formats a coherent response with citations




-----------------
SAMPLE .env FILE
-----------------
# Slack Credentials
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token

# Confluence Credentials
CONFLUENCE_URL=https://your-workspace.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACE_KEY=SPACE1,SPACE2  # Optional, comma-separated list of spaces to index

# Bot Configuration
CACHE_DIR=cache
REFRESH_INTERVAL_HOURS=24
MAX_WORKERS=4
CHUNK_SIZE=512
CHUNK_OVERLAP=128
TOP_K_RESULTS=5
EMBEDDING_MODEL=msmarco-distilbert-base-v4

# Vertex AI Integration (Optional)
USE_VERTEX_AI=true  # Set to false to disable Vertex AI
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us-central1
VERTEX_MODEL=gemini-1.5-pro
LANGCHAIN_CACHE_DIR=langchain_cache
