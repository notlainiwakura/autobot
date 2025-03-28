import os
import re
import logging
import signal
import sys
import json
import hashlib
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

# Import components
from config import load_config, Config
from confluence_client import ConfluenceClient
from knowledge_base import KnowledgeBase
from response_generator import extract_answer, format_response
from cache_manager import ConfluenceCache
from vertex_ai_connector import setup_vertex_ai
from utils import should_process_message, count_tokens, send_long_message, manage_active_thread

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

# Load configuration
config = load_config()

# Initialize the Slack app
app = App(token=config.SLACK_BOT_TOKEN)

# Initialize Confluence client
confluence_client = ConfluenceClient(
    config.CONFLUENCE_URL,
    config.CONFLUENCE_USERNAME,
    config.CONFLUENCE_API_TOKEN
)

# Initialize Knowledge Base
knowledge_base = KnowledgeBase(config, confluence_client)

# Message deduplication cache
processed_messages = {}

# Flag to track if we've shown the initialization message
kb_init_message_shown = False

# Mapping of Slack channel IDs to in-progress threads
active_threads = {}  # Format: {"channel_id:ts": (start_time, user_id)}
active_threads_lock = threading.Lock()

# Vertex AI integration
vertex_connector = None
if config.USE_VERTEX_AI:
    vertex_connector = setup_vertex_ai(config)
    if vertex_connector:
        logger.info("Vertex AI integration successfully initialized")
    else:
        logger.info("Vertex AI integration failed to initialize")

def ensure_knowledge_base(say=None):
    """Ensure the knowledge base is initialized"""
    global kb_init_message_shown
    
    # Get the current state of the knowledge base
    kb_is_initialized = len(knowledge_base.document_chunks) > 0
    
    # Show initialization message only if not initialized and message not shown before
    if not kb_is_initialized and not kb_init_message_shown and say:
        say(text="Initializing knowledge base. This may take a few minutes...")
        kb_init_message_shown = True
    
    # Initialize if needed
    success = knowledge_base.initialize()
    
    # Show success message only if we showed the initialization message
    if success and not kb_is_initialized and say:
        say(text="Knowledge base initialized! Now processing your question...")
    
    return success

@app.event("app_mention")
def handle_app_mentions(body, say, client):
    """Process messages where the bot is mentioned in channels."""
    event = body["event"]
    
    # Skip if we've processed this message recently
    if not should_process_message(event, processed_messages):
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
            success = knowledge_base.initialize(force=True)
            if success:
                say(text="Knowledge base refreshed successfully! 🎉")
            else:
                say(text="There was an error refreshing the knowledge base. Please check the logs.")
        else:
            say(text="Sorry, only workspace admins can refresh the knowledge base.")
        return
    
    # Check if this thread is already being processed
    if not manage_active_thread(channel_id, thread_ts, user_id, "add", active_threads, active_threads_lock, config.THREAD_TIMEOUT):
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
        relevant_chunks = knowledge_base.search_documents(question, top_k=config.TOP_K_RESULTS)
        
        if not relevant_chunks:
            say(text="I couldn't find any relevant information in the Confluence documents. Please try rephrasing your question or check that the documents you're looking for are in the indexed spaces.")
            return
        
        # Extract answer
        answer = extract_answer(question, relevant_chunks, vertex_connector)
        
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
            logger.warning(f"Could not remove reaction: {str(e)}. This is non-critical.")
        
        # Send the response
        send_long_message(say, formatted_response, thread_ts=thread_ts)
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        say(text="Sorry, I encountered an error while processing your question. Please try again later.")
    
    finally:
        # Remove the thread from active tracking
        manage_active_thread(channel_id, thread_ts, user_id, "remove", active_threads, active_threads_lock, config.THREAD_TIMEOUT)

@app.message("help")
def help_message(message, say):
    """Provide help information in any channel or DM."""
    help_text = """
*Confluence Knowledge Bot Help*

I'm your secure, local Confluence knowledge assistant. I can help you find information from your Confluence workspace without sending your data to external services!

*Commands:*
- Send me a DM with your question about Confluence documents.
- In channels, mention me with your question (e.g., `@ConfluenceBot What is our return policy?`).
- Admins can send `refresh` to update my knowledge base.
- `help` - Show this help message.

*Tips for Good Questions:*
- Be specific in your questions
- Include keywords that might appear in the documents
- If you don't get a good answer, try rephrasing your question

*My Knowledge:*
- I only know what's in your Confluence workspace
- My knowledge updates automatically every 24 hours
- I can search across all spaces or specific ones (configured by your admin)
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
    if not should_process_message(event, processed_messages):
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
            success = knowledge_base.initialize(force=True)
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
    if not manage_active_thread(channel_id, thread_ts, user_id, "add", active_threads, active_threads_lock, config.THREAD_TIMEOUT):
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
            logger.warning(f"Could not add reaction: {str(e)}. This is non-critical.")
        
        # Ensure knowledge base is initialized
        if not ensure_knowledge_base(say):
            say(text="I'm having trouble accessing the knowledge base. Please try again later.")
            return
        
        # Log the query
        logger.info(f"Processing DM question from <@{user_id}>: {question}")
        
        # Search for relevant documents
        relevant_chunks = knowledge_base.search_documents(question, top_k=config.TOP_K_RESULTS)
        
        if not relevant_chunks:
            say(text="I couldn't find any relevant information in the Confluence documents. Please try rephrasing your question or check that the documents you're looking for are in the indexed spaces.")
            return
        
        # Extract answer
        answer = extract_answer(question, relevant_chunks, vertex_connector)
        
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
            logger.warning(f"Could not remove reaction: {str(e)}. This is non-critical.")
        
        # Send the response
        send_long_message(say, formatted_response, thread_ts=thread_ts)
    
    except Exception as e:
        logger.error(f"Error processing DM question: {str(e)}")
        say(text="Sorry, I encountered an error while processing your question. Please try again later.")
    
    finally:
        # Remove the thread from active tracking
        manage_active_thread(channel_id, thread_ts, user_id, "remove", active_threads, active_threads_lock, config.THREAD_TIMEOUT)

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
                                "text": f"Last knowledge base refresh: {knowledge_base.last_refresh_time.strftime('%Y-%m-%d %H:%M:%S') if knowledge_base.last_refresh_time else 'Never'}"
                            }
                        ]
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {str(e)}")

# Function to save state (for graceful shutdowns)
def save_state():
    """Save the current state to disk for recovery"""
    knowledge_base.save_state()

# Add a signal handler for graceful shutdown
def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info("Received shutdown signal, cleaning up...")
    knowledge_base.stop_auto_refresh()
    save_state()
    logger.info("Cleanup complete, exiting")
    sys.exit(0)

# Initialize the app and start it using Socket Mode
if __name__ == "__main__":
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
    
    print(f"Vertex AI integration enabled: {config.USE_VERTEX_AI}")
    if config.USE_VERTEX_AI:
        print(f"GCP Project ID: {config.GCP_PROJECT_ID or 'Not set'}")
        print(f"Vertex AI model: {config.VERTEX_MODEL}")
        
        if not config.GCP_PROJECT_ID:
            print("WARNING: GCP_PROJECT_ID is not set but USE_VERTEX_AI is enabled")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Initializing knowledge base...")
    # Ensure the knowledge base is initialized
    knowledge_base.initialize()
    
    print("Starting Slack bot...")
    logger.info("Starting Confluence Knowledge Bot...")
    SocketModeHandler(app, config.SLACK_APP_TOKEN).start()
