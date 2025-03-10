# Autobot

Autobot is a Slack application that connects to PLQA workspace, fetches and processes Confluence pages, and allows users to ask questions about the content directly from Slack. It leverages semantic search using sentence embeddings to provide contextually relevant answers.

## Features

- **Confluence Integration:**  
  Retrieves pages from specified Confluence spaces and extracts plain text from HTML content.

- **Document Processing & Chunking:**  
  Splits Confluence content into manageable, overlapping chunks to maintain context and improve search relevance.

- **Semantic Search:**  
  Uses the `all-MiniLM-L6-v2` SentenceTransformer model to generate embeddings and perform fast, efficient semantic similarity searches.

- **Slack Bot Interaction:**  
  Responds to app mentions and interactive messages.
