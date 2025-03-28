import re
import logging
from datetime import datetime
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

def extract_answer(query, relevant_chunks, vertex_connector=None, max_tokens=1200):
    """Extract a coherent answer from relevant chunks, prioritizing single-source information."""
    if not relevant_chunks:
        return "I couldn't find relevant information to answer your question."

    # If Vertex AI integration is enabled and available, use it
    if vertex_connector:
        try:
            logger.info("Using Vertex AI for response generation")
            raw_response = vertex_connector.generate_response(query, relevant_chunks)
            # Clean the response for Slack
            cleaned_response = clean_markdown_for_slack(raw_response)
            return cleaned_response
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
        context_sentences = sent_tokenize(context)
        
        # Extract most relevant sentences
        selected_sentences = []
        current_tokens = 0
        
        # First, include sentences with key information based on keywords
        query_keywords = re.findall(r'\b\w+\b', query.lower())
        query_keywords = [kw for kw in query_keywords if len(kw) > 3]  # Filter short words
        
        for sentence in context_sentences:
            # Check if sentence contains query keywords
            sentence_lower = sentence.lower()
            keyword_matches = sum(1 for kw in query_keywords if kw in sentence_lower)
            
            if keyword_matches > 0:
                sentence_tokens = count_tokens(sentence)
                if current_tokens + sentence_tokens <= max_tokens:
                    selected_sentences.append(sentence)
                    current_tokens += sentence_tokens
        
        # If we still have token budget, add more sentences to maintain context
        remaining_sentences = [s for s in context_sentences if s not in selected_sentences]
        for sentence in remaining_sentences:
            sentence_tokens = count_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                selected_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        # Sort sentences back to original order
        sentence_positions = {s: i for i, s in enumerate(context_sentences)}
        selected_sentences.sort(key=lambda s: sentence_positions.get(s, 0))
        
        context = " ".join(selected_sentences)

    # Clean up by removing section headings and formatting
    cleaned_context = re.sub(r'#{1,6}\s+.*?\n', '', context)  # Remove markdown headings
    cleaned_context = re.sub(r'\*\*.*?\*\*', '', cleaned_context)  # Remove bold text markers

    # Clean for Slack formatting
    return clean_markdown_for_slack(cleaned_context)

def format_response(query, answer, relevant_chunks):
    """Format the response with only the most relevant source."""
    # Format answer text, preserving paragraph breaks
    formatted_answer = answer.strip()

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

def clean_markdown_for_slack(text):
    """
    Clean markdown syntax in text to make it display correctly in Slack.
    Transforms markdown headers and formatting to Slack-friendly format.
    """
    if not text:
        return text
        
    lines = text.split('\n')
    result_lines = []
    
    # Process line by line for better control
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Handle headers (## Heading) - convert to bold with spacing
        header_match = re.match(r'^\s*(#{2,3})\s+(.+?)$', line)
        if header_match:
            # Add empty line before header if not at the beginning and previous line isn't empty
            if i > 0 and lines[i-1].strip():
                result_lines.append('')
                
            # Add the header as bold text
            result_lines.append(f"*{header_match.group(2)}*")
            
            # Add empty line after header if next line isn't empty
            if i < len(lines) - 1 and lines[i+1].strip():
                result_lines.append('')
        
        # Handle bold text (**text**) - ensure proper bold formatting for Slack
        elif '**' in line:
            # Replace **text** with *text* (Slack bold)
            processed_line = re.sub(r'\*\*([^*]+?)\*\*', r'*\1*', line)
            
            # If the entire line is bold and looks like a subheading, add spacing
            if re.match(r'^\s*\*\*[^*]+?\*\*\s*$', line):
                # Add empty line before if not at the beginning and previous line isn't empty
                if i > 0 and lines[i-1].strip():
                    result_lines.append('')
                
                # Add the bold line (converted to Slack format)
                result_lines.append(processed_line)
                
                # Add empty line after if next line isn't empty
                if i < len(lines) - 1 and lines[i+1].strip():
                    result_lines.append('')
            else:
                # Regular bold text within a line
                result_lines.append(processed_line)
        else:
            # Keep other lines as is
            result_lines.append(line)
        
        i += 1
    
    # Join lines back together
    result = '\n'.join(result_lines)
    
    # Fix markdown links to display properly
    # [text](url) -> text (url)
    result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', result)
    
    # Fix code blocks for better Slack display
    # ```code``` -> `code`
    result = re.sub(r'```(?:\w+)?\n(.+?)\n```', r'`\1`', result, flags=re.DOTALL)
    
    return result

def count_tokens(text):
    """Rough estimate of token count"""
    # Simple approximation: ~4 chars per token
    return len(text) // 4
