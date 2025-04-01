import html2text
import logging
from concurrent.futures import ThreadPoolExecutor
from atlassian import Confluence
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfluenceClient:
    """Client for interacting with Confluence API"""

    def __init__(self, url, username, api_token):
        """Initialize the Confluence client"""
        self.url = url
        self.username = username
        self.api_token = api_token

        # Initialize Confluence client
        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token,
            timeout=60  # Increased timeout for larger Confluence instances
        )

        # Initialize HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.body_width = 0  # Don't wrap text at a specific width

    def fetch_content(self, space_key=None, max_pages_per_space=500, max_workers=4):
        """Fetch content from Confluence with proper pagination handling"""
        all_pages = []

        if space_key:
            # If specific spaces are provided, get pages from those spaces
            space_keys = [s.strip() for s in space_key.split(',')] if space_key else []
            for space in space_keys:
                logger.info(f"Fetching pages from space: {space}")
                try:
                    # Initialize variables for pagination
                    start = 0
                    page_limit = 100  # This is the API's internal limit per request
                    all_space_pages = []
                    has_more = True

                    # Keep fetching pages until we get all of them or reach our maximum limit
                    while has_more and start < max_pages_per_space:
                        # Get a batch of pages with pagination parameters
                        logger.info(f"Fetching pages {start} to {start + page_limit} from space {space}")
                        batch = self.confluence.get_all_pages_from_space(
                            space,
                            start=start,
                            limit=page_limit,
                            expand="body.storage,version"  # Add version to get lastUpdated info
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
                all_spaces = self.confluence.get_all_spaces()
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
                        while has_more and start < max_pages_per_space:
                            # Get a batch of pages with pagination parameters
                            logger.info(f"Fetching pages {start} to {start + page_limit} from space {space['key']}")
                            batch = self.confluence.get_all_pages_from_space(
                                space['key'],
                                start=start,
                                limit=page_limit,
                                expand="body.storage,version"  # Add version to get lastUpdated info
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

        # Sort pages by last updated date (newest first)
        try:
            # Extract and parse date from each page
            for page in all_pages:
                try:
                    if 'version' in page and 'when' in page['version']:
                        when_str = page['version']['when']
                        # Parse ISO format date
                        page['_parsed_date'] = datetime.fromisoformat(when_str.replace('Z', '+00:00'))
                    else:
                        # Set a very old date for pages without a date
                        page['_parsed_date'] = datetime.min
                except Exception:
                    # If date parsing fails, use a very old date
                    page['_parsed_date'] = datetime.min

            # Sort pages - newest first
            all_pages.sort(key=lambda x: x.get('_parsed_date', datetime.min), reverse=True)
            logger.info("Sorted pages by last updated date (newest first)")
        except Exception as e:
            logger.error(f"Error sorting pages by date: {str(e)}")

        # Process pages in parallel - already sorted by date now
        documents = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page processing tasks
            future_to_page = {executor.submit(self.process_page, page): page for page in all_pages}

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

    def process_page(self, page):
        """Process a single Confluence page and return a document"""
        try:
            # Get the page ID
            page_id = page['id']

            # Get page content with expanded body and space information
            page_content = self.confluence.get_page_by_id(page_id,
                                                          expand='body.storage,version,history,metadata.labels,space')

            # Extract HTML content
            html_content = page_content['body']['storage']['value']

            # Convert HTML to text
            text_content = self.html_converter.handle(html_content)

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
                    'url': f"{self.url}/wiki/spaces/{space_key}/pages/{page_id}",
                    'space': page_content.get('space', {}).get('name', page.get('space', {}).get('name', 'Unknown')),
                    'space_key': space_key,
                    'labels': labels,
                    'last_updated': updated_date,
                    'last_updater': last_updater
                }
            }

            logger.info(f"Created URL for page {page_id} in space {space_key}: {doc['metadata']['url']}")

            return doc
        except Exception as e:
            logger.error(f"Error processing page {page.get('title', 'Unknown')}: {str(e)}")
            return None