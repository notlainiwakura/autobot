import os
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfluenceCache:
    """Class to handle caching of Confluence content and embeddings"""
    
    @staticmethod
    def get_cache_path(cache_type, cache_dir="cache"):
        """Get the path to a specific cache file"""
        return os.path.join(cache_dir, f"{cache_type}.pkl")
    
    @staticmethod
    def save_to_cache(data, cache_type, cache_dir="cache"):
        """Save data to cache file"""
        try:
            # Ensure the cache directory exists
            Path(cache_dir).mkdir(exist_ok=True)
            
            with open(ConfluenceCache.get_cache_path(cache_type, cache_dir), 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {cache_type} to cache")
            return True
        except Exception as e:
            logger.error(f"Error saving {cache_type} to cache: {str(e)}")
            return False
    
    @staticmethod
    def load_from_cache(cache_type, cache_dir="cache"):
        """Load data from cache file"""
        try:
            cache_path = ConfluenceCache.get_cache_path(cache_type, cache_dir)
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
    def is_cache_valid(cache_dir="cache", refresh_interval=24):
        """Check if cache exists and is not too old"""
        try:
            metadata_path = ConfluenceCache.get_cache_path("metadata", cache_dir)
            if not os.path.exists(metadata_path):
                return False
                
            # Check when the cache was last modified
            mtime = os.path.getmtime(metadata_path)
            cache_time = datetime.fromtimestamp(mtime)
            now = datetime.now()
            
            # If cache is older than refresh interval, it's invalid
            return (now - cache_time) < timedelta(hours=refresh_interval)
        except Exception as e:
            logger.error(f"Error checking cache validity: {str(e)}")
            return False
