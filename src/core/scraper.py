"""Scraper module for extracting comments from social media platforms."""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from apify_client import ApifyClient
from src.config.settings import (
    APIFY_TOKEN,
    APIFY_ACTORS,
    DEFAULT_COMMENTS_PER_POST,
    DEFAULT_RESULTS_PER_PAGE,
    RAW_DATA_DIR,
)


class SocialMediaScraper:
    """Scraper for extracting comments from TikTok and Instagram."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the scraper with Apify token.
        
        Args:
            token: Apify API token. If None, uses token from environment.
        """
        self.token = token or APIFY_TOKEN
        if not self.token:
            raise ValueError("Apify token is required. Set it in .env file or pass to constructor.")
        
        self.client = ApifyClient(self.token)
    
    def scrape_tiktok(
        self,
        post_urls: List[str],
        comments_per_post: int = DEFAULT_COMMENTS_PER_POST,
        max_replies: int = 0,
    ) -> List[Dict[str, Any]]:
        """Scrape comments from TikTok posts.
        
        Args:
            post_urls: List of TikTok post URLs
            comments_per_post: Number of comments to extract per post
            max_replies: Maximum number of replies per comment
            
        Returns:
            List of comment data dictionaries
        """
        run_input = {
            "postURLs": post_urls,
            "commentsPerPost": comments_per_post,
            "topLevelCommentsPerPost": comments_per_post,
            "maxRepliesPerComment": max_replies,
            "resultsPerPage": DEFAULT_RESULTS_PER_PAGE,
            "excludePinnedPosts": False,
        }
        
        actor_id = APIFY_ACTORS["tiktok"]
        return self._run_actor(actor_id, run_input)
    
    def scrape_instagram(
        self,
        post_urls: List[str],
        comments_per_post: int = DEFAULT_COMMENTS_PER_POST,
    ) -> List[Dict[str, Any]]:
        """Scrape comments from Instagram posts.
        
        Args:
            post_urls: List of Instagram post URLs
            comments_per_post: Number of comments to extract per post
            
        Returns:
            List of comment data dictionaries
        """
        run_input = {
            "directUrls": post_urls,
            "resultsLimit": comments_per_post,
        }
        
        actor_id = APIFY_ACTORS["instagram"]
        return self._run_actor(actor_id, run_input)
    
    def _run_actor(
        self,
        actor_id: str,
        run_input: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run an Apify actor and return the results.
        
        Args:
            actor_id: The Apify actor ID
            run_input: Input parameters for the actor
            
        Returns:
            List of items from the dataset
        """
        try:
            run = self.client.actor(actor_id).call(run_input=run_input)
            items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
            
            # Save raw data to file
            self._save_raw_data(actor_id, items)
            
            return items
        except Exception as e:
            print(f"Error running actor {actor_id}: {str(e)}")
            return []
    
    def _save_raw_data(
        self,
        actor_id: str,
        data: List[Dict[str, Any]],
    ) -> Optional[Path]:
        """Save raw extraction data to JSON file.
        
        Args:
            actor_id: The Apify actor ID used for extraction
            data: List of extracted items
            
        Returns:
            Path to saved file or None if save failed
        """
        try:
            # Determine platform from actor_id
            platform = "unknown"
            for p, aid in APIFY_ACTORS.items():
                if aid == actor_id:
                    platform = p
                    break
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{platform}_raw_{timestamp}.json"
            filepath = RAW_DATA_DIR / filename
            
            # Save data as JSON
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Raw data saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving raw data: {str(e)}")
            return None
