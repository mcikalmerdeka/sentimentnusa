"""Scraper module for extracting comments from social media platforms."""
from typing import List, Dict, Any, Optional
from apify_client import ApifyClient
from src.config.settings import (
    APIFY_TOKEN,
    APIFY_ACTORS,
    DEFAULT_COMMENTS_PER_POST,
    DEFAULT_RESULTS_PER_PAGE,
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
            return items
        except Exception as e:
            print(f"Error running actor {actor_id}: {str(e)}")
            return []
