"""Scraper module for extracting comments from social media platforms."""
from typing import List, Dict, Any, Optional
from apify_client import ApifyClient
from src.config.settings import (
    APIFY_TOKEN,
    APIFY_ACTORS,
    DEFAULT_COMMENTS_PER_POST,
    DEFAULT_MAX_REPLIES,
    DEFAULT_RESULTS_PER_PAGE,
)


class SocialMediaScraper:
    """Scraper for extracting comments from TikTok, Instagram, and X (Twitter)."""
    
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
        max_replies: int = DEFAULT_MAX_REPLIES,
        exclude_pinned: bool = False,
    ) -> List[Dict[str, Any]]:
        """Scrape comments from TikTok posts.
        
        Args:
            post_urls: List of TikTok post URLs
            comments_per_post: Number of comments to extract per post
            max_replies: Maximum number of replies per comment
            exclude_pinned: Whether to exclude pinned posts
            
        Returns:
            List of comment data dictionaries
        """
        run_input = {
            "commentsPerPost": comments_per_post,
            "excludePinnedPosts": exclude_pinned,
            "maxRepliesPerComment": max_replies,
            "postURLs": post_urls,
            "resultsPerPage": DEFAULT_RESULTS_PER_PAGE,
        }
        
        actor_id = APIFY_ACTORS["tiktok"]
        return self._run_actor(actor_id, run_input)
    
    def scrape_instagram(
        self,
        post_urls: List[str],
        comments_per_post: int = DEFAULT_COMMENTS_PER_POST,
        max_replies: int = DEFAULT_MAX_REPLIES,
    ) -> List[Dict[str, Any]]:
        """Scrape comments from Instagram posts.
        
        Args:
            post_urls: List of Instagram post URLs
            comments_per_post: Number of comments to extract per post
            max_replies: Maximum number of replies per comment
            
        Returns:
            List of comment data dictionaries
        """
        run_input = {
            "posts": post_urls,
            "commentsPerPost": comments_per_post,
            "maxReplies": max_replies,
        }
        
        actor_id = APIFY_ACTORS["instagram"]
        return self._run_actor(actor_id, run_input)
    
    def scrape_x(
        self,
        search_queries: List[str],
        max_tweets: int = DEFAULT_COMMENTS_PER_POST,
        include_replies: bool = True,
    ) -> List[Dict[str, Any]]:
        """Scrape tweets from X (Twitter) based on search queries.
        
        Args:
            search_queries: List of search queries or tweet URLs
            max_tweets: Maximum number of tweets to extract
            include_replies: Whether to include replies
            
        Returns:
            List of tweet data dictionaries
        """
        run_input = {
            "searchTerms": search_queries,
            "maxTweets": max_tweets,
            "includeReplies": include_replies,
        }
        
        actor_id = APIFY_ACTORS["x"]
        return self._run_actor(actor_id, run_input)
    
    def scrape_by_topic(
        self,
        topic: str,
        platforms: List[str] = None,
        limit_per_platform: int = DEFAULT_COMMENTS_PER_POST,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Scrape content from multiple platforms by topic.
        
        Args:
            topic: Search topic or keyword
            platforms: List of platforms to scrape (tiktok, instagram, x)
            limit_per_platform: Number of items to fetch per platform
            
        Returns:
            Dictionary with platform names as keys and lists of items as values
        """
        if platforms is None:
            platforms = ["tiktok", "instagram", "x"]
        
        results = {}
        
        for platform in platforms:
            try:
                if platform == "tiktok":
                    # For TikTok, search by keyword first to get post URLs
                    # This is a simplified approach - in production, you might want
                    # to use a separate actor to search for posts first
                    results[platform] = []
                    
                elif platform == "instagram":
                    # Similar approach for Instagram
                    results[platform] = []
                    
                elif platform == "x":
                    # X (Twitter) supports direct search
                    items = self.scrape_x(
                        search_queries=[topic],
                        max_tweets=limit_per_platform,
                    )
                    results[platform] = items
                    
            except Exception as e:
                print(f"Error scraping {platform}: {str(e)}")
                results[platform] = []
        
        return results
    
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
