"""
Seed URL Manager - Automatically adds search result URLs to crawl queue

This module integrates SearXNG search results with the Go crawler,
creating a self-improving search loop where search results become
new crawl targets.
"""

import json
import logging
from typing import List, Dict, Any

import redis

logger = logging.getLogger(__name__)


class SeedURLManager:
    """Manages automatic seeding of URLs from search results to crawler"""
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.crawl_queue = "crawl_queue"
        self.visited_key_prefix = "visited_urls:"
        self.search_results_key = "search_results_urls"
    
    def add_urls_to_crawl_queue(self, urls: List[str], priority: int = 5, depth: int = 1) -> int:
        """
        Add URLs to the Go crawler queue
        
        Args:
            urls: List of URLs to add
            priority: Crawl priority (1-10, higher = more priority)
            depth: Crawl depth level
            
        Returns:
            Number of URLs added (excludes already visited)
        """
        added_count = 0
        
        for url in urls:
            # Check if already visited
            visited_key = f"{self.visited_key_prefix}{self._hash_url(url)}"
            if self.redis_client.exists(visited_key):
                continue
            
            # Create crawl queue item
            import time
            from datetime import datetime
            
            item = {
                "id": f"crawl_{int(time.time() * 1000)}",
                "url": url,
                "priority": priority,
                "depth": depth,
                "status": "pending",
                "scheduledAt": datetime.utcnow().isoformat(),
                "createdAt": datetime.utcnow().isoformat(),
                "updatedAt": datetime.utcnow().isoformat(),
                "source": "search_results"  # Track that this came from search
            }
            
            # Add to crawl queue (sorted set with priority as score)
            self.redis_client.zadd(
                self.crawl_queue,
                {json.dumps(item): float(priority)}
            )
            
            # Mark as pending visit (24 hour TTL)
            self.redis_client.setex(visited_key, 86400, "1")
            
            # Track for analytics
            self.redis_client.hincrby(self.search_results_key, "total_added", 1)
            
            added_count += 1
            logger.info(f"Added URL to crawl queue: {url} (priority={priority})")
        
        if added_count > 0:
            logger.info(f"Added {added_count} new URLs to crawl queue from search results")
        
        return added_count
    
    def extract_urls_from_searxng_results(self, searxng_results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract unique URLs from SearXNG search results
        
        Args:
            searxng_results: List of result dicts from SearXNG
            
        Returns:
            List of unique URLs
        """
        urls = []
        seen = set()
        
        for result in searxng_results:
            url = result.get("url", "")
            if url and url not in seen:
                # Filter out unwanted domains
                if self._should_crawl_url(url):
                    urls.append(url)
                    seen.add(url)
        
        return urls
    
    def _should_crawl_url(self, url: str) -> bool:
        """Check if URL should be crawled"""
        # Skip known non-crawlable domains
        skip_domains = [
            "facebook.com",
            "twitter.com",
            "instagram.com",
            "linkedin.com",
            "youtube.com",
            "netflix.com",
            "amazon.com",
            "login",
            "signup",
            "register",
        ]
        
        url_lower = url.lower()
        for domain in skip_domains:
            if domain in url_lower:
                return False
        
        # Skip non-HTML content
        skip_extensions = [
            ".pdf", ".doc", ".docx", ".xls", ".xlsx",
            ".zip", ".rar", ".tar", ".gz",
            ".mp3", ".mp4", ".avi", ".mov",
            ".css", ".js", ".woff", ".woff2",
        ]
        
        for ext in skip_extensions:
            if url_lower.endswith(ext):
                return False
        
        return True
    
    def _hash_url(self, url: str) -> str:
        """Create a hash of URL for Redis key"""
        h = 0
        for char in url:
            h = 31 * h + ord(char)
        return str(abs(h))
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get crawl queue statistics"""
        try:
            queue_size = self.redis_client.zcard(self.crawl_queue)
            total_added = self.redis_client.hget(self.search_results_key, "total_added") or 0
            
            return {
                "queue_size": queue_size,
                "total_added_from_search": int(total_added),
                "status": "active"
            }
        except Exception as e:
            return {
                "queue_size": 0,
                "total_added_from_search": 0,
                "status": "error",
                "error": str(e)
            }
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()


# Global instance
_seed_url_manager = None


def get_seed_url_manager() -> SeedURLManager:
    """Get or create seed URL manager instance"""
    global _seed_url_manager
    if _seed_url_manager is None:
        _seed_url_manager = SeedURLManager()
    return _seed_url_manager
