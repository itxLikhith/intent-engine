"""
Intent Engine - Seed URL Injector Service

Injects discovered seed URLs directly into the Go crawler's Redis queue.
This allows dynamic expansion of the crawler's frontier without modifying the Go code.
"""

import asyncio
import json
import logging
from datetime import datetime

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class SeedURLInjector:
    """
    Injects seed URLs directly into the crawler's Redis queue.

    The Go crawler uses Redis for URL queue management.
    We can add URLs directly to the queue without modifying Go code.
    """

    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.redis_url = redis_url
        self.redis_client = None

    async def connect(self):
        """Connect to Redis."""
        if not self.redis_client:
            self.redis_client = redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
            logger.info(f"Connected to Redis: {self.redis_url}")

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")

    async def inject_seed_urls(self, urls: list[dict], priority: int = 5, max_depth: int = 2) -> int:
        """
        Inject seed URLs into the crawler's Redis queue.

        Args:
            urls: List of URL dictionaries with 'url', 'score', etc.
            priority: Queue priority (higher = more important)
            max_depth: Maximum crawl depth for these URLs

        Returns:
            Number of URLs successfully injected
        """
        if not self.redis_client:
            await self.connect()

        injected = 0
        queue_key = "crawl_queue"

        for url_data in urls:
            try:
                url = url_data.get("url")
                if not url:
                    continue

                # Check if URL already in queue (by checking visited set)
                visited_key = "visited_urls"
                url_hash = self._hash_url(url)

                if await self.redis_client.sismember(visited_key, url_hash):
                    logger.debug(f"URL already visited: {url}")
                    continue

                # Create queue item matching Go crawler's expected format
                queue_item = {
                    "id": f"seed_{self._generate_id()}",
                    "url": url,
                    "priority": priority,
                    "depth": 0,
                    "max_depth": max_depth,
                    "status": "pending",
                    "scheduled_at": datetime.utcnow().isoformat(),
                    "created_at": datetime.utcnow().isoformat(),
                    "source": "seed_discovery",
                    "topic": url_data.get("topic", "general"),
                    "discovery_score": url_data.get("score", 0.5),
                }

                # Add to Redis sorted set (priority queue)
                # Score = priority (higher priority = processed first)
                await self.redis_client.zadd(queue_key, {json.dumps(queue_item): float(priority)})

                injected += 1
                logger.debug(f"Injected seed URL: {url} (priority={priority})")

            except Exception as e:
                logger.error(f"Failed to inject URL {url}: {e}")
                continue

        logger.info(f"Successfully injected {injected}/{len(urls)} seed URLs")
        return injected

    async def inject_raw_urls(self, urls: list[str], priority: int = 5, max_depth: int = 2) -> int:
        """
        Inject raw URL strings into the crawler queue.

        Args:
            urls: List of URL strings
            priority: Queue priority
            max_depth: Maximum crawl depth

        Returns:
            Number of URLs injected
        """
        # Convert to dict format
        url_dicts = [{"url": url, "score": 0.5, "topic": "general"} for url in urls]
        return await self.inject_seed_urls(url_dicts, priority, max_depth)

    def _hash_url(self, url: str) -> str:
        """Generate hash for URL deduplication."""
        import hashlib

        return hashlib.md5(url.encode()).hexdigest()

    def _generate_id(self) -> str:
        """Generate unique ID for queue item."""
        import uuid

        return uuid.uuid4().hex[:16]

    async def get_queue_stats(self) -> dict:
        """Get current queue statistics."""
        if not self.redis_client:
            await self.connect()

        queue_key = "crawl_queue"
        visited_key = "visited_urls"

        queue_size = await self.redis_client.zcard(queue_key)
        visited_count = await self.redis_client.scard(visited_key)

        return {"queue_size": queue_size, "visited_count": visited_count, "timestamp": datetime.utcnow().isoformat()}


async def inject_discovered_urls():
    """
    Main entry point for injecting discovered URLs.

    Discovers URLs using SeedURLDiscovery and injects them into crawler queue.
    """
    from searxng.seed_discovery import DISCOVERY_TOPICS, SeedURLDiscovery

    logger.info("Starting seed URL discovery and injection...")

    # Initialize components
    discovery = SeedURLDiscovery()
    injector = SeedURLInjector()

    await injector.connect()

    # Discover URLs for Go language
    logger.info("Discovering Go language URLs...")
    go_urls = await discovery.discover_multiple_topics(topics=DISCOVERY_TOPICS["go_language"], urls_per_topic=10)

    if go_urls:
        injected = await injector.inject_seed_urls(go_urls, priority=8)
        logger.info(f"Injected {injected} Go language URLs")

    # Discover URLs for general programming
    logger.info("Discovering programming URLs...")
    prog_urls = await discovery.discover_multiple_topics(topics=DISCOVERY_TOPICS["programming"], urls_per_topic=5)

    if prog_urls:
        injected = await injector.inject_seed_urls(prog_urls, priority=6)
        logger.info(f"Injected {injected} programming URLs")

    # Get queue stats
    stats = await injector.get_queue_stats()
    logger.info(f"Queue stats: {stats}")

    await injector.close()

    return {
        "go_urls_discovered": len(go_urls),
        "prog_urls_discovered": len(prog_urls),
        "queue_size": stats["queue_size"],
        "visited_count": stats["visited_count"],
    }


if __name__ == "__main__":
    results = asyncio.run(inject_discovered_urls())
    print(f"Injection results: {results}")
