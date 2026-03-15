"""
Intent Engine - Seed URL Discovery Service

Dynamically discovers new seed URLs using SearXNG search results
to expand the Go crawler's coverage automatically.
"""

import asyncio
import logging
from urllib.parse import urlparse

from searxng.client import get_searxng_client

logger = logging.getLogger(__name__)


class SeedURLDiscovery:
    """
    Discovers new seed URLs by searching SearXNG for relevant topics.

    Features:
    - Topic-based URL discovery
    - Domain deduplication
    - Quality scoring based on search ranking
    - Automatic integration with Go crawler
    """

    def __init__(self, searxng_url: str = "http://searxng:8080"):
        self.searxng_url = searxng_url
        self._discovered_domains: set[str] = set()

    async def discover_urls_for_topic(self, topic: str, max_urls: int = 20, min_score: float = 0.5) -> list[dict]:
        """
        Discover seed URLs for a specific topic.

        Args:
            topic: Search topic (e.g., "Go programming language documentation")
            max_urls: Maximum number of URLs to return
            min_score: Minimum quality score (0-1)

        Returns:
            List of discovered URLs with metadata
        """
        logger.info(f"Discovering URLs for topic: {topic}")

        try:
            client = get_searxng_client(self.searxng_url)

            # Search SearXNG for the topic (simplified to avoid category issues)
            response = await client.search(
                query=topic,
                categories=["general"],
                pageno=1,
            )

            if not response or not response.results:
                logger.warning(f"No results found for topic: {topic}")
                return []

            # Extract and score URLs
            discovered = []
            for i, result in enumerate(response.results[: max_urls * 2]):
                url = result.url
                score = self._calculate_url_score(result, i)

                if score < min_score:
                    continue

                # Parse domain for deduplication
                domain = urlparse(url).netloc

                # Skip already discovered domains
                if domain in self._discovered_domains:
                    logger.debug(f"Skipping already discovered domain: {domain}")
                    continue

                discovered.append(
                    {
                        "url": url,
                        "domain": domain,
                        "title": result.title,
                        "score": score,
                        "topic": topic,
                        "engine": result.engine,
                        "position": i + 1,
                    }
                )

                self._discovered_domains.add(domain)

                if len(discovered) >= max_urls:
                    break

            logger.info(f"Discovered {len(discovered)} new URLs for topic: {topic}")
            return discovered

        except Exception as e:
            logger.error(f"Error discovering URLs for topic '{topic}': {e}")
            return []

    def _calculate_url_score(self, result, position: int) -> float:
        """
        Calculate quality score for a discovered URL.

        Scoring factors:
        - Search engine position (higher = better)
        - Number of engines returning this result
        - Result has content/thumbnail
        - Domain authority (based on known good domains)
        """
        score = 0.0

        # Position score (inverse of rank)
        position_score = max(0, 1.0 - (position / 20))
        score += position_score * 0.4

        # Multi-engine score (appearing in multiple engines = more relevant)
        engines = getattr(result, "engines", [result.engine])
        engine_score = min(1.0, len(engines) / 5)
        score += engine_score * 0.3

        # Content quality indicators
        if hasattr(result, "content") and result.content:
            score += 0.15
        if hasattr(result, "thumbnail") and result.thumbnail:
            score += 0.1

        # Domain authority bonus
        domain = urlparse(result.url).netloc
        if self._is_authoritative_domain(domain):
            score += 0.15

        return score

    def _is_authoritative_domain(self, domain: str) -> bool:
        """Check if domain is from an authoritative source."""
        authoritative_patterns = [
            ".org",
            ".edu",
            ".gov",
            "github.com",
            "stackoverflow.com",
            "medium.com",
            "dev.to",
            "freecodecamp.org",
            "tutorialspoint.com",
            "w3schools.com",
            "geeksforgeeks.org",
        ]
        return any(pattern in domain for pattern in authoritative_patterns)

    async def discover_multiple_topics(self, topics: list[str], urls_per_topic: int = 10) -> list[dict]:
        """
        Discover URLs for multiple topics in parallel.

        Args:
            topics: List of search topics
            urls_per_topic: URLs to discover per topic

        Returns:
            Combined list of discovered URLs
        """
        logger.info(f"Discovering URLs for {len(topics)} topics")

        # Run discovery for all topics in parallel
        tasks = [self.discover_urls_for_topic(topic, max_urls=urls_per_topic) for topic in topics]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_urls = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Topic '{topics[i]}' discovery failed: {result}")
                continue
            all_urls.extend(result)

        # Sort by score and deduplicate
        seen_urls = set()
        unique_urls = []
        for url_data in sorted(all_urls, key=lambda x: x["score"], reverse=True):
            if url_data["url"] not in seen_urls:
                seen_urls.add(url_data["url"])
                unique_urls.append(url_data)

        logger.info(f"Total discovered URLs: {len(unique_urls)}")
        return unique_urls

    async def add_to_crawler(self, urls: list[dict], crawler_api_url: str = "http://go-search-api:8080") -> bool:
        """
        Add discovered URLs to the Go crawler queue.

        Args:
            urls: List of discovered URL dictionaries
            crawler_api_url: Go crawler API URL

        Returns:
            True if successfully added
        """
        import aiohttp

        if not urls:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                # Extract just the URLs
                url_list = [u["url"] for u in urls]

                # POST to crawler's seed endpoint
                payload = {
                    "seed_urls": url_list,
                    "priority": 5,  # Medium priority for discovered URLs
                    "depth": 2,
                }

                async with session.post(
                    f"{crawler_api_url}/api/v1/crawl/seed", json=payload, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Added {len(urls)} URLs to crawler queue")
                        return True
                    else:
                        logger.warning(f"Crawler seed endpoint returned {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to add URLs to crawler: {e}")
            return False

    def get_discovered_domains(self) -> set[str]:
        """Return set of all discovered domains."""
        return self._discovered_domains.copy()

    def reset(self):
        """Reset discovered domains set."""
        self._discovered_domains.clear()
        logger.info("Seed URL discovery reset")


# Predefined topics for different content categories
DISCOVERY_TOPICS = {
    "programming": [
        "programming language tutorials",
        "software development best practices",
        "coding bootcamp resources",
        "computer science fundamentals",
        "API documentation examples",
    ],
    "go_language": [
        "Go programming language tutorial",
        "Golang best practices",
        "Go web development",
        "Go microservices examples",
        "Go concurrency patterns",
    ],
    "python": [
        "Python programming tutorial",
        "Python web development Django Flask",
        "Python data science tutorial",
        "Python machine learning examples",
        "Python automation scripts",
    ],
    "web_dev": [
        "web development tutorial",
        "JavaScript frameworks React Vue",
        "CSS responsive design",
        "backend development Node.js",
        "full stack development guide",
    ],
    "devops": [
        "DevOps tutorial for beginners",
        "Docker Kubernetes guide",
        "CI/CD pipeline setup",
        "cloud infrastructure AWS Azure",
        "monitoring and logging best practices",
    ],
}


async def run_seed_discovery():
    """
    Main entry point for seed URL discovery.

    Discovers URLs for predefined topics and adds them to crawler.
    """
    logger.info("Starting seed URL discovery...")

    discovery = SeedURLDiscovery()

    # Discover URLs for Go language topics
    go_urls = await discovery.discover_multiple_topics(topics=DISCOVERY_TOPICS["go_language"], urls_per_topic=10)

    if go_urls:
        logger.info(f"Discovered {len(go_urls)} Go-related URLs")
        await discovery.add_to_crawler(go_urls)

    # Discover URLs for general programming
    prog_urls = await discovery.discover_multiple_topics(topics=DISCOVERY_TOPICS["programming"], urls_per_topic=5)

    if prog_urls:
        logger.info(f"Discovered {len(prog_urls)} programming URLs")
        await discovery.add_to_crawler(prog_urls)

    logger.info(f"Discovery complete. Total domains: {len(discovery.get_discovered_domains())}")

    return {
        "go_urls": len(go_urls),
        "programming_urls": len(prog_urls),
        "total_domains": len(discovery.get_discovered_domains()),
    }


if __name__ == "__main__":
    # Run discovery
    results = asyncio.run(run_seed_discovery())
    print(f"Discovery results: {results}")
