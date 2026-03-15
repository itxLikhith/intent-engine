"""
Intent Engine - Go Search API Client

Async client for integrating Go crawler search with Python Intent Engine.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class GoSearchResult:
    """Represents a single search result from Go API."""

    url: str
    title: str
    content: str
    score: float
    rank: int
    match_reasons: Optional[List[str]] = None


@dataclass
class GoSearchResponse:
    """Represents search response from Go API."""

    query: str
    results: List[GoSearchResult]
    total_results: int
    processing_time_ms: float
    engines_used: List[str]
    ranking_applied: bool


class GoSearchClient:
    """Async client for Go Search API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        timeout: float = 10.0,
        enabled: bool = True,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.enabled = enabled
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check Go Search API health."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                return {"status": "unhealthy", "error": f"Status {response.status}"}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def search(
        self, query: str, limit: int = 10, filters: Optional[Dict] = None
    ) -> Optional[GoSearchResponse]:
        """Search using Go crawler index."""
        if not self.enabled:
            return None

        try:
            session = await self._get_session()
            payload = {"query": query, "limit": limit}
            if filters:
                payload["filters"] = filters

            async with session.post(
                f"{self.base_url}/api/v1/search", json=payload
            ) as response:
                if response.status != 200:
                    logger.warning(f"Search failed with status {response.status}")
                    return None

                data = await response.json()

                # Parse results
                results = [
                    GoSearchResult(
                        url=r["url"],
                        title=r["title"],
                        content=r.get("content", ""),
                        score=r.get("score", 0.0),
                        rank=r.get("rank", i + 1),
                        match_reasons=r.get("match_reasons"),
                    )
                    for i, r in enumerate(data.get("results", []))
                ]

                return GoSearchResponse(
                    query=data.get("query", query),
                    results=results,
                    total_results=data.get("total_results", len(results)),
                    processing_time_ms=data.get("processing_time_ms", 0),
                    engines_used=data.get("engines_used", ["go-crawler"]),
                    ranking_applied=data.get("ranking_applied", True),
                )

        except asyncio.TimeoutError:
            logger.error(f"Search timeout for query: {query}")
            return None
        except Exception as e:
            logger.error(f"Search error: {e}")
            return None

    async def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get crawler/indexer statistics."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/stats") as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return None

    async def add_seed_urls(
        self, urls: List[str], priority: int = 5, depth: int = 0
    ) -> bool:
        """Add seed URLs to crawl queue."""
        try:
            session = await self._get_session()
            payload = {"urls": urls, "priority": priority, "depth": depth}

            async with session.post(
                f"{self.base_url}/api/v1/crawl/seed", json=payload
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Add seed URLs error: {e}")
            return False


# Convenience functions for synchronous usage
def search_sync(
    query: str,
    limit: int = 10,
    base_url: str = "http://localhost:8081",
    timeout: float = 10.0,
) -> Optional[GoSearchResponse]:
    """Synchronous search wrapper."""
    client = GoSearchClient(base_url=base_url, timeout=timeout)
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(client.search(query, limit))
    finally:
        loop.run_until_complete(client.close())


def health_check_sync(base_url: str = "http://localhost:8081") -> Dict[str, Any]:
    """Synchronous health check wrapper."""
    client = GoSearchClient(base_url=base_url)
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(client.health_check())
    finally:
        loop.run_until_complete(client.close())


# Example usage
if __name__ == "__main__":
    import json

    async def main():
        client = GoSearchClient()

        # Health check
        health = await client.health_check()
        print(f"Health: {json.dumps(health, indent=2)}")

        # Search
        if health.get("status") == "healthy":
            results = await client.search("golang programming", limit=5)
            if results:
                print(f"\nFound {results.total_results} results")
                for r in results.results:
                    print(f"{r.rank}. {r.title}")
                    print(f"   {r.url}")
                    print()

        # Stats
        stats = await client.get_stats()
        if stats:
            print(f"Stats: {json.dumps(stats, indent=2)}")

        await client.close()

    asyncio.run(main())
