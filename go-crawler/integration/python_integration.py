"""
Intent Engine - Go Crawler Integration

This module provides integration between the Python Intent Engine
and the Go-based crawler/search backend.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from Go Search API"""
    url: str
    title: str
    content: str
    score: float
    rank: int


@dataclass
class SearchResponse:
    """Represents a search response from Go Search API"""
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float
    engines_used: List[str]
    ranking_applied: bool


class GoSearchClient:
    """Client for Go Search API"""

    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        timeout: int = 30,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """Close the client session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check Search API health"""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/health") as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def get_stats(self) -> Dict[str, Any]:
        """Get Search API statistics"""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/stats") as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """
        Search using Go Search API

        Args:
            query: Search query
            limit: Maximum results to return
            filters: Optional filters

        Returns:
            SearchResponse object
        """
        session = await self._get_session()
        
        payload = {
            "query": query,
            "limit": limit,
        }
        
        if filters:
            payload["filters"] = filters

        try:
            async with session.post(
                f"{self.base_url}/api/v1/search",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

                # Parse results
                results = [
                    SearchResult(
                        url=r["url"],
                        title=r["title"],
                        content=r["content"],
                        score=r["score"],
                        rank=r["rank"],
                    )
                    for r in data.get("results", [])
                ]

                return SearchResponse(
                    query=data["query"],
                    results=results,
                    total_results=data["total_results"],
                    processing_time_ms=data["processing_time_ms"],
                    engines_used=data.get("engines_used", []),
                    ranking_applied=data.get("ranking_applied", False),
                )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise


class UnifiedSearchIntegration:
    """
    Integrates Go Search API with Python Intent Engine
    
    Usage:
        integration = UnifiedSearchIntegration()
        await integration.initialize()
        
        # Search
        results = await integration.search("golang microservices")
        
        # Cleanup
        await integration.close()
    """

    def __init__(self, go_search_url: str = "http://localhost:8081"):
        self.go_search_url = go_search_url
        self.go_client: Optional[GoSearchClient] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the integration"""
        self.go_client = GoSearchClient(base_url=self.go_search_url)
        
        # Check if Go Search API is available
        health = await self.go_client.health_check()
        if health.get("status") == "healthy":
            logger.info("Go Search API integration initialized successfully")
            self._initialized = True
        else:
            logger.warning(f"Go Search API not healthy: {health}")
            self._initialized = False

    async def close(self):
        """Close the integration"""
        if self.go_client:
            await self.go_client.close()

    async def search(
        self,
        query: str,
        limit: int = 10,
        use_go_backend: bool = True,
    ) -> SearchResponse:
        """
        Search with optional Go backend
        
        Args:
            query: Search query
            limit: Result limit
            use_go_backend: If True, use Go Search API
            
        Returns:
            SearchResponse
        """
        if use_go_backend and self._initialized:
            try:
                return await self.go_client.search(query, limit)
            except Exception as e:
                logger.error(f"Go Search API failed, fallback: {e}")
                # Could fallback to SearXNG here
        
        # Fallback to SearXNG or other search backend
        raise NotImplementedError("Fallback search not implemented")

    async def get_stats(self) -> Dict[str, Any]:
        """Get Go Search API statistics"""
        if not self._initialized:
            return {"error": "Not initialized"}
        
        return await self.go_client.get_stats()


# Convenience functions for synchronous usage
def search_sync(
    query: str,
    limit: int = 10,
    go_search_url: str = "http://localhost:8081",
) -> SearchResponse:
    """
    Synchronous search using Go Search API
    
    Usage:
        results = search_sync("golang microservices")
    """
    async def _search():
        client = GoSearchClient(base_url=go_search_url)
        try:
            return await client.search(query, limit)
        finally:
            await client.close()
    
    return asyncio.run(_search())


def health_check_sync(go_search_url: str = "http://localhost:8081") -> Dict[str, Any]:
    """
    Synchronous health check
    
    Usage:
        health = health_check_sync()
    """
    async def _health():
        client = GoSearchClient(base_url=go_search_url)
        try:
            return await client.health_check()
        finally:
            await client.close()
    
    return asyncio.run(_health())


if __name__ == "__main__":
    # Example usage
    print("Testing Go Search API integration...")
    
    # Health check
    health = health_check_sync()
    print(f"Health: {health}")
    
    # Search
    try:
        results = search_sync("golang programming", limit=5)
        print(f"\nSearch Results for 'golang programming':")
        print(f"Total: {results.total_results}")
        print(f"Processing Time: {results.processing_time_ms}ms")
        print(f"Engines: {results.engines_used}")
        print("\nTop Results:")
        for i, result in enumerate(results.results[:5], 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Score: {result.score:.3f}")
            print()
    except Exception as e:
        print(f"Search failed: {e}")
