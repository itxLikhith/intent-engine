"""
SearXNG Service Client

This module provides a client for interacting with the SearXNG privacy search engine.
It handles search queries, result parsing, and integration with the Intent Engine.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SearXNGResult:
    """Represents a single search result from SearXNG"""

    url: str
    title: str
    content: str
    engine: str
    score: float
    category: str
    thumbnail: Optional[str] = None
    published_date: Optional[str] = None
    position: int = 0


@dataclass
class SearXNGResponse:
    """Represents a complete search response from SearXNG"""

    query: str
    results: List[SearXNGResult]
    number_of_results: int
    suggestions: List[str]
    corrections: List[str]
    infoboxes: List[Dict[str, Any]]
    processing_time: float
    engines: List[str]


import os


class SearXNGClient:
    """
    Client for SearXNG privacy search engine.

    Features:
    - Privacy-focused search (no tracking)
    - Multiple search engine aggregation
    - JSON API support
    - Intent-aware result filtering
    - Connection pooling for better performance
    """

    def __init__(self, base_url: str = None, redis_host: str = None, redis_port: int = 6379):
        """
        Initialize SearXNG client.

        Args:
            base_url: Base URL of the SearXNG instance.
                     Defaults to Docker service name for containerized deployments.
            redis_host: Redis host for caching (optional).
            redis_port: Redis port (default: 6379).
        """
        # Use environment variable or Docker service name by default for containerized deployments
        self.base_url = (base_url or os.getenv("SEARXNG_BASE_URL", "http://searxng:8080")).rstrip("/")
        self.timeout = 10.0  # seconds (reduced from 30s for better performance)
        self.connect_timeout = 3.0  # connection timeout
        self.cache_ttl = 600  # Cache TTL: 10 minutes

        # Initialize Redis cache if available
        self.cache = None
        if REDIS_AVAILABLE and redis_host:
            try:
                self.cache = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self.cache.ping()
                logger.info(f"SearXNG Redis cache connected: {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis cache: {e}")
                self.cache = None
        else:
            logger.info("SearXNG cache disabled (Redis not available)")

        # Initialize persistent HTTP client with connection pooling
        # This avoids creating a new connection for each request
        timeout_config = httpx.Timeout(timeout=self.timeout, connect=self.connect_timeout)
        limits = httpx.Limits(
            max_connections=100,  # Maximum concurrent connections
            max_keepalive_connections=20,  # Keep connections alive for reuse
        )
        self._client = httpx.AsyncClient(
            timeout=timeout_config, limits=limits, http2=True  # Enable HTTP/2 for better performance
        )

        logger.info(
            f"SearXNG client initialized with base URL: {self.base_url}, timeout: {self.timeout}s, connection pooling enabled"
        )

    async def search(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: str = "en",
        pageno: int = 1,
        safe_search: int = 0,
        format: str = "json",
        time_range: Optional[str] = None,
    ) -> SearXNGResponse:
        """
        Perform a search query on SearXNG.

        Args:
            query: Search query string
            categories: List of categories to search (e.g., ['general', 'news', 'science'])
            engines: List of specific engines to use (e.g., ['google', 'duckduckgo'])
            language: Language code (default: 'en')
            pageno: Page number for pagination
            safe_search: Safe search level (0=off, 1=moderate, 2=strict)
            format: Response format (default: 'json')
            time_range: Time range filter (e.g., 'day', 'week', 'month', 'year')

        Returns:
            SearXNGResponse with search results

        Raises:
            httpx.RequestError: If the request fails
        """
        # Generate cache key from query parameters
        cache_key_data = f"{query}:{categories}:{engines}:{language}:{pageno}:{safe_search}:{time_range}"
        cache_key = f"searxng:search:{hashlib.md5(cache_key_data.encode()).hexdigest()}"

        # Try to get from cache first
        if self.cache:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    logger.debug(f"Cache hit for query: {query[:50]}")
                    cached_data = json.loads(cached)
                    return SearXNGResponse(**cached_data)
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")

        # Build query parameters
        params = {
            "q": query,
            "format": format,
            "language": language,
            "pageno": pageno,
            "safe_search": safe_search,
        }

        # Add optional parameters
        if categories:
            params["categories"] = ",".join(categories)

        if engines:
            params["engines"] = ",".join(engines)

        if time_range:
            params["time_range"] = time_range

        # Make the request using persistent client with connection pooling
        try:
            logger.debug(f"Searching SearXNG: query='{query}', categories={categories}")
            response = await self._client.get(f"{self.base_url}/search", params=params)
            response.raise_for_status()

            data = response.json()
            logger.debug(
                f"SearXNG response: query={data.get('query')}, results_count={len(data.get('results', [])) if data.get('results') else 'None'}"
            )

            # Parse results - handle None case
            raw_results = data.get("results")
            if raw_results is None:
                logger.warning(
                    f"SearXNG returned None for results field, data keys: {list(data.keys()) if data else 'None'}"
                )
                raw_results = []

            results = self._parse_results(raw_results)

            response_obj = SearXNGResponse(
                query=data.get("query", query),
                results=results,
                number_of_results=data.get("number_of_results", len(results)),
                suggestions=data.get("suggestions", []),
                corrections=data.get("corrections", []),
                infoboxes=data.get("infoboxes", []),
                processing_time=data.get("search_duration", 0.0),
                engines=list(set(r.engine for r in results)),
            )

            # Cache the response
            if self.cache:
                try:
                    cache_data = {
                        "query": response_obj.query,
                        "results": [asdict(r) for r in response_obj.results],
                        "number_of_results": response_obj.number_of_results,
                        "suggestions": response_obj.suggestions,
                        "corrections": response_obj.corrections,
                        "infoboxes": response_obj.infoboxes,
                        "processing_time": response_obj.processing_time,
                        "engines": response_obj.engines,
                    }
                    self.cache.setex(cache_key, self.cache_ttl, json.dumps(cache_data))
                    logger.debug(f"Cached SearXNG response for query: {query[:50]}")
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")

            return response_obj

        except httpx.RequestError as e:
            logger.error(f"SearXNG request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing SearXNG response: {e}")
            raise

    def _parse_results(self, raw_results: List[Dict[str, Any]]) -> List[SearXNGResult]:
        """Parse raw search results into SearXNGResult objects."""
        results = []
        for idx, raw in enumerate(raw_results):
            if not raw:
                continue
            try:
                result = SearXNGResult(
                    url=raw.get("url", ""),
                    title=raw.get("title", ""),
                    content=raw.get("content", ""),
                    engine=raw.get("engine", "unknown"),
                    score=raw.get("score") or 0.0,
                    category=raw.get("category", "general"),
                    thumbnail=raw.get("thumbnail"),
                    published_date=raw.get("publishedDate"),
                    position=idx + 1,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse result: {e}")
                continue

        # Sort by score (highest first) - handle None values
        results.sort(key=lambda r: r.score if r.score is not None else 0.0, reverse=True)
        return results

    def get_engines(self) -> List[Dict[str, Any]]:
        """
        Get list of available search engines.

        Note: This requires accessing SearXNG's engine list endpoint
        which may need to be enabled in settings.
        """
        # This would require a separate endpoint or configuration
        # For now, return a default list of common engines
        return [
            {"name": "google", "category": "general"},
            {"name": "duckduckgo", "category": "general"},
            {"name": "brave", "category": "general"},
            {"name": "startpage", "category": "general"},
            {"name": "wikipedia", "category": "general"},
            {"name": "bing", "category": "general"},
        ]

    async def health_check(self) -> bool:
        """
        Check if SearXNG instance is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self._client.get(self.base_url)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"SearXNG health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client and release resources."""
        await self._client.aclose()
        logger.info("SearXNG client closed")


# Singleton instance
_searxng_client: Optional[SearXNGClient] = None


def get_searxng_client(base_url: Optional[str] = None) -> SearXNGClient:
    """
    Get or create SearXNG client singleton.

    Args:
        base_url: Optional base URL override.
                  Defaults to Docker service name for containerized deployments.

    Returns:
        SearXNGClient instance
    """
    global _searxng_client

    if _searxng_client is None or base_url:
        # Use Docker service name by default
        url = base_url or "http://searxng:8080"
        _searxng_client = SearXNGClient(base_url=url)

    return _searxng_client
