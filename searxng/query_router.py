"""
Intent Engine - Unified Query Router

Routes queries to optimal search backends based on intent analysis.
Supports Go Crawler, SearXNG, and custom intent indices.

Features:
- Intent-based routing logic
- Parallel query execution
- Timeout handling
- Fallback chain support
- Result aggregation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.schema import (
    EthicalDimension,
    IntentGoal,
    Recency,
    UniversalIntent,
)

logger = logging.getLogger(__name__)


class SearchBackend(Enum):
    """Available search backends"""

    GO_CRAWLER = "go_crawler"
    SEARXNG = "searxng"
    CUSTOM_INDEX = "custom_index"


@dataclass
class QueryRoute:
    """Routing configuration for a query"""

    backends: list[SearchBackend]
    weights: dict[SearchBackend, float] = field(default_factory=dict)
    parallel: bool = True
    timeout_ms: int = 4000
    fallback_chain: list[SearchBackend] = field(default_factory=list)
    max_results_per_backend: int = 20


@dataclass
class SearchResult:
    """Unified search result from any backend"""

    source: SearchBackend
    url: str
    title: str
    content: str
    score: float
    engine: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "source": self.source.value,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "engine": self.engine,
            "metadata": self.metadata,
        }


class UnifiedQueryRouter:
    """
    Routes queries to optimal backends based on intent analysis.

    Usage:
        router = UnifiedQueryRouter()
        route = router.route(intent)
        results = await router.execute_search(route, query)

    Routing Logic:
    1. Analyze intent goal (LEARN, COMPARISON, TROUBLESHOOTING, etc.)
    2. Check temporal requirements (BREAKING, RECENT, EVERGREEN)
    3. Consider ethical signals (privacy, openness, etc.)
    4. Assign backend weights and execute
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.default_timeout_ms = self.config.get("default_timeout_ms", 4000)
        self.default_max_results = self.config.get("max_results", 20)

        # Backend clients (initialized lazily)
        self._go_client = None
        self._searxng_client = None

        # Backend-specific configurations
        self.searxng_categories = self.config.get("searxng_categories", ["general", "news", "science", "tech"])

        logger.info("UnifiedQueryRouter initialized")

    def route(self, intent: UniversalIntent) -> QueryRoute:
        """
        Determine optimal routing strategy based on intent.

        Algorithm:
        1. Analyze intent goal
        2. Check temporal requirements
        3. Consider ethical signals
        4. Assign backend weights

        Args:
            intent: UniversalIntent object with extracted intent

        Returns:
            QueryRoute with routing configuration

        Examples:
            - Troubleshooting → SearXNG (community discussions)
            - Comparison → Go Crawler + SearXNG (comprehensive)
            - Breaking news → SearXNG news engines
            - Privacy queries → Go Crawler (curated index)
        """
        goal = intent.declared.goal
        temporal = intent.inferred.temporalIntent if intent.inferred else None
        ethical_signals = intent.inferred.ethicalSignals if intent.inferred else []

        # Rule 1: Troubleshooting → prefer community discussions (SearXNG)
        if goal == IntentGoal.TROUBLESHOOTING:
            logger.debug("Routing troubleshooting query to SearXNG")
            return QueryRoute(
                backends=[SearchBackend.SEARXNG],
                weights={SearchBackend.SEARXNG: 1.0},
                parallel=False,
                timeout_ms=3000,
                fallback_chain=[SearchBackend.GO_CRAWLER],
                max_results_per_backend=self.default_max_results,
            )

        # Rule 2: Comparison → use both for comprehensive coverage
        if goal == IntentGoal.COMPARISON:
            logger.debug("Routing comparison query to both backends")
            return QueryRoute(
                backends=[SearchBackend.GO_CRAWLER, SearchBackend.SEARXNG],
                weights={
                    SearchBackend.GO_CRAWLER: 0.6,
                    SearchBackend.SEARXNG: 0.4,
                },
                parallel=True,
                timeout_ms=5000,
                fallback_chain=[SearchBackend.GO_CRAWLER],
                max_results_per_backend=15,
            )

        # Rule 3: Purchase/Local Service → prefer Go Crawler (structured data)
        if goal in [IntentGoal.PURCHASE, IntentGoal.LOCAL_SERVICE]:
            logger.debug(f"Routing {goal.value} query to Go Crawler")
            return QueryRoute(
                backends=[SearchBackend.GO_CRAWLER],
                weights={SearchBackend.GO_CRAWLER: 1.0},
                parallel=False,
                timeout_ms=3000,
                fallback_chain=[SearchBackend.SEARXNG],
                max_results_per_backend=self.default_max_results,
            )

        # Rule 4: Breaking news → SearXNG news engines
        if temporal and temporal.recency == Recency.BREAKING:
            logger.debug("Routing breaking news query to SearXNG")
            return QueryRoute(
                backends=[SearchBackend.SEARXNG],
                weights={SearchBackend.SEARXNG: 1.0},
                parallel=False,
                timeout_ms=2000,
                fallback_chain=[SearchBackend.GO_CRAWLER],
                max_results_per_backend=10,
            )

        # Rule 5: Privacy-focused queries → prefer Go crawler (curated)
        privacy_signals = [s for s in ethical_signals if s.dimension == EthicalDimension.PRIVACY]
        if privacy_signals:
            logger.debug("Routing privacy-focused query to Go Crawler")
            return QueryRoute(
                backends=[SearchBackend.GO_CRAWLER],
                weights={SearchBackend.GO_CRAWLER: 1.0},
                parallel=False,
                timeout_ms=3000,
                fallback_chain=[SearchBackend.SEARXNG],
                max_results_per_backend=self.default_max_results,
            )

        # Rule 6: Learning/Information → hybrid approach
        if goal in [IntentGoal.LEARN, IntentGoal.FIND_INFORMATION]:
            logger.debug("Routing learning query to both backends (hybrid)")
            return QueryRoute(
                backends=[SearchBackend.GO_CRAWLER, SearchBackend.SEARXNG],
                weights={
                    SearchBackend.GO_CRAWLER: 0.5,
                    SearchBackend.SEARXNG: 0.5,
                },
                parallel=True,
                timeout_ms=self.default_timeout_ms,
                fallback_chain=[SearchBackend.GO_CRAWLER, SearchBackend.SEARXNG],
                max_results_per_backend=self.default_max_results,
            )

        # Default: hybrid approach for unknown goals
        logger.debug("Using default hybrid routing")
        return QueryRoute(
            backends=[SearchBackend.GO_CRAWLER, SearchBackend.SEARXNG],
            weights={
                SearchBackend.GO_CRAWLER: 0.5,
                SearchBackend.SEARXNG: 0.5,
            },
            parallel=True,
            timeout_ms=self.default_timeout_ms,
            fallback_chain=[SearchBackend.GO_CRAWLER, SearchBackend.SEARXNG],
            max_results_per_backend=self.default_max_results,
        )

    async def execute_search(self, route: QueryRoute, query: str) -> list[SearchResult]:
        """
        Execute search across configured backends.

        Supports:
        - Parallel execution (asyncio.gather)
        - Timeout handling (asyncio.wait_for)
        - Fallback chain (sequential retry)

        Args:
            route: QueryRoute with routing configuration
            query: Search query string

        Returns:
            List of SearchResult from all backends
        """
        tasks = []
        results = []

        # Create search tasks for each backend
        for backend in route.backends:
            weight = route.weights.get(backend, 0.5)
            max_results = int(route.max_results_per_backend * weight)

            task = self._search_backend(backend, query, max_results)
            tasks.append((backend, task))

        logger.info(f"Executing search across {len(tasks)} backends: {[b.value for b, _ in tasks]}")

        # Execute in parallel or sequentially
        if route.parallel:
            backend_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            # Pair backends with results
            for (backend, _), result in zip(tasks, backend_results, strict=False):
                if isinstance(result, Exception):
                    logger.warning(f"Backend {backend.value} failed: {result}")
                    continue
                results.extend(result)
        else:
            # Sequential execution with fallback
            for backend, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=route.timeout_ms / 1000)
                    results.extend(result)
                    logger.debug(f"Backend {backend.value} returned {len(result)} results")
                    break  # Success, don't try fallback
                except TimeoutError:
                    logger.warning(f"Backend {backend.value} timed out")
                    continue
                except Exception as e:
                    logger.warning(f"Backend {backend.value} failed: {e}")
                    # Continue to next backend in fallback chain

        logger.info(f"Search completed: {len(results)} total results")
        return results

    async def _search_backend(self, backend: SearchBackend, query: str, max_results: int) -> list[SearchResult]:
        """Search a specific backend"""
        if backend == SearchBackend.GO_CRAWLER:
            return await self._search_go_crawler(query, max_results)
        elif backend == SearchBackend.SEARXNG:
            return await self._search_searxng(query, max_results)
        elif backend == SearchBackend.CUSTOM_INDEX:
            return await self._search_custom_index(query, max_results)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    async def _search_go_crawler(self, query: str, max_results: int) -> list[SearchResult]:
        """Search Go crawler index"""
        from go_search_client import GoSearchClient

        try:
            # Create client and check health
            client = GoSearchClient(base_url=self._get_go_crawler_url(), timeout=5.0)
            try:
                # Health check (async)
                health = await client.health_check()
                if health.get("status") != "healthy":
                    logger.warning(f"Go crawler unhealthy: {health}")
                    return []

                # Search
                response = await client.search(query=query, limit=max_results)

                if not response or not response.results:
                    return []

                results = [
                    SearchResult(
                        source=SearchBackend.GO_CRAWLER,
                        url=r.url,
                        title=r.title,
                        content=r.content,
                        score=r.score,
                        engine="go-crawler",
                        metadata={
                            "rank": r.rank,
                            "match_reasons": r.match_reasons or [],
                        },
                    )
                    for r in response.results
                ]

                logger.debug(f"Go crawler returned {len(results)} results")
                return results

            finally:
                await client.close()

        except Exception as e:
            logger.error(f"Go crawler search failed: {e}")
            return []

    async def _search_searxng(self, query: str, max_results: int) -> list[SearchResult]:
        """Search SearXNG"""
        from searxng.client import get_searxng_client

        try:
            client = get_searxng_client(self._get_searxng_url())
            response = await client.search(
                query=query,
                categories=self.searxng_categories,
                pageno=1,
            )

            if not response or not response.results:
                return []

            results = [
                SearchResult(
                    source=SearchBackend.SEARXNG,
                    url=r.url,
                    title=r.title,
                    content=r.content,
                    score=r.score,
                    engine=r.engine,
                    metadata={
                        "category": r.category,
                        "published_date": r.published_date,
                        "position": r.position,
                    },
                )
                for r in response.results[:max_results]
            ]

            logger.debug(f"SearXNG returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"SearXNG search failed: {e}")
            return []

    async def _search_custom_index(self, query: str, max_results: int) -> list[SearchResult]:
        """Search custom intent-indexed content (Qdrant)"""
        try:
            from core.vector_store import get_vector_store

            vector_store = get_vector_store()

            # Get embedding for query
            from core.embedding_service import get_embedding_service

            embedding_service = get_embedding_service()
            embedding = embedding_service.encode_text(query)

            if embedding is None:
                logger.warning("Failed to generate query embedding")
                return []

            # Search vector store
            results = vector_store.search_similar(query_embedding=embedding.tolist(), limit=max_results)

            # Convert to SearchResult
            search_results = [
                SearchResult(
                    source=SearchBackend.CUSTOM_INDEX,
                    url=r.url,
                    title=r.intent_tags.get("title", "Untitled"),
                    content=r.intent_tags.get("description", ""),
                    score=r.score,
                    engine="qdrant",
                    metadata={
                        "intent_tags": r.intent_tags,
                        "primary_goal": r.intent_tags.get("primary_goal"),
                    },
                )
                for r in results
            ]

            logger.debug(f"Custom index returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Custom index search failed: {e}")
            return []

    def _get_go_crawler_url(self) -> str:
        """Get Go crawler URL from config or environment"""
        # Always use Docker service name in containerized environment
        return self.config.get("go_crawler_url", "http://go-search-api:8080")

    def _get_searxng_url(self) -> str:
        """Get SearXNG URL from config or environment"""
        # Always use Docker service name in containerized environment
        return self.config.get("searxng_url", "http://searxng:8080")


# Singleton instance
_router_instance: UnifiedQueryRouter | None = None
_router_lock = asyncio.Lock()


def get_query_router(config: dict[str, Any] | None = None) -> UnifiedQueryRouter:
    """
    Get or create query router singleton.

    Args:
        config: Optional configuration override

    Returns:
        UnifiedQueryRouter instance
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = UnifiedQueryRouter(config)
    return _router_instance


async def reset_query_router():
    """Reset router singleton (useful for testing)"""
    global _router_instance
    async with _router_lock:
        _router_instance = None
