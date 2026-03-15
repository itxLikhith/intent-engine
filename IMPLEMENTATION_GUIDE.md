# Intent Engine: Implementation Guide

**Version:** 1.0.0  
**Date:** March 15, 2026  
**Purpose:** Step-by-step implementation instructions for the Unified Architecture

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Phase 1: Query Router](#phase-1-query-router)
3. [Phase 2: Intent Indexing](#phase-2-intent-indexing)
4. [Phase 3: Distributed Processing](#phase-3-distributed-processing)
5. [Phase 4: Observability](#phase-4-observability)
6. [Testing Strategy](#testing-strategy)
7. [Deployment Guide](#deployment-guide)

---

## Quick Start

### Prerequisites

```bash
# Required
- Python 3.11+
- Go 1.21+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis/Valkey 8+

# Recommended
- Kubernetes cluster (minikube for dev)
- Apache Kafka (or Redpanda for dev)
- Qdrant vector DB
```

### Development Setup (15 minutes)

```bash
# 1. Clone and setup
git clone https://github.com/itxLikhith/intent-engine.git
cd intent-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate on Windows
pip install -e ".[dev]"

# 3. Start infrastructure
docker-compose up -d postgres redis searxng

# 4. Run migrations
python scripts/init_db_standalone.py

# 5. Test existing functionality
python main.py demo-search
curl http://localhost:8000/search -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"best privacy-focused email"}'
```

---

## Phase 1: Query Router

### Step 1.1: Create Query Router Module

**File:** `searxng/query_router.py`

```python
"""
Intent Engine - Unified Query Router

Routes queries to optimal search backends based on intent.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from core.schema import (
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
    engine: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedQueryRouter:
    """
    Routes queries to optimal backends based on intent analysis.
    
    Usage:
        router = UnifiedQueryRouter()
        route = router.route(intent)
        results = await router.execute_search(route, query)
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.default_timeout_ms = self.config.get("default_timeout_ms", 4000)
        self.default_max_results = self.config.get("max_results", 20)
        
        # Backend clients (initialized lazily)
        self._go_client = None
        self._searxng_client = None
    
    def route(self, intent: UniversalIntent) -> QueryRoute:
        """
        Determine optimal routing strategy based on intent.
        
        Algorithm:
        1. Analyze intent goal
        2. Check temporal requirements
        3. Consider ethical signals
        4. Assign backend weights
        """
        goal = intent.declared.goal
        temporal = intent.inferred.temporalIntent
        ethical_signals = intent.inferred.ethicalSignals
        
        # Rule 1: Troubleshooting → prefer community discussions (SearXNG)
        if goal == IntentGoal.TROUBLESHOOTING:
            return QueryRoute(
                backends=[SearchBackend.SEARXNG],
                weights={SearchBackend.SEARXNG: 1.0},
                parallel=False,
                timeout_ms=3000,
                fallback_chain=[SearchBackend.GO_CRAWLER],
                max_results_per_backend=self.default_max_results
            )
        
        # Rule 2: Comparison → use both for comprehensive coverage
        if goal == IntentGoal.COMPARISON:
            return QueryRoute(
                backends=[SearchBackend.GO_CRAWLER, SearchBackend.SEARXNG],
                weights={
                    SearchBackend.GO_CRAWLER: 0.6,
                    SearchBackend.SEARXNG: 0.4
                },
                parallel=True,
                timeout_ms=5000,
                fallback_chain=[SearchBackend.GO_CRAWLER],
                max_results_per_backend=15
            )
        
        # Rule 3: Breaking news → SearXNG news engines
        if temporal and temporal.recency == Recency.BREAKING:
            return QueryRoute(
                backends=[SearchBackend.SEARXNG],
                weights={SearchBackend.SEARXNG: 1.0},
                parallel=False,
                timeout_ms=2000,
                fallback_chain=[SearchBackend.GO_CRAWLER],
                max_results_per_backend=10
            )
        
        # Rule 4: Privacy-focused queries → prefer Go crawler (curated)
        privacy_signals = [
            s for s in ethical_signals 
            if s.dimension.value == "privacy"
        ]
        if privacy_signals:
            return QueryRoute(
                backends=[SearchBackend.GO_CRAWLER],
                weights={SearchBackend.GO_CRAWLER: 1.0},
                parallel=False,
                timeout_ms=3000,
                fallback_chain=[SearchBackend.SEARXNG],
                max_results_per_backend=self.default_max_results
            )
        
        # Default: hybrid approach
        return QueryRoute(
            backends=[SearchBackend.GO_CRAWLER, SearchBackend.SEARXNG],
            weights={
                SearchBackend.GO_CRAWLER: 0.5,
                SearchBackend.SEARXNG: 0.5
            },
            parallel=True,
            timeout_ms=self.default_timeout_ms,
            fallback_chain=[SearchBackend.GO_CRAWLER, SearchBackend.SEARXNG],
            max_results_per_backend=self.default_max_results
        )
    
    async def execute_search(
        self, 
        route: QueryRoute, 
        query: str
    ) -> list[SearchResult]:
        """
        Execute search across configured backends.
        
        Supports:
        - Parallel execution
        - Timeout handling
        - Fallback chain
        """
        tasks = []
        results = []
        
        # Create search tasks for each backend
        for backend in route.backends:
            weight = route.weights.get(backend, 0.5)
            max_results = int(route.max_results_per_backend * weight)
            
            task = self._search_backend(backend, query, max_results)
            tasks.append((backend, task))
        
        # Execute in parallel or sequentially
        if route.parallel:
            backend_results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            # Pair backends with results
            for (backend, _), result in zip(tasks, backend_results, strict=False):
                if isinstance(result, Exception):
                    logger.warning(f"Backend {backend} failed: {result}")
                    continue
                results.extend(result)
        else:
            # Sequential execution with fallback
            for backend, task in tasks:
                try:
                    result = await asyncio.wait_for(
                        task, 
                        timeout=route.timeout_ms / 1000
                    )
                    results.extend(result)
                    break  # Success, don't try fallback
                except asyncio.TimeoutError:
                    logger.warning(f"Backend {backend} timed out")
                    continue
                except Exception as e:
                    logger.warning(f"Backend {backend} failed: {e}")
                    continue
        
        return results
    
    async def _search_backend(
        self, 
        backend: SearchBackend, 
        query: str, 
        max_results: int
    ) -> list[SearchResult]:
        """Search a specific backend"""
        if backend == SearchBackend.GO_CRAWLER:
            return await self._search_go_crawler(query, max_results)
        elif backend == SearchBackend.SEARXNG:
            return await self._search_searxng(query, max_results)
        elif backend == SearchBackend.CUSTOM_INDEX:
            return await self._search_custom_index(query, max_results)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    async def _search_go_crawler(
        self, 
        query: str, 
        max_results: int
    ) -> list[SearchResult]:
        """Search Go crawler index"""
        from go_search_client import GoSearchClient
        
        if self._go_client is None:
            self._go_client = GoSearchClient(
                base_url=self.config.get("go_crawler_url", "http://localhost:8081")
            )
        
        response = await self._go_client.search(
            query=query,
            limit=max_results
        )
        
        if not response:
            return []
        
        return [
            SearchResult(
                source=SearchBackend.GO_CRAWLER,
                url=r.url,
                title=r.title,
                content=r.content,
                score=r.score,
                engine="go-crawler",
                metadata={"rank": r.rank, "match_reasons": r.match_reasons}
            )
            for r in response.results
        ]
    
    async def _search_searxng(
        self, 
        query: str, 
        max_results: int
    ) -> list[SearchResult]:
        """Search SearXNG"""
        from searxng.client import SearXNGClient
        
        if self._searxng_client is None:
            self._searxng_client = SearXNGClient(
                base_url=self.config.get("searxng_url", "http://localhost:8080")
            )
        
        results = await self._searxng_client.search(
            query=query,
            max_results=max_results
        )
        
        return [
            SearchResult(
                source=SearchBackend.SEARXNG,
                url=r.url,
                title=r.title,
                content=r.content,
                score=r.get("score", 0.5),
                engine=r.get("engine", "unknown"),
                metadata={"category": r.get("category", "general")}
            )
            for r in results
        ]
    
    async def _search_custom_index(
        self, 
        query: str, 
        max_results: int
    ) -> list[SearchResult]:
        """Search custom intent-indexed content"""
        # TODO: Implement Qdrant/vector DB search
        logger.info("Custom index search not yet implemented")
        return []


# Singleton instance
_router_instance: Optional[UnifiedQueryRouter] = None


def get_query_router(config: Optional[dict[str, Any]] = None) -> UnifiedQueryRouter:
    """Get or create query router singleton"""
    global _router_instance
    if _router_instance is None:
        _router_instance = UnifiedQueryRouter(config)
    return _router_instance
```

### Step 1.2: Update Unified Search Endpoint

**File:** `searxng/unified_search.py` (modify existing)

```python
# Add to existing unified_search.py

from .query_router import get_query_router, SearchResult as UnifiedSearchResult

class UnifiedSearchService:
    """Enhanced unified search with query routing"""
    
    def __init__(self):
        self.query_router = get_query_router()
        # ... existing initialization ...
    
    async def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        """
        Enhanced search with intent-based routing.
        """
        # Step 1: Extract intent (existing)
        intent = await self._extract_intent(request)
        
        # Step 2: Route query (NEW)
        route = self.query_router.route(intent)
        
        # Step 3: Execute federated search (NEW)
        raw_results = await self.query_router.execute_search(
            route=route,
            query=request.query
        )
        
        # Step 4: Aggregate and rank (existing + enhancement)
        ranked_results = await self._rank_with_intent(raw_results, intent)
        
        # Step 5: Build response
        return UnifiedSearchResponse(
            query=request.query,
            results=ranked_results,
            intent=intent,
            metrics={
                "totalResults": len(raw_results),
                "backendDistribution": self._count_by_backend(raw_results),
                "processingTimeMs": self._get_processing_time(),
            }
        )
```

### Step 1.3: Result Aggregator

**File:** `searxng/result_aggregator.py` (new)

```python
"""
Intent Engine - Result Aggregation and Deduplication
"""

import hashlib
from dataclasses import dataclass
from typing import Any

from .query_router import SearchResult


@dataclass
class AggregatedResult:
    """Result after aggregation and deduplication"""
    url: str
    title: str
    content: str
    sources: list[str]  # Which backends returned this
    best_score: float
    metadata: dict[str, Any]


class ResultAggregator:
    """
    Aggregates results from multiple backends.
    
    Features:
    - URL-based deduplication
    - Score normalization
    - Source attribution
    """
    
    def __init__(self, dedup_threshold: float = 0.95):
        self.dedup_threshold = dedup_threshold
    
    def aggregate(self, results: list[SearchResult]) -> list[AggregatedResult]:
        """
        Aggregate and deduplicate results.
        
        Algorithm:
        1. Group by URL (normalized)
        2. Merge duplicate entries
        3. Normalize scores across backends
        4. Sort by final score
        """
        # Group by URL hash
        url_groups: dict[str, list[SearchResult]] = {}
        
        for result in results:
            url_key = self._normalize_url(result.url)
            if url_key not in url_groups:
                url_groups[url_key] = []
            url_groups[url_key].append(result)
        
        # Merge duplicates
        aggregated = []
        for url_key, group in url_groups.items():
            merged = self._merge_results(group)
            aggregated.append(merged)
        
        # Sort by score
        aggregated.sort(key=lambda x: x.best_score, reverse=True)
        
        return aggregated
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication"""
        # Remove tracking parameters
        from urllib.parse import urlparse, parse_qs, urlencode
        
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        # Remove common tracking params
        tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign',
            'fbclid', 'gclid', 'ref', 'source'
        }
        filtered_params = {
            k: v for k, v in query_params.items() 
            if k not in tracking_params
        }
        
        # Rebuild URL
        normalized = parsed._replace(
            query=urlencode(filtered_params, doseq=True)
        )
        
        return normalized.geturl()
    
    def _merge_results(self, results: list[SearchResult]) -> AggregatedResult:
        """Merge multiple results for same URL"""
        # Use best title (longest)
        best_result = max(results, key=lambda r: len(r.title))
        
        # Combine sources
        sources = list(set(str(r.source) for r in results))
        
        # Best score
        best_score = max(r.score for r in results)
        
        # Merge metadata
        merged_metadata = {}
        for r in results:
            merged_metadata.update(r.metadata)
        
        return AggregatedResult(
            url=best_result.url,
            title=best_result.title,
            content=best_result.content,
            sources=sources,
            best_score=best_score,
            metadata=merged_metadata
        )
    
    def _count_by_backend(self, results: list[SearchResult]) -> dict[str, int]:
        """Count results per backend"""
        counts = {}
        for result in results:
            source = str(result.source)
            counts[source] = counts.get(source, 0) + 1
        return counts
```

---

## Phase 2: Intent Indexing

### Step 2.1: Intent Extractor for Web Content

**File:** `extraction/web_extractor.py` (new)

```python
"""
Intent Engine - Web Content Intent Extraction

Extracts intent tags from web pages for indexing.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from core.schema import (
    Complexity,
    IntentGoal,
    ResultType,
    UseCase,
)

logger = logging.getLogger(__name__)


@dataclass
class WebIntent:
    """Intent metadata for a web page"""
    url: str
    primary_goal: IntentGoal
    use_cases: list[UseCase]
    result_type: ResultType
    complexity: Complexity
    skill_level: str  # beginner, intermediate, advanced
    topics: list[str]
    confidence: float


class WebIntentExtractor:
    """
    Extracts intent metadata from web page content.
    
    Usage:
        extractor = WebIntentExtractor()
        intent = await extractor.extract_from_url(url)
    """
    
    def __init__(self):
        # Intent keyword dictionaries
        self.goal_keywords = {
            IntentGoal.LEARN: [
                "tutorial", "guide", "how to", "learn", "introduction",
                "basics", "fundamentals", "course", "lesson"
            ],
            IntentGoal.COMPARISON: [
                "vs", "versus", "compare", "comparison", "better",
                "alternative", "review", "top 10", "best"
            ],
            IntentGoal.TROUBLESHOOTING: [
                "fix", "error", "problem", "issue", "troubleshoot",
                "debug", "solve", "not working"
            ],
            IntentGoal.FIND_INFORMATION: [
                "what is", "definition", "explain", "overview",
                "introduction", "guide"
            ],
        }
        
        self.use_case_keywords = {
            UseCase.LEARNING: ["tutorial", "learn", "study", "course"],
            UseCase.COMPARISON: ["compare", "review", "best", "top"],
            UseCase.TROUBLESHOOTING: ["fix", "error", "problem"],
            UseCase.PROFESSIONAL_DEVELOPMENT: ["career", "professional", "skills"],
        }
    
    async def extract_from_url(self, url: str) -> Optional[WebIntent]:
        """Extract intent from a web page"""
        try:
            # Fetch content
            content = await self._fetch_content(url)
            if not content:
                return None
            
            # Extract intent
            return self.extract_from_content(url, content)
        
        except Exception as e:
            logger.error(f"Error extracting intent from {url}: {e}")
            return None
    
    def extract_from_content(
        self, 
        url: str, 
        content: str
    ) -> WebIntent:
        """Extract intent from page content"""
        content_lower = content.lower()[:5000]  # First 5K chars
        
        # Detect primary goal
        primary_goal = self._detect_goal(content_lower)
        
        # Detect use cases
        use_cases = self._detect_use_cases(content_lower)
        
        # Detect result type
        result_type = self._detect_result_type(content_lower)
        
        # Detect complexity
        complexity = self._detect_complexity(content_lower)
        
        # Detect skill level
        skill_level = self._detect_skill_level(content_lower)
        
        # Extract topics (simple keyword extraction)
        topics = self._extract_topics(content_lower)
        
        # Calculate confidence
        confidence = self._calculate_confidence(primary_goal, use_cases)
        
        return WebIntent(
            url=url,
            primary_goal=primary_goal,
            use_cases=use_cases,
            result_type=result_type,
            complexity=complexity,
            skill_level=skill_level,
            topics=topics,
            confidence=confidence
        )
    
    def _detect_goal(self, content: str) -> IntentGoal:
        """Detect primary intent goal"""
        goal_scores = {}
        
        for goal, keywords in self.goal_keywords.items():
            score = sum(1 for kw in keywords if kw in content)
            goal_scores[goal] = score
        
        # Return highest scoring goal
        if goal_scores:
            return max(goal_scores, key=goal_scores.get)
        
        return IntentGoal.FIND_INFORMATION
    
    def _detect_use_cases(self, content: str) -> list[UseCase]:
        """Detect use cases"""
        detected = []
        
        for use_case, keywords in self.use_case_keywords.items():
            if any(kw in content for kw in keywords):
                detected.append(use_case)
        
        return detected if detected else [UseCase.LEARNING]
    
    def _detect_result_type(self, content: str) -> ResultType:
        """Detect expected result type"""
        if any(kw in content for kw in ["tutorial", "guide", "steps"]):
            return ResultType.TUTORIAL
        elif any(kw in content for kw in ["tool", "software", "app"]):
            return ResultType.TOOL
        elif any(kw in content for kw in ["buy", "price", "purchase"]):
            return ResultType.MARKETPLACE
        else:
            return ResultType.ANSWER
    
    def _detect_complexity(self, content: str) -> Complexity:
        """Detect content complexity"""
        # Simple heuristics
        word_count = len(content.split())
        
        if word_count < 500:
            return Complexity.SIMPLE
        elif word_count > 2000:
            return Complexity.ADVANCED
        else:
            return Complexity.MODERATE
    
    def _detect_skill_level(self, content: str) -> str:
        """Detect target skill level"""
        if any(kw in content for kw in ["beginner", "introduction", "basics"]):
            return "beginner"
        elif any(kw in content for kw in ["advanced", "expert", "deep dive"]):
            return "advanced"
        else:
            return "intermediate"
    
    def _extract_topics(self, content: str) -> list[str]:
        """Extract main topics (simple TF-based)"""
        # Remove stopwords
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"
        }
        
        words = content.split()
        word_freq = {}
        
        for word in words:
            word = word.lower().strip(".,!?;:")
            if word not in stopwords and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top 5 topics
        sorted_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:5]]
    
    def _calculate_confidence(
        self, 
        goal: IntentGoal, 
        use_cases: list[UseCase]
    ) -> float:
        """Calculate extraction confidence"""
        base_confidence = 0.5
        
        # Boost for clear goal detection
        if goal != IntentGoal.FIND_INFORMATION:
            base_confidence += 0.2
        
        # Boost for multiple use cases
        if len(use_cases) > 1:
            base_confidence += 0.1 * len(use_cases)
        
        return min(1.0, base_confidence)
    
    async def _fetch_content(self, url: str) -> Optional[str]:
        """Fetch web page content"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.text()
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
        
        return None
```

### Step 2.2: Vector Storage with Qdrant

**File:** `core/vector_store.py` (new)

```python
"""
Intent Engine - Vector Storage for Intent Embeddings

Uses Qdrant for storing and searching intent vectors.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from core.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class IntentVector:
    """Intent vector with metadata"""
    id: str
    url: str
    embedding: list[float]
    intent_tags: dict[str, Any]
    score: float


class QdrantVectorStore:
    """
    Vector storage and search using Qdrant.
    
    Usage:
        store = QdrantVectorStore()
        await store.store_intent_vector(intent_vector)
        results = await store.search_similar(query_embedding, limit=10)
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.collection_name = self.config.get("collection", "intent_index")
        self.qdrant_url = self.config.get("url", "http://localhost:6333")
        
        self._client = None
        self._embedding_service = None
    
    def _get_client(self):
        """Lazy initialization of Qdrant client"""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                
                self._client = QdrantClient(url=self.qdrant_url)
                
                # Create collection if not exists
                self._ensure_collection()
                
            except ImportError:
                logger.warning("Qdrant client not installed. Install with: pip install qdrant-client")
                return None
        
        return self._client
    
    def _get_embedding_service(self):
        """Get embedding service"""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service
    
    def _ensure_collection(self):
        """Create Qdrant collection if not exists"""
        from qdrant_client.http.models import (
            Distance,
            VectorParams,
        )
        
        client = self._get_client()
        if not client:
            return
        
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            # Create collection with 384-dim vectors (sentence-transformers)
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    def store_intent_vector(self, intent_vector: IntentVector) -> bool:
        """Store an intent vector"""
        from qdrant_client.http.models import PointStruct
        
        client = self._get_client()
        if not client:
            return False
        
        point = PointStruct(
            id=intent_vector.id,
            vector=intent_vector.embedding,
            payload={
                "url": intent_vector.url,
                "intent_tags": intent_vector.intent_tags,
                "score": intent_vector.score,
            }
        )
        
        client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        logger.debug(f"Stored intent vector for {intent_vector.url}")
        return True
    
    def search_similar(
        self, 
        query_embedding: list[float], 
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[IntentVector]:
        """Search for similar intent vectors"""
        client = self._get_client()
        if not client:
            return []
        
        # Build filter
        search_filter = None
        if filters:
            from qdrant_client.http.models import FieldCondition, MatchValue, Filter
            
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=f"intent_tags.{key}",
                        match=MatchValue(value=value)
                    )
                )
            
            if conditions:
                search_filter = Filter(must=conditions)
        
        # Search
        results = client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter
        )
        
        # Convert to IntentVector
        return [
            IntentVector(
                id=str(r.id),
                url=r.payload.get("url", ""),
                embedding=[],  # Don't return embedding
                intent_tags=r.payload.get("intent_tags", {}),
                score=r.score
            )
            for r in results
        ]
    
    def search_by_intent(
        self, 
        goal: Optional[str] = None,
        use_case: Optional[str] = None,
        limit: int = 10
    ) -> list[IntentVector]:
        """Search by intent tags"""
        filters = {}
        
        if goal:
            filters["primary_goal"] = goal
        if use_case:
            filters["use_cases"] = use_case
        
        # For tag-based search, we need to fetch and filter
        # This is a simplified implementation
        client = self._get_client()
        if not client:
            return []
        
        # Scroll through collection with filter
        results, _ = client.scroll(
            collection_name=self.collection_name,
            scroll_filter=None,  # TODO: implement proper filtering
            limit=limit
        )
        
        return [
            IntentVector(
                id=str(r.id),
                url=r.payload.get("url", ""),
                embedding=[],
                intent_tags=r.payload.get("intent_tags", {}),
                score=r.payload.get("score", 0.0)
            )
            for r in results
        ]


# Singleton
_vector_store: Optional[QdrantVectorStore] = None


def get_vector_store(config: Optional[dict[str, Any]] = None) -> QdrantVectorStore:
    """Get or create vector store singleton"""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore(config)
    return _vector_store
```

---

## Phase 3: Distributed Processing

### Step 3.1: Kafka Event Streaming Setup

**File:** `analytics/kafka_events.py` (new)

```python
"""
Intent Engine - Kafka Event Streaming

Publishes and subscribes to intent/search events.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class KafkaEventPublisher:
    """
    Publishes events to Kafka topics.
    
    Topics:
    - intents.raw: Raw intent extractions
    - intents.processed: Processed intents with enrichment
    - searches.executed: Search queries
    - results.served: Results returned
    - clicks.recorded: User clicks
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.bootstrap_servers = self.config.get(
            "bootstrap_servers", 
            "localhost:9092"
        )
        self._producer = None
    
    def _get_producer(self):
        """Lazy Kafka producer initialization"""
        if self._producer is None:
            try:
                from kafka import KafkaProducer
                
                self._producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    compression_type='gzip'
                )
                
                logger.info("Kafka producer initialized")
                
            except ImportError:
                logger.warning(
                    "kafka-python not installed. "
                    "Install with: pip install kafka-python"
                )
                return None
        
        return self._producer
    
    def publish_intent_extracted(self, intent_data: dict[str, Any]):
        """Publish intent extraction event"""
        producer = self._get_producer()
        if not producer:
            return
        
        future = producer.send(
            'intents.extracted',
            value=intent_data
        )
        
        # Don't wait for confirmation (async)
        logger.debug(f"Published intent event: {intent_data.get('intentId')}")
    
    def publish_search_executed(self, search_data: dict[str, Any]):
        """Publish search execution event"""
        producer = self._get_producer()
        if not producer:
            return
        
        producer.send(
            'searches.executed',
            value=search_data
        )
    
    def publish_click_recorded(self, click_data: dict[str, Any]):
        """Publish click event"""
        producer = self._get_producer()
        if not producer:
            return
        
        producer.send(
            'clicks.recorded',
            value=click_data
        )
    
    def flush(self):
        """Flush pending messages"""
        if self._producer:
            self._producer.flush()


class KafkaEventSubscriber:
    """
    Subscribes to Kafka topics for real-time processing.
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.bootstrap_servers = self.config.get(
            "bootstrap_servers", 
            "localhost:9092"
        )
        self._consumer = None
    
    def subscribe(self, topics: list[str], callback):
        """Subscribe to topics and process messages"""
        try:
            from kafka import KafkaConsumer
            
            consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='intent-engine-workers'
            )
            
            logger.info(f"Subscribed to topics: {topics}")
            
            for message in consumer:
                try:
                    callback(message.topic, message.value)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        except ImportError:
            logger.warning("kafka-python not installed")


# Singleton
_event_publisher: Optional[KafkaEventPublisher] = None


def get_event_publisher(config: Optional[dict[str, Any]] = None) -> KafkaEventPublisher:
    """Get event publisher singleton"""
    global _event_publisher
    if _event_publisher is None:
        _event_publisher = KafkaEventPublisher(config)
    return _event_publisher
```

---

## Phase 4: Observability

### Step 4.1: Distributed Tracing with Jaeger

**File:** `config/tracing.py` (new)

```python
"""
Intent Engine - Distributed Tracing Configuration

Uses OpenTelemetry + Jaeger for tracing.
"""

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_tracing(service_name: str = "intent-engine") -> Optional[trace.TracerProvider]:
    """
    Setup distributed tracing with Jaeger.
    
    Usage:
        setup_tracing()
        tracer = trace.get_tracer(__name__)
        
        with tracer.start_as_current_span("intent_extraction"):
            # Your code here
    """
    jaeger_endpoint = os.getenv(
        "JAEGER_ENDPOINT", 
        "http://localhost:14268/api/traces"
    )
    
    # Create resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0"
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Create Jaeger exporter
    exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    # Add span processor
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    return provider


# Decorator for tracing
def traced(span_name: str):
    """Decorator to trace a function"""
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name):
                return func(*args, **kwargs)
        
        # Detect if function is async
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_query_router.py

import pytest
from searxng.query_router import UnifiedQueryRouter, SearchBackend
from core.schema import UniversalIntent, DeclaredIntent, InferredIntent, IntentGoal

@pytest.fixture
def router():
    return UnifiedQueryRouter()

def test_route_troubleshooting_intent(router):
    """Troubleshooting queries should prefer SearXNG"""
    intent = UniversalIntent(
        intentId="test-123",
        context={},
        declared=DeclaredIntent(goal=IntentGoal.TROUBLESHOOTING),
        inferred=InferredIntent()
    )
    
    route = router.route(intent)
    
    assert SearchBackend.SEARXNG in route.backends
    assert route.weights[SearchBackend.SEARXNG] == 1.0
    assert route.parallel is False

def test_route_comparison_intent(router):
    """Comparison queries should use both backends"""
    intent = UniversalIntent(
        intentId="test-123",
        context={},
        declared=DeclaredIntent(goal=IntentGoal.COMPARISON),
        inferred=InferredIntent()
    )
    
    route = router.route(intent)
    
    assert SearchBackend.GO_CRAWLER in route.backends
    assert SearchBackend.SEARXNG in route.backends
    assert route.parallel is True
```

### Integration Tests

```python
# tests/integration/test_unified_search.py

import pytest
import asyncio
from searxng.unified_search import UnifiedSearchService
from models import UnifiedSearchRequest

@pytest.mark.asyncio
async def test_unified_search_end_to_end():
    """Test complete search flow"""
    service = UnifiedSearchService()
    
    request = UnifiedSearchRequest(
        query="best privacy-focused email providers",
        maxResults=10
    )
    
    response = await service.search(request)
    
    assert response.query == request.query
    assert len(response.results) > 0
    assert response.intent is not None
    assert "processingTimeMs" in response.metrics
```

---

## Deployment Guide

### Docker Compose (Development)

**File:** `docker-compose.dev.yml` (new)

```yaml
version: "3.8"

services:
  # Existing services...
  
  # NEW: Qdrant vector DB
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - intent-network
  
  # NEW: Kafka (using Redpanda for simplicity)
  redpanda:
    image: docker.redpanda.com/redpandadata/redpanda:latest
    command:
      - redpanda start
      - --smp 1
      - --overprovisioned
      - --node-id 0
      - --kafka-addr PLAINTEXT://0.0.0.0:29092
      - --advertise-kafka-addr PLAINTEXT://localhost:29092
      - --pandaproxy-addr 0.0.0.0:28082
      - --advertise-pandaproxy-addr localhost:28082
    ports:
      - "29092:29092"
      - "28082:28082"
    volumes:
      - redpanda_data:/var/lib/redpanda/data
    networks:
      - intent-network
  
  # NEW: Jaeger tracing
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "14268:14268"  # Thrift HTTP
      - "6831:6831"    # Thrift compact
    networks:
      - intent-network

volumes:
  qdrant_data:
  redpanda_data:
```

### Kubernetes (Production)

**File:** `k8s/intent-engine.yaml` (new)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: intent-engine-api
  labels:
    app: intent-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: intent-engine-api
  template:
    metadata:
      labels:
        app: intent-engine-api
    spec:
      containers:
      - name: api
        image: anony45/intent-engine-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: intent-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        - name: QDRANT_URL
          value: "http://qdrant-cluster:6333"
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka-cluster:9092"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: intent-engine-api
spec:
  selector:
    app: intent-engine-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

---

## Conclusion

This implementation guide provides step-by-step instructions for building the unified Intent Engine architecture. Start with Phase 1 (Query Router) and progress through each phase, testing thoroughly before moving to the next.

### Next Steps

1. **Review architecture** with team
2. **Setup development environment**
3. **Implement Phase 1** (Query Router)
4. **Test and iterate**
5. **Deploy to staging**
6. **Monitor and optimize**

For questions or issues, refer to the main documentation or open a GitHub issue.

---

*Document Version: 1.0.0 | Last Updated: March 15, 2026*
