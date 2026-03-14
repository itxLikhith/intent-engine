# Intent Engine Performance Optimization Plan

**Based on research of industry best practices (2025-2026)**

---

## Executive Summary

Research reveals **three major optimization opportunities** that can reduce latency by **60-80%** and increase throughput by **3-5x**:

| Optimization | Expected Impact | Implementation Effort |
|--------------|-----------------|----------------------|
| **Redis Caching** | 40-60% latency reduction | Low (1-2 days) |
| **SearXNG Tuning** | 30-50% latency reduction | Low (2-4 hours) |
| **Model Optimization** | 5-10x faster inference | Medium (1 week) |

---

## 1. Redis Caching Layer (HIGH PRIORITY)

### Research Findings

According to Redis.io and industry benchmarks:
- **Cache hit rate**: 70-80% for repeated queries
- **Latency reduction**: 3x faster responses (from ~2s to ~600ms)
- **Throughput increase**: 5x more concurrent requests

### Implementation

#### Step 1: Create Cache Module

```python
# config/redis_cache.py
"""
Enhanced Redis caching with async support and intelligent key generation
"""
import asyncio
import hashlib
import json
import logging
from typing import Any, Optional
from datetime import timedelta

import redis.asyncio as redis
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)


class RedisCache:
    """Async Redis cache with intelligent key generation and serialization"""
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._initialized = False
        self._default_ttl = 300  # 5 minutes for search results
        self._intent_ttl = 3600  # 1 hour for intent extraction
        self._key_prefix = "intent-engine"
    
    async def initialize(self, redis_url: str = "redis://redis:6379/0"):
        """Initialize Redis connection on startup"""
        if not self._initialized:
            try:
                self._client = redis.Redis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5.0,
                    socket_timeout=5.0,
                    retry_on_timeout=True,
                )
                await self._client.ping()
                self._initialized = True
                logger.info(f"Redis cache connected: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed, caching disabled: {e}")
                self._initialized = False
    
    async def close(self):
        """Close Redis connection on shutdown"""
        if self._client:
            await self._client.close()
            logger.info("Redis cache connection closed")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key from arguments"""
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{self._key_prefix}:{prefix}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._initialized:
            return None
        
        try:
            value = await self._client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        background: bool = False
    ):
        """Set value in cache with TTL"""
        if not self._initialized:
            return
        
        try:
            serialized = json.dumps(value, default=str)
            effective_ttl = ttl or self._default_ttl
            
            if background:
                # Non-blocking cache update
                asyncio.create_task(self._set_async(key, serialized, effective_ttl))
            else:
                await self._set_async(key, serialized, effective_ttl)
                
            logger.debug(f"Cache SET: {key} (TTL: {effective_ttl}s)")
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
    
    async def _set_async(self, key: str, value: str, ttl: int):
        """Internal async set operation"""
        await self._client.setex(key, ttl, value)
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self._initialized:
            return
        
        try:
            await self._client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
    
    async def get_or_set(
        self,
        key: str,
        factory,
        ttl: Optional[int] = None,
        background_refresh: bool = False
    ) -> Any:
        """
        Get from cache or compute and cache result
        
        Args:
            key: Cache key
            factory: Async function to compute value if cache miss
            ttl: Time to live in seconds
            background_refresh: If True, refresh cache in background after returning
        
        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        # Cache miss: compute value
        logger.info(f"Cache miss, computing: {key}")
        value = await factory()
        
        # Cache result
        await self.set(key, value, ttl)
        
        return value
    
    async def warm_up(self, queries: list[str]):
        """Pre-warm cache with common queries"""
        logger.info(f"Warming up cache with {len(queries)} queries")
        # Implementation depends on search service integration


# Global cache instance
cache = RedisCache()


async def get_cache() -> RedisCache:
    """Get cache instance (singleton pattern)"""
    return cache
```

#### Step 2: Add Cache Decorator

```python
# config/cache_decorator.py
"""
Reusable cache decorators for FastAPI endpoints
"""
import functools
import hashlib
import json
import logging
from typing import Callable

from fastapi import Request, Response

from .redis_cache import get_cache

logger = logging.getLogger(__name__)


def cache_response(ttl: int = 300, key_prefix: str = "api"):
    """
    Decorator for caching FastAPI endpoint responses
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    
    Usage:
        @app.get("/search")
        @cache_response(ttl=300, key_prefix="search")
        async def search(query: str):
            return await expensive_search(query)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request context if available
            request = kwargs.get('request')
            
            # Generate cache key from function name + arguments
            cache_key_data = {
                'function': func.__name__,
                'args': args[1:] if args else [],  # Skip self
                'kwargs': {k: v for k, v in kwargs.items() 
                          if k not in ['request', 'background_tasks']}
            }
            key_hash = hashlib.md5(
                json.dumps(cache_key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            cache_key = f"{key_prefix}:{key_hash}"
            
            # Try cache
            cache = await get_cache()
            cached = await cache.get(cache_key)
            
            if cached is not None:
                logger.info(f"Cache HIT: {func.__name__} ({cache_key[:50]}...)")
                return cached
            
            # Cache miss: execute function
            logger.info(f"Cache MISS: {func.__name__} ({cache_key[:50]}...)")
            result = await func(*args, **kwargs)
            
            # Cache result (non-blocking)
            await cache.set(cache_key, result, ttl, background=True)
            
            return result
        return wrapper
    return decorator


def cache_search_results(ttl: int = 300):
    """Specialized decorator for search endpoint caching"""
    return cache_response(ttl=ttl, key_prefix="search")


def cache_intent_extraction(ttl: int = 3600):
    """Specialized decorator for intent extraction caching"""
    return cache_response(ttl=ttl, key_prefix="intent")
```

#### Step 3: Integrate with Unified Search

```python
# searxng/unified_search.py (modified)
from config.redis_cache import get_cache, cache_search_results

class UnifiedSearchService:
    async def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        cache = await get_cache()
        
        # Generate cache key
        cache_key = f"search:{request.query}:{request.pageno}:{request.language}"
        
        # Try cache first
        cached_result = await cache.get(cache_key)
        if cached_result:
            logger.info(f"Search cache HIT: {request.query[:50]}...")
            return UnifiedSearchResponse(**cached_result)
        
        # Cache miss: perform search
        logger.info(f"Search cache MISS: {request.query[:50]}...")
        
        # ... existing search logic ...
        response = await self._perform_search(request)
        
        # Cache result (non-blocking, in background)
        await cache.set(cache_key, response.dict(), ttl=300, background=True)
        
        return response
```

#### Step 4: Update Docker Compose

```yaml
# docker-compose.yml
services:
  intent-engine-api:
    environment:
      # Enable Redis caching
      - REDIS_ENABLED=true
      - REDIS_URL=redis://redis:6379/0
      - REDIS_MAX_CONNECTIONS=50
      - REDIS_TIMEOUT=5.0
    
    # Increase workers for better concurrency
    - WORKERS=8
    
    # Increase database pool
    - DATABASE_POOL_SIZE=20
    - DATABASE_MAX_OVERFLOW=40
    
    depends_on:
      redis:
        condition: service_healthy
  
  redis:
    image: docker.io/valkey/valkey:8-alpine
    container_name: intent-redis
    command: valkey-server --save 30 1 --loglevel warning --maxmemory 512mb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    networks:
      - intent-network
      - searxng
    volumes:
      - valkey-data:/data
    healthcheck:
      test: ["CMD", "valkey-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 512M
```

### Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Cache Hit Rate** | 0% | 70-80% | - |
| **Mean Latency** | 2,000ms | 600-800ms | **60-70% ↓** |
| **Throughput** | 20 req/s | 100 req/s | **5x ↑** |
| **Max Concurrent** | 20 users | 100 users | **5x ↑** |

---

## 2. SearXNG Performance Tuning (HIGH PRIORITY)

### Research Findings

From SearXNG configuration analysis:
- Default timeout: 3.0s per engine
- Many engines enabled by default (slow)
- No caching enabled by default
- Connection pool: 100 connections, 20 max size

### Implementation

#### Step 1: Optimize SearXNG Settings

```yaml
# searxng/settings.yml (updated)
use_default_settings: true

search:
  # Reduce number of simultaneous engines
  max_page: 5  # Limit results per page
  
  # Faster timeout for most engines
  default_lang: en
  
  # Enable result caching with Valkey
  formats:
    - html
    - json

# Critical: Enable Valkey for caching
valkey:
  url: "redis://redis:6379/0"
  # Cache search results for 5 minutes
  # Cache URLs for 1 hour

outgoing:
  # Connection pooling
  pool_connections: 50  # Reduced from 100 (sufficient for our load)
  pool_maxsize: 10      # Reduced from 20
  enable_http2: true    # HTTP/2 for multiplexing
  
  # Faster timeouts
  request_timeout: 2.0  # Reduced from 3.0s
  max_request_timeout: 5.0  # Reduced from 10.0s
  
  # Retry configuration
  max_retries: 1
  retry_on_timeout: true

# Engine-specific optimizations
engines:
  # Disable slow/unreliable engines
  - name: google
    engine: google
    shortcut: g
    disabled: false
    timeout: 3.0  # Specific timeout
  
  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    disabled: false
    timeout: 2.0
  
  - name: brave
    engine: brave
    shortcut: br
    disabled: false
    timeout: 2.0
  
  # Disable slow engines
  - name: wikidata
    engine: wikidata
    shortcut: wd
    disabled: true  # Slow, not essential
  
  - name: bitcoin
    engine: bitcoin
    disabled: true  # Niche, slow
  
  - name: tor
    engine: tor
    disabled: true  # Very slow
  
  # Keep only fast, reliable engines
  - name: startpage
    engine: startpage
    shortcut: sp
    disabled: false
    timeout: 4.0  # Slower but privacy-focused
  
  - name: qwant
    engine: qwant
    shortcut: qw
    disabled: false
    timeout: 3.0

# Server performance settings
server:
  port: 8080
  bind_address: "0.0.0.0"
  limiter: false  # Disable for internal use
  public_instance: false
  http_protocol_version: "1.1"  # Better than 1.0
  method: "POST"  # Better for production
  
# Enable metrics for monitoring
general:
  enable_metrics: true
  open_metrics: "metrics_password_change_in_prod"

# Disable unnecessary plugins
plugins:
  - searx.plugins.infinite_scroll.SXNGPlugin:
      active: false  # Impacts performance
  - searx.plugins.tracker_url_remover.SXNGPlugin:
      active: true  # Keep privacy plugin
```

#### Step 2: Reduce Active Search Engines

```python
# searxng/unified_search.py (modified)
class UnifiedSearchService:
    def __init__(self):
        self.searxng_client = get_searxng_client()
        
        # Only use fastest engines
        self.fast_engines = ["google", "duckduckgo", "brave", "startpage"]
        logger.info(f"Initialized with {len(self.fast_engines)} engines: {self.fast_engines}")
    
    async def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        # Limit to fastest engines for better latency
        request.engines = self.fast_engines
        
        # ... rest of search logic ...
```

### Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Active Engines** | 40+ | 4-6 | **85% ↓** |
| **Timeout per Engine** | 3.0s | 2.0s | **33% ↓** |
| **Search Latency** | 3,000ms | 1,500ms | **50% ↓** |
| **Cache Hit Rate** | 0% | 60% | - |

---

## 3. Model Optimization (MEDIUM PRIORITY)

### Research Findings

From Hugging Face and sentence-transformers research:
- **Model2Vec**: 500x faster on CPU, 50x smaller
- **Knowledge Distillation**: Train smaller student model
- **Quantization**: INT8 quantization reduces size 4x
- **ONNX Runtime**: 2-3x faster inference

### Implementation Options

#### Option A: Switch to Model2Vec (Recommended)

```python
# core/embedding_service.py (optimized version)
"""
Optimized embedding service using Model2Vec for 500x faster CPU inference
"""
import logging
from functools import lru_cache
from typing import Optional

import numpy as np

try:
    from model2vec import StaticModel
    MODEL2VEC_AVAILABLE = True
except ImportError:
    MODEL2VEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizedEmbeddingService:
    """High-performance embedding service using Model2Vec"""
    
    def __init__(self, model_name: str = "minishlab/potion-base-8M"):
        self.model_name = model_name
        self.model = None
        self._initialized = False
    
    def initialize(self):
        """Load optimized Model2Vec model"""
        if not self._initialized:
            if MODEL2VEC_AVAILABLE:
                logger.info(f"Loading Model2Vec: {self.model_name}")
                # Model2Vec loads 500x faster than sentence-transformers
                self.model = StaticModel.from_pretrained(self.model_name)
                logger.info("Model2Vec loaded successfully (500x faster than SBERT)")
            else:
                # Fallback to sentence-transformers
                logger.warning("Model2Vec not available, falling back to sentence-transformers")
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self._initialized = True
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding (500x faster on CPU)"""
        if not self._initialized:
            self.initialize()
        
        try:
            if MODEL2VEC_AVAILABLE:
                # Model2Vec: ~0.5ms per query vs ~250ms for SBERT
                embeddings = self.model.encode([text])
                return embeddings[0]
            else:
                # Fallback to sentence-transformers
                return self.model.encode(text)
        except Exception as e:
            logger.error(f"Embedding encoding failed: {e}")
            return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


# Global instance
_embedding_service = None


def get_embedding_service():
    """Get optimized embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = OptimizedEmbeddingService()
    return _embedding_service
```

#### Option B: Quantization (Alternative)

```python
# requirements.txt addition
optimum[openvino,nncf]>=1.16.0
onnxruntime>=1.16.0
```

```python
# core/embedding_service_quantized.py
from optimum.intel import IPEXSentenceTransformer

class QuantizedEmbeddingService:
    """INT8 quantized embeddings for 2-3x faster inference"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Load with INT8 quantization
        self.model = IPEXSentenceTransformer.from_pretrained(
            model_name,
            export=True,
            load_in_8bit=True  # INT8 quantization
        )
    
    def encode_text(self, text: str):
        """2-3x faster than standard SBERT"""
        return self.model.encode(text)
```

### Expected Results

| Metric | Before | After (Model2Vec) | Improvement |
|--------|--------|-------------------|-------------|
| **Model Size** | 90MB | 2MB | **98% ↓** |
| **Load Time** | 5-10s | <1s | **90% ↓** |
| **Inference Time** | 250ms | 0.5ms | **500x ↑** |
| **Memory Usage** | 500MB | 50MB | **90% ↓** |

---

## 4. Async I/O Optimization (MEDIUM PRIORITY)

### Research Findings

From Nordic APIs and async best practices:
- **Parallel API calls**: 3x faster than sequential
- **Speculative execution**: Start likely calls before needed
- **Timeout budgets**: Prevent cascading failures
- **Decouple reasoning from execution**: Plan first, execute in parallel

### Implementation

```python
# searxng/unified_search.py (optimized)
import asyncio
from asyncio import Semaphore

class OptimizedUnifiedSearchService:
    """Optimized search with parallel execution and timeouts"""
    
    def __init__(self):
        self.searxng_client = get_searxng_client()
        self._search_semaphore = Semaphore(10)  # Max 10 concurrent searches
    
    async def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        """Optimized search with parallel intent extraction and search"""
        
        # Use semaphore to limit concurrent searches
        async with self._search_semaphore:
            # Step 1: Run intent extraction and SearXNG search IN PARALLEL
            # These are independent operations
            intent_task = asyncio.create_task(
                self._extract_intent_safe(request.query)
            )
            search_task = asyncio.create_task(
                self._search_searxng_with_timeout(request)
            )
            
            # Wait for both with individual timeouts
            try:
                intent_result, search_results = await asyncio.gather(
                    intent_task,
                    search_task,
                    return_exceptions=True
                )
            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise
            
            # Step 2: Rank results (if intent extracted)
            if isinstance(intent_result, Exception):
                logger.warning(f"Intent extraction failed: {intent_result}")
                universal_intent = None
            else:
                universal_intent = intent_result.intent if intent_result else None
            
            if isinstance(search_results, Exception):
                logger.error(f"SearXNG search failed: {search_results}")
                raise search_results
            
            # Step 3: Rank if intent available
            if universal_intent and request.rank_results:
                ranked_results = await self._rank_results_parallel(
                    search_results,
                    universal_intent
                )
            else:
                ranked_results = search_results
            
            return self._build_response(ranked_results, universal_intent)
    
    async def _search_searxng_with_timeout(
        self,
        request: UnifiedSearchRequest,
        timeout: float = 5.0
    ) -> list:
        """Search SearXNG with strict timeout"""
        try:
            # Individual timeout for SearXNG (not total)
            async with asyncio.timeout(timeout):
                return await self.searxng_client.search(request)
        except asyncio.TimeoutError:
            logger.warning(f"SearXNG timeout after {timeout}s")
            return []  # Return empty instead of failing
    
    async def _extract_intent_safe(self, query: str) -> Optional[Any]:
        """Extract intent with error handling"""
        try:
            # Run in thread pool (CPU-bound)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: extract_intent(IntentExtractionRequest(
                    product='search',
                    input={'text': query},
                    context={'sessionId': 'search'}
                ))
            )
        except Exception as e:
            logger.error(f"Intent extraction error: {e}")
            return None
    
    async def _rank_results_parallel(
        self,
        results: list,
        intent: Any,
        max_concurrent: int = 20
    ) -> list:
        """Rank results in parallel batches"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def rank_with_semaphore(result):
            async with semaphore:
                return await self._rank_single_result(result, intent)
        
        tasks = [rank_with_semaphore(r) for r in results]
        ranked = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures and sort by score
        valid_results = [
            r for r in ranked
            if isinstance(r, dict) and r.get('ranked_score', 0) > 0
        ]
        
        return sorted(
            valid_results,
            key=lambda x: x.get('ranked_score', 0),
            reverse=True
        )
```

### Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Parallel Operations** | Sequential | Parallel | **2-3x ↑** |
| **Timeout Handling** | Total timeout | Per-operation | **Better reliability** |
| **Error Recovery** | Fail fast | Graceful degradation | **Higher success rate** |
| **Concurrent Searches** | Limited | 10 concurrent | **10x ↑** |

---

## 5. Complete Optimization Summary

### Combined Impact

| Optimization | Latency Impact | Throughput Impact | Implementation |
|--------------|---------------|-------------------|----------------|
| **Redis Caching** | -60% | +5x | 1-2 days |
| **SearXNG Tuning** | -50% | +2x | 2-4 hours |
| **Model2Vec** | -99% (inference) | +10x | 3-5 days |
| **Async I/O** | -30% | +3x | 2-3 days |
| **Combined** | **-80-90%** | **+10-15x** | **1-2 weeks** |

### Projected Performance After All Optimizations

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| **Mean Latency** | 2,000ms | 200-400ms | **80-90% ↓** |
| **P95 Latency** | 4,800ms | 600-800ms | **85% ↓** |
| **Throughput** | 20 req/s | 200-300 req/s | **10-15x ↑** |
| **Max Concurrent** | 20 users | 200+ users | **10x ↑** |
| **Daily Capacity** | 1.7M | 17-25M | **10-15x ↑** |
| **Monthly Capacity** | 50M | 500-750M | **10-15x ↑** |

### Implementation Priority

```
Week 1:
├── Day 1-2: Redis caching layer (HIGH IMPACT)
├── Day 3: SearXNG optimization (QUICK WIN)
└── Day 4-5: Async I/O improvements

Week 2:
├── Day 1-3: Model2Vec integration
├── Day 4: Testing and benchmarking
└── Day 5: Documentation and deployment
```

### Cost-Benefit Analysis

| Optimization | Dev Time | Infrastructure Cost | Performance Gain | ROI |
|--------------|----------|---------------------|------------------|-----|
| Redis Caching | 2 days | +512MB RAM | 5x throughput | ⭐⭐⭐⭐⭐ |
| SearXNG Tuning | 4 hours | None | 2x throughput | ⭐⭐⭐⭐⭐ |
| Async I/O | 3 days | None | 3x throughput | ⭐⭐⭐⭐ |
| Model2Vec | 5 days | None | 10x inference | ⭐⭐⭐⭐ |

---

## 6. Next Steps

1. **Immediate (Today)**: Update SearXNG settings.yml
2. **Short-term (This Week)**: Implement Redis caching
3. **Medium-term (Next Week)**: Add async I/O optimizations
4. **Long-term (Next Sprint)**: Migrate to Model2Vec

### Monitoring After Optimization

```python
# Add to main_api.py
@app.get("/performance/stats")
async def performance_stats():
    """Real-time performance statistics"""
    return {
        "cache_hit_rate": await cache.get_hit_rate(),
        "avg_latency_ms": get_avg_latency(),
        "requests_per_second": get_rps(),
        "active_connections": get_active_connections(),
        "redis_memory_mb": await cache.get_memory_usage(),
    }
```

---

**References:**
- Redis.io FastAPI Tutorial (2026)
- SearXNG Configuration Documentation
- Nordic APIs: 9 Tips for Reducing API Latency (2026)
- Hugging Face: Model2Vec Performance Guide (2025)
- sentence-transformers GitHub Releases
