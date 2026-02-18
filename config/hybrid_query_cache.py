"""
Intent Engine - Hybrid Query Result Cache (Redis + Local LRU)

This module provides a two-tier caching strategy for query results:
- L1: Local in-memory LRU cache for ultra-fast access
- L2: Redis distributed cache for multi-instance deployments

Features:
- TTL-based expiration
- Automatic fallback to local cache if Redis unavailable
- Cache coherence across instances via Redis
- Separate caches for different operations (intent, ranking, URLs, ads)
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from config.redis_cache import RedisCache, get_redis_cache

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    value: Any
    created_at: float
    ttl_seconds: float
    access_count: int = 0
    last_accessed: float = 0

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() - self.created_at > self.ttl_seconds

    def touch(self) -> None:
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()


class HybridQueryCache:
    """
    Two-tier query result cache with Redis + local LRU.

    Cache lookup order:
    1. Check local L1 cache (fastest)
    2. Check Redis L2 cache (distributed)
    3. Compute and store in both caches
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: float = 300,
        redis_ttl_seconds: Optional[float] = None,
        use_redis: bool = True,
        name: str = "default",
    ):
        """
        Initialize hybrid cache.

        Args:
            max_size: Local cache max size
            default_ttl_seconds: Local cache default TTL
            redis_ttl_seconds: Redis TTL (defaults to default_ttl_seconds)
            use_redis: Whether to use Redis
            name: Cache name for logging
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.redis_ttl_seconds = redis_ttl_seconds or default_ttl_seconds
        self.name = name
        self.use_redis = use_redis and os.getenv("REDIS_ENABLED", "false").lower() == "true"

        # L1: Local cache
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()

        # L2: Redis cache
        self.redis_cache: Optional[RedisCache] = None
        if self.use_redis:
            self.redis_cache = get_redis_cache()
            if self.redis_cache is None:
                logger.warning(f"Redis not available for cache '{name}', using local cache only")
                self.use_redis = False

        # Stats
        self.stats = {
            "hits_l1": 0,
            "hits_l2": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

        # Start cleanup thread
        self._start_cleanup_thread()

        logger.info(f"HybridQueryCache '{name}' initialized (Redis: {self.use_redis})")

    def _start_cleanup_thread(self) -> None:
        """Start background thread to clean expired entries"""

        def cleanup_worker():
            while True:
                time.sleep(60)
                self._cleanup_expired()

        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()

    def _cleanup_expired(self) -> int:
        """Remove expired entries from local cache"""
        with self.lock:
            expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
            for key in expired_keys:
                del self.cache[key]
            self.stats["expirations"] += len(expired_keys)
            return len(expired_keys)

    def _evict_lru(self) -> None:
        """Evict least recently used entry from local cache"""
        if not self.cache:
            return

        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
        del self.cache[lru_key]
        self.stats["evictions"] += 1

    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments"""
        key_data = {"args": args, "kwargs": {k: v for k, v in sorted(kwargs.items())}}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return f"cache:{self.name}:{hashlib.md5(key_str.encode()).hexdigest()}"

    def _redis_key(self, local_key: str) -> str:
        """Convert local key to Redis key"""
        return f"cache:{self.name}:{local_key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Check L1 first
        with self.lock:
            entry = self.cache.get(key)

            if entry is not None:
                if entry.is_expired():
                    del self.cache[key]
                    self.stats["expirations"] += 1
                    self.stats["misses"] += 1
                else:
                    entry.touch()
                    self.stats["hits_l1"] += 1
                    return entry.value

        # Check L2 (Redis)
        if self.use_redis and self.redis_cache:
            redis_key = self._redis_key(key)
            cached = self.redis_cache.get(redis_key)
            if cached is not None:
                # Promote to L1
                with self.lock:
                    while len(self.cache) >= self.max_size:
                        self._evict_lru()
                    self.cache[key] = CacheEntry(
                        value=cached,
                        created_at=time.time(),
                        ttl_seconds=self.default_ttl_seconds,
                        last_accessed=time.time(),
                    )
                self.stats["hits_l2"] += 1
                return cached

        self.stats["misses"] += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Store value in cache"""
        ttl_local = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        ttl_redis = self.redis_ttl_seconds

        # Store in L1
        with self.lock:
            while len(self.cache) >= self.max_size:
                self._evict_lru()

            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl_local,
                last_accessed=time.time(),
            )
            self.cache[key] = entry

        # Store in L2 (Redis)
        if self.use_redis and self.redis_cache:
            redis_key = self._redis_key(key)
            self.redis_cache.set(redis_key, value, ttl=ttl_redis)

    def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], Any],
        ttl_seconds: Optional[float] = None,
    ) -> Any:
        """Get from cache or compute and store"""
        cached = self.get(key)
        if cached is not None:
            return cached

        value = compute_func()
        self.set(key, value, ttl_seconds)
        return value

    def invalidate(self, key: str) -> bool:
        """Remove entry from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]

        if self.use_redis and self.redis_cache:
            redis_key = self._redis_key(key)
            self.redis_cache.delete(redis_key)

        return True

    def invalidate_pattern(self, pattern: str) -> int:
        """Remove all entries matching pattern"""
        count = 0

        with self.lock:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
            count = len(keys_to_remove)

        if self.use_redis and self.redis_cache:
            keys = self.redis_cache.keys(f"cache:{self.name}:*{pattern}*")
            if keys:
                self.redis_cache.delete_many(keys)
                count += len(keys)

        return count

    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()

        if self.use_redis and self.redis_cache:
            keys = self.redis_cache.keys(f"cache:{self.name}:*")
            if keys:
                self.redis_cache.delete_many(keys)

        self.stats = {
            "hits_l1": 0,
            "hits_l2": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

        logger.info(f"HybridQueryCache '{self.name}' cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats["hits_l1"] + self.stats["hits_l2"] + self.stats["misses"]
            hit_rate_l1 = self.stats["hits_l1"] / total_requests if total_requests > 0 else 0
            hit_rate_l2 = self.stats["hits_l2"] / total_requests if total_requests > 0 else 0

            stats = {
                "name": self.name,
                "size": len(self.cache),
                "max_size": self.max_size,
                **self.stats,
                "hit_rate_l1": round(hit_rate_l1, 4),
                "hit_rate_l2": round(hit_rate_l2, 4),
                "hit_rate_total": round(hit_rate_l1 + hit_rate_l2, 4),
                "use_redis": self.use_redis,
                "default_ttl_seconds": self.default_ttl_seconds,
            }

            if self.use_redis and self.redis_cache:
                stats["redis_stats"] = self.redis_cache.get_stats()

            return stats


# Import os at module level (needed for HybridQueryCache)
import os

# Global cache instances for different operations
_intent_extraction_cache: Optional[HybridQueryCache] = None
_ranking_cache: Optional[HybridQueryCache] = None
_url_analysis_cache: Optional[HybridQueryCache] = None
_ad_matching_cache: Optional[HybridQueryCache] = None
_cache_lock = threading.Lock()


def get_intent_extraction_cache() -> HybridQueryCache:
    """Get cache for intent extraction results (10 min TTL)"""
    global _intent_extraction_cache
    if _intent_extraction_cache is None:
        with _cache_lock:
            if _intent_extraction_cache is None:
                _intent_extraction_cache = HybridQueryCache(
                    max_size=5000,
                    default_ttl_seconds=600,
                    name="intent_extraction",
                )
    return _intent_extraction_cache


def get_ranking_cache() -> HybridQueryCache:
    """Get cache for ranking results (5 min TTL)"""
    global _ranking_cache
    if _ranking_cache is None:
        with _cache_lock:
            if _ranking_cache is None:
                _ranking_cache = HybridQueryCache(
                    max_size=2000,
                    default_ttl_seconds=300,
                    name="ranking",
                )
    return _ranking_cache


def get_url_analysis_cache() -> HybridQueryCache:
    """Get cache for URL analysis results (1 hour TTL)"""
    global _url_analysis_cache
    if _url_analysis_cache is None:
        with _cache_lock:
            if _url_analysis_cache is None:
                _url_analysis_cache = HybridQueryCache(
                    max_size=10000,
                    default_ttl_seconds=3600,
                    name="url_analysis",
                )
    return _url_analysis_cache


def get_ad_matching_cache() -> HybridQueryCache:
    """Get cache for ad matching results (2 min TTL)"""
    global _ad_matching_cache
    if _ad_matching_cache is None:
        with _cache_lock:
            if _ad_matching_cache is None:
                _ad_matching_cache = HybridQueryCache(
                    max_size=1000,
                    default_ttl_seconds=120,
                    name="ad_matching",
                )
    return _ad_matching_cache


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches"""
    return {
        "intent_extraction": get_intent_extraction_cache().get_stats(),
        "ranking": get_ranking_cache().get_stats(),
        "url_analysis": get_url_analysis_cache().get_stats(),
        "ad_matching": get_ad_matching_cache().get_stats(),
    }


def clear_all_caches() -> None:
    """Clear all caches"""
    get_intent_extraction_cache().clear()
    get_ranking_cache().clear()
    get_url_analysis_cache().clear()
    get_ad_matching_cache().clear()
    logger.info("All caches cleared")


def cached(
    cache_instance: HybridQueryCache,
    ttl_seconds: Optional[float] = None,
    key_func: Optional[Callable] = None,
):
    """Decorator to cache function results"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_instance._make_key(*args, **kwargs)

            # Try to get from cache
            cached_value = cache_instance.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Compute and store
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl_seconds)
            return result

        return wrapper

    return decorator
