"""
Intent Engine - Redis Cache Module

High-performance async Redis caching for search results and intent extraction.

Features:
- Async Redis client using redis.asyncio
- Intelligent cache key generation
- Configurable TTL for different data types
- Cache-aside pattern with background refresh
- Graceful degradation when Redis unavailable

Usage:
    from config.redis_cache import cache
    
    # Initialize on startup
    await cache.initialize(redis_url="redis://redis:6379/0")
    
    # Get/set cache
    cached = await cache.get("key")
    await cache.set("key", value, ttl=300)
    
    # Cache-aside pattern
    result = await cache.get_or_set(
        "search:query123",
        lambda: expensive_search("query123"),
        ttl=300
    )
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Callable, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Async Redis cache with intelligent key generation and serialization.
    
    Provides:
    - Automatic connection management
    - JSON serialization with datetime support
    - Configurable TTL per operation
    - Background cache updates
    - Cache statistics tracking
    """
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._initialized = False
        self._enabled = False
        
        # Default TTL settings (in seconds)
        self._default_ttl = 300  # 5 minutes for general cache
        self._search_ttl = 300   # 5 minutes for search results
        self._intent_ttl = 3600  # 1 hour for intent extraction
        self._ranking_ttl = 600  # 10 minutes for ranking results
        
        # Key prefix for namespacing
        self._key_prefix = "intent-engine"
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "sets": 0,
        }
    
    async def initialize(self, redis_url: str = "redis://redis:6379/0"):
        """
        Initialize Redis connection on application startup.
        
        Args:
            redis_url: Redis connection URL (redis://host:port/db)
        """
        if self._initialized:
            return
        
        try:
            logger.info(f"Connecting to Redis: {redis_url}")
            
            self._client = redis.Redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5.0,
                socket_timeout=5.0,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            
            # Test connection
            await self._client.ping()
            
            self._initialized = True
            self._enabled = True
            logger.info("Redis cache connected and enabled")
            
        except Exception as e:
            logger.warning(f"Redis connection failed, caching disabled: {e}")
            self._initialized = True
            self._enabled = False
    
    async def close(self):
        """Close Redis connection on application shutdown"""
        if self._client:
            await self._client.close()
            logger.info("Redis cache connection closed")
            self._enabled = False
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate consistent cache key from arguments.
        
        Args:
            prefix: Key prefix (e.g., "search", "intent")
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key
        
        Returns:
            MD5-hashed cache key with prefix
        """
        # Create deterministic key string
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else [],
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{self._key_prefix}:{prefix}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/error
        """
        if not self._enabled or not self._initialized:
            return None
        
        try:
            start_time = time.perf_counter()
            value = await self._client.get(key)
            elapsed = (time.perf_counter() - start_time) * 1000
            
            if value:
                self._stats["hits"] += 1
                logger.debug(f"Cache HIT: {key[:80]}... ({elapsed:.2f}ms)")
                return json.loads(value)
            else:
                self._stats["misses"] += 1
                logger.debug(f"Cache MISS: {key[:80]}... ({elapsed:.2f}ms)")
                return None
                
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis GET error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        background: bool = False
    ):
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (default: 300)
            background: If True, cache update happens asynchronously
        """
        if not self._enabled or not self._initialized:
            return
        
        try:
            serialized = json.dumps(value, default=str)
            effective_ttl = ttl or self._default_ttl
            
            if background:
                # Non-blocking cache update
                asyncio.create_task(self._set_async(key, serialized, effective_ttl))
            else:
                await self._set_async(key, serialized, effective_ttl)
            
            self._stats["sets"] += 1
            logger.debug(f"Cache SET: {key[:80]}... (TTL: {effective_ttl}s)")
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis SET error: {e}")
    
    async def _set_async(self, key: str, value: str, ttl: int):
        """Internal async set operation"""
        await self._client.setex(key, ttl, value)
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self._enabled or not self._initialized:
            return
        
        try:
            await self._client.delete(key)
            logger.debug(f"Cache DELETE: {key[:80]}...")
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis DELETE error: {e}")
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get from cache or compute and cache result (cache-aside pattern).
        
        Args:
            key: Cache key
            factory: Async function to compute value if cache miss
            ttl: Time to live in seconds
        
        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        # Cache miss: compute value
        logger.info(f"Cache miss, computing: {key[:80]}...")
        
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        # Cache result
        await self.set(key, value, ttl)
        
        return value
    
    async def get_search(self, query: str, pageno: int = 1, language: str = "en") -> Optional[dict]:
        """Get cached search results"""
        key = self._generate_key("search", query=query, pageno=pageno, language=language)
        return await self.get(key)
    
    async def set_search(self, query: str, pageno: int, language: str, results: dict):
        """Cache search results"""
        key = self._generate_key("search", query=query, pageno=pageno, language=language)
        await self.set(key, results, ttl=self._search_ttl, background=True)
    
    async def get_intent(self, query: str) -> Optional[dict]:
        """Get cached intent extraction"""
        key = self._generate_key("intent", query=query)
        return await self.get(key)
    
    async def set_intent(self, query: str, intent: dict):
        """Cache intent extraction result"""
        key = self._generate_key("intent", query=query)
        await self.set(key, intent, ttl=self._intent_ttl, background=True)
    
    async def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            **self._stats,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            "enabled": self._enabled,
            "initialized": self._initialized,
        }
    
    async def get_memory_usage(self) -> Optional[int]:
        """Get Redis memory usage in MB"""
        if not self._enabled:
            return None
        
        try:
            info = await self._client.info("memory")
            return round(info.get("used_memory", 0) / 1024 / 1024, 2)
        except Exception:
            return None
    
    async def flush(self, pattern: Optional[str] = None):
        """
        Flush cache keys matching pattern.
        
        Args:
            pattern: Key pattern to delete (default: all intent-engine keys)
        """
        if not self._enabled:
            return
        
        try:
            search_pattern = pattern or f"{self._key_prefix}:*"
            cursor = 0
            
            while True:
                cursor, keys = await self._client.scan(cursor, match=search_pattern, count=100)
                if keys:
                    await self._client.delete(*keys)
                    logger.info(f"Flushed {len(keys)} cache keys")
                
                if cursor == 0:
                    break
                    
        except Exception as e:
            logger.error(f"Cache flush error: {e}")


# Global cache instance
cache = RedisCache()


async def get_cache() -> RedisCache:
    """Get global cache instance"""
    return cache


async def initialize_cache(redis_url: str = "redis://redis:6379/0"):
    """Initialize global cache instance"""
    await cache.initialize(redis_url)


async def close_cache():
    """Close global cache instance"""
    await cache.close()
