"""
Query Cache for Ranking Results

This module provides caching for ranking results to improve performance.
"""

import hashlib
import threading
from typing import Any, Optional
from collections import OrderedDict


class RankingCache:
    """Thread-safe LRU cache for ranking results"""
    
    _instance: Optional["RankingCache"] = None
    _lock = threading.Lock()
    
    def __new__(cls, capacity: int = 1000):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._capacity = capacity
        return cls._instance
    
    def __init__(self, capacity: int = 1000):
        if self._initialized:
            return
        self._capacity = capacity
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._initialized = True
    
    def _make_key(self, query: str, **kwargs) -> str:
        """Create cache key from query and kwargs"""
        key_parts = [query]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Get cached result for query"""
        key = self._make_key(query, **kwargs)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, query: str, result: Any, **kwargs) -> None:
        """Cache result for query"""
        key = self._make_key(query, **kwargs)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._capacity:
                    self._cache.popitem(last=False)
            self._cache[key] = result
    
    def clear(self) -> None:
        """Clear the cache"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "capacity": self._capacity,
        }


# Singleton instance
_ranking_cache_instance: Optional[RankingCache] = None


def get_ranking_cache(capacity: int = 1000) -> RankingCache:
    """Get singleton instance of RankingCache"""
    global _ranking_cache_instance
    if _ranking_cache_instance is None:
        _ranking_cache_instance = RankingCache(capacity=capacity)
    return _ranking_cache_instance
