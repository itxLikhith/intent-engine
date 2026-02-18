"""
Intent Engine - Redis Cache Implementation

This module provides a Redis-backed cache for distributed deployments.
It replaces the in-memory cache for multi-instance setups.

Features:
- Distributed caching across multiple instances
- TTL-based expiration
- LRU eviction via Redis maxmemory policy
- Serialization for complex objects (embeddings, query results)
- Connection pooling
- Health checks
"""

import hashlib
import json
import logging
import os
import threading
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import redis
from redis import ConnectionPool

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-backed distributed cache with support for:
    - String, JSON, and numpy array values
    - TTL-based expiration
    - LRU eviction (via Redis maxmemory policy)
    - Connection pooling
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        max_connections: int = 50,
        decode_responses: bool = False,
    ):
        """
        Initialize Redis cache client.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            max_connections: Maximum connections in pool
            decode_responses: Whether to decode responses to strings
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password

        # Create connection pool
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            max_connections=max_connections,
            decode_responses=decode_responses,
        )

        self.client = redis.Redis(connection_pool=self.pool)
        self._lock = threading.Lock()

        logger.info(f"Redis cache initialized: {host}:{port}/{db}")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            value = self.client.get(key)
            if value is None:
                return None

            # Try to deserialize
            return self._deserialize(value)
        except redis.RedisError as e:
            logger.error(f"Redis GET error for key '{key}': {e}")
            return None
        except Exception as e:
            logger.error(f"Error deserializing value for key '{key}': {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (string, dict, list, numpy array)
            ttl: Time-to-live in seconds (None = no expiration)

        Returns:
            True if successful, False otherwise
        """
        try:
            serialized = self._serialize(value)
            if ttl:
                return self.client.setex(key, ttl, serialized)
            else:
                return self.client.set(key, serialized)
        except redis.RedisError as e:
            logger.error(f"Redis SET error for key '{key}': {e}")
            return False
        except Exception as e:
            logger.error(f"Error serializing value for key '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False otherwise
        """
        try:
            return self.client.delete(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis DELETE error for key '{key}': {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        try:
            return self.client.exists(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis EXISTS error for key '{key}': {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """
        Get TTL for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        try:
            return self.client.ttl(key)
        except redis.RedisError as e:
            logger.error(f"Redis TTL error for key '{key}': {e}")
            return -2

    def set_ttl(self, key: str, ttl: int) -> bool:
        """
        Set TTL for an existing key.

        Args:
            key: Cache key
            ttl: Time-to-live in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            return self.client.expire(key, ttl)
        except redis.RedisError as e:
            logger.error(f"Redis EXPIRE error for key '{key}': {e}")
            return False

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key-value pairs
        """
        try:
            values = self.client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
            return result
        except redis.RedisError as e:
            logger.error(f"Redis MGET error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error deserializing values: {e}")
            return {}

    def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set multiple values in cache.

        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time-to-live in seconds (applied to all)

        Returns:
            True if successful, False otherwise
        """
        try:
            serialized = {k: self._serialize(v) for k, v in mapping.items()}
            if ttl:
                # Use pipeline for atomic operation with TTL
                pipe = self.client.pipeline()
                for key, value in serialized.items():
                    pipe.setex(key, ttl, value)
                pipe.execute()
                return True
            else:
                return self.client.mset(serialized)
        except redis.RedisError as e:
            logger.error(f"Redis MSET error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error serializing values: {e}")
            return False

    def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys from cache.

        Args:
            keys: List of cache keys

        Returns:
            Number of keys deleted
        """
        try:
            return self.client.delete(*keys)
        except redis.RedisError as e:
            logger.error(f"Redis DELETE error: {e}")
            return 0

    def keys(self, pattern: str = "*") -> List[str]:
        """
        Get all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "embedding:*", "query:*")

        Returns:
            List of matching keys

        Note: Use with caution in production - can be slow for large datasets
        """
        try:
            return [k.decode() if isinstance(k, bytes) else k for k in self.client.keys(pattern)]
        except redis.RedisError as e:
            logger.error(f"Redis KEYS error: {e}")
            return []

    def flush(self) -> bool:
        """
        Flush current database.

        Returns:
            True if successful, False otherwise

        Warning: This deletes ALL keys in the current database!
        """
        try:
            self.client.flushdb()
            logger.warning("Redis cache flushed - all keys deleted")
            return True
        except redis.RedisError as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            return False

    def health_check(self) -> bool:
        """
        Check Redis connection health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            return self.client.ping()
        except redis.RedisError as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Redis server statistics.

        Returns:
            Dictionary with Redis stats
        """
        try:
            info = self.client.info("memory")
            keys_count = self.client.dbsize()
            return {
                "connected": self.health_check(),
                "keys_count": keys_count,
                "used_memory": info.get("used_memory_human", "unknown"),
                "used_memory_peak": info.get("used_memory_peak_human", "unknown"),
                "maxmemory": info.get("maxmemory_human", "unlimited"),
                "eviction_policy": info.get("maxmemory_policy", "noeviction"),
            }
        except redis.RedisError as e:
            logger.error(f"Redis INFO error: {e}")
            return {"connected": False, "error": str(e)}

    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for Redis storage.

        Supports:
        - Strings
        - Numbers
        - Lists/Dicts (JSON)
        - Numpy arrays
        """
        if isinstance(value, np.ndarray):
            # Serialize numpy array as JSON with metadata
            data = {
                "__type__": "numpy",
                "data": value.tolist(),
                "dtype": str(value.dtype),
                "shape": value.shape,
            }
            return json.dumps(data).encode("utf-8")
        elif isinstance(value, (dict, list, tuple)):
            # Serialize as JSON with type marker
            data = {
                "__type__": "json",
                "data": value,
            }
            return json.dumps(data).encode("utf-8")
        elif isinstance(value, str):
            # Check if it's already a serialized value
            if value.startswith('{"__type__":'):
                return value.encode("utf-8")
            # Plain string
            return value.encode("utf-8")
        else:
            # Other types (numbers, bool, etc.)
            return json.dumps(value).encode("utf-8")

    def _deserialize(self, value: bytes) -> Any:
        """
        Deserialize value from Redis storage.
        """
        if isinstance(value, bytes):
            value = value.decode("utf-8")

        try:
            data = json.loads(value)
            if isinstance(data, dict):
                type_marker = data.get("__type__")
                if type_marker == "numpy":
                    # Reconstruct numpy array
                    arr = np.array(data["data"], dtype=data["dtype"])
                    return arr.reshape(data["shape"])
                elif type_marker == "json":
                    return data["data"]
            return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Return as string if not JSON
            return value

    def close(self) -> None:
        """
        Close Redis connection.
        """
        try:
            self.client.close()
            logger.info("Redis connection closed")
        except redis.RedisError as e:
            logger.error(f"Error closing Redis connection: {e}")


# Global instance
_redis_cache_instance: Optional[RedisCache] = None
_init_lock = threading.Lock()


def get_redis_cache() -> Optional[RedisCache]:
    """
    Get or create Redis cache singleton from environment configuration.

    Returns:
        RedisCache instance or None if Redis is not configured
    """
    global _redis_cache_instance

    # Check if Redis is enabled
    redis_enabled = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    if not redis_enabled:
        return None

    if _redis_cache_instance is None:
        with _init_lock:
            if _redis_cache_instance is None:
                # Read configuration from environment
                host = os.getenv("REDIS_HOST", "localhost")
                port = int(os.getenv("REDIS_PORT", "6379"))
                db = int(os.getenv("REDIS_DB", "0"))
                password = os.getenv("REDIS_PASSWORD")
                max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))

                _redis_cache_instance = RedisCache(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    max_connections=max_connections,
                )

    return _redis_cache_instance


def initialize_redis_cache(
    host: str = None,
    port: int = None,
    db: int = None,
    password: str = None,
) -> RedisCache:
    """
    Initialize Redis cache with explicit configuration.

    Args:
        host: Redis hostname
        port: Redis port
        db: Redis database
        password: Redis password

    Returns:
        RedisCache instance
    """
    global _redis_cache_instance

    with _init_lock:
        # Close existing connection
        if _redis_cache_instance is not None:
            _redis_cache_instance.close()

        # Use provided values or environment defaults
        host = host or os.getenv("REDIS_HOST", "localhost")
        port = port or int(os.getenv("REDIS_PORT", "6379"))
        db = db if db is not None else int(os.getenv("REDIS_DB", "0"))
        password = password or os.getenv("REDIS_PASSWORD")

        _redis_cache_instance = RedisCache(
            host=host,
            port=port,
            db=db,
            password=password,
        )

        logger.info(f"Redis cache initialized: {host}:{port}/{db}")
        return _redis_cache_instance


def clear_redis_cache() -> None:
    """
    Clear the global Redis cache instance.
    """
    global _redis_cache_instance
    with _init_lock:
        if _redis_cache_instance is not None:
            _redis_cache_instance.close()
        _redis_cache_instance = None
