"""
Intent Engine - Hybrid Embedding Cache (Redis + Local LRU)

This module provides a two-tier caching strategy:
- L1: Local in-memory LRU cache for ultra-fast access
- L2: Redis distributed cache for multi-instance deployments

Features:
- Automatic fallback to local cache if Redis unavailable
- Cache coherence across instances via Redis
- Configurable TTL for embeddings
- Batch processing for efficiency
- Thread-safe operations
"""

import hashlib
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from config.optimized_cache import LRUCache, OptimizedEmbeddingCache
from config.redis_cache import RedisCache, get_redis_cache

logger = logging.getLogger(__name__)


class HybridEmbeddingCache:
    """
    Two-tier embedding cache with Redis + local LRU.

    Cache lookup order:
    1. Check local L1 cache (fastest)
    2. Check Redis L2 cache (distributed)
    3. Compute embedding and store in both caches

    Cache invalidation:
    - Writes go to both caches
    - Redis TTL ensures automatic expiration
    - Local cache uses LRU eviction
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        local_capacity: int = 5000,
        redis_ttl: int = 3600,  # 1 hour
        use_redis: bool = True,
    ):
        """
        Initialize hybrid cache.

        Args:
            model_name: Sentence transformer model to use
            local_capacity: Local LRU cache capacity
            redis_ttl: Redis cache TTL in seconds
            use_redis: Whether to use Redis (falls back to local only if False)
        """
        self.model_name = model_name
        self.redis_ttl = redis_ttl
        self.use_redis = use_redis and os.getenv("REDIS_ENABLED", "false").lower() == "true"

        # L1: Local cache
        self.local_cache = LRUCache(capacity=local_capacity)

        # L2: Redis cache (optional)
        self.redis_cache: Optional[RedisCache] = None
        if self.use_redis:
            self.redis_cache = get_redis_cache()
            if self.redis_cache is None:
                logger.warning("Redis not available, using local cache only")
                self.use_redis = False

        # Model
        self.model = None
        self.tokenizer = None
        self.batch_size = 32
        self._load_model()

        # Stats
        self._lock = threading.Lock()
        self.hits_l1 = 0
        self.hits_l2 = 0
        self.misses = 0

        logger.info(f"HybridEmbeddingCache initialized (Redis: {self.use_redis})")

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SentenceTransformer model: {self.model_name}...")

            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
            )
            self.model = self.model.to("cpu")
            self.model.eval()

            logger.info(f"Successfully loaded SentenceTransformer model: {self.model_name}")
        except ImportError as e:
            logger.warning(f"SentenceTransformers library not available: {e}. Using mock embeddings.")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading embedding model {self.model_name}: {e}. Using mock embeddings.")
            self.model = None

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{self.model_name}:{text_hash}"

    def encode_text(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Encode single text to embedding with two-tier caching.

        Args:
            text: Text to encode
            use_cache: Whether to use caching

        Returns:
            Embedding vector or None
        """
        if not text or not text.strip():
            return None

        # Check L1 cache
        if use_cache:
            cached = self.local_cache.get(text)
            if cached is not None:
                with self._lock:
                    self.hits_l1 += 1
                return cached

        # Check L2 (Redis) cache
        if use_cache and self.use_redis and self.redis_cache:
            cache_key = self._get_cache_key(text)
            cached = self.redis_cache.get(cache_key)
            if cached is not None and isinstance(cached, np.ndarray):
                # Store in L1 for faster next access
                self.local_cache.put(text, cached)
                with self._lock:
                    self.hits_l2 += 1
                return cached

        # Cache miss - compute embedding
        with self._lock:
            self.misses += 1

        embedding = self._compute_embedding(text)
        if embedding is None:
            return None

        # Store in both caches
        if use_cache:
            self.local_cache.put(text, embedding)

            if self.use_redis and self.redis_cache:
                cache_key = self._get_cache_key(text)
                self.redis_cache.set(cache_key, embedding, ttl=self.redis_ttl)

        return embedding

    def encode_batch(self, texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
        """
        Encode multiple texts with two-tier caching.

        Args:
            texts: List of texts to encode
            use_cache: Whether to use caching

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        results: List[Optional[np.ndarray]] = [None] * len(texts)
        texts_to_encode: List[str] = []
        indices_to_encode: List[int] = []

        # Check caches for all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = np.random.rand(384).astype(np.float32)
                continue

            # Check L1 cache
            cached = self.local_cache.get(text)
            if cached is not None:
                results[i] = cached
                with self._lock:
                    self.hits_l1 += 1
                continue

            # Check L2 (Redis) cache
            if use_cache and self.use_redis and self.redis_cache:
                cache_key = self._get_cache_key(text)
                cached = self.redis_cache.get(cache_key)
                if cached is not None and isinstance(cached, np.ndarray):
                    self.local_cache.put(text, cached)  # Promote to L1
                    results[i] = cached
                    with self._lock:
                        self.hits_l2 += 1
                    continue

            # Need to encode
            texts_to_encode.append(text)
            indices_to_encode.append(i)

        # Encode remaining texts in batches
        if texts_to_encode and self.model is not None:
            try:
                embeddings = self.model.encode(
                    texts_to_encode,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                )

                # Store results and cache
                for idx, embedding in zip(indices_to_encode, embeddings):
                    result = embedding.astype(np.float32)
                    results[idx] = result

                    if use_cache:
                        text = texts_to_encode[idx]
                        self.local_cache.put(text, result)

                        if self.use_redis and self.redis_cache:
                            cache_key = self._get_cache_key(text)
                            self.redis_cache.set(cache_key, result, ttl=self.redis_ttl)

            except Exception as e:
                logger.error(f"Error encoding batch: {e}")
                # Fallback to individual encoding
                for idx, text in zip(indices_to_encode, texts_to_encode):
                    results[idx] = self.encode_text(text, use_cache)

        # Replace any remaining None placeholders
        for i, result in enumerate(results):
            if result is None:
                results[i] = np.random.rand(384).astype(np.float32)

        return results

    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding for text."""
        if self.model is None:
            return np.random.rand(384).astype(np.float32)

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return np.random.rand(384).astype(np.float32)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        dot_product = np.dot(vec1, vec2)
        return float(np.clip(dot_product, -1.0, 1.0))

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits_l1 + self.hits_l2 + self.misses
        hit_rate_l1 = self.hits_l1 / total if total > 0 else 0
        hit_rate_l2 = self.hits_l2 / total if total > 0 else 0

        stats = {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "use_redis": self.use_redis,
            "local_cache": self.local_cache.get_stats(),
            "hits_l1": self.hits_l1,
            "hits_l2": self.hits_l2,
            "misses": self.misses,
            "hit_rate_l1": round(hit_rate_l1, 3),
            "hit_rate_l2": round(hit_rate_l2, 3),
            "total_requests": total,
        }

        if self.use_redis and self.redis_cache:
            stats["redis_cache"] = self.redis_cache.get_stats()

        return stats

    def clear(self) -> None:
        """Clear both caches."""
        self.local_cache.clear()

        if self.use_redis and self.redis_cache:
            # Clear only embedding keys
            keys = self.redis_cache.keys("embedding:*")
            if keys:
                self.redis_cache.delete_many(keys)
                logger.info(f"Cleared {len(keys)} embedding keys from Redis")

        with self._lock:
            self.hits_l1 = 0
            self.hits_l2 = 0
            self.misses = 0

        logger.info("Hybrid embedding cache cleared")


# Global instance
_embedding_cache_instance: Optional[HybridEmbeddingCache] = None
_init_lock = threading.Lock()


def get_embedding_cache(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    local_capacity: int = 5000,
    redis_ttl: int = 3600,
) -> HybridEmbeddingCache:
    """
    Get singleton instance of HybridEmbeddingCache.

    Args:
        model_name: Model name for initialization
        local_capacity: Local cache capacity
        redis_ttl: Redis TTL in seconds

    Returns:
        HybridEmbeddingCache instance
    """
    global _embedding_cache_instance

    if _embedding_cache_instance is None:
        with _init_lock:
            if _embedding_cache_instance is None:
                _embedding_cache_instance = HybridEmbeddingCache(
                    model_name=model_name,
                    local_capacity=local_capacity,
                    redis_ttl=redis_ttl,
                )

    return _embedding_cache_instance


def clear_embedding_cache() -> None:
    """Clear the global cache instance."""
    global _embedding_cache_instance
    with _init_lock:
        if _embedding_cache_instance is not None:
            _embedding_cache_instance.clear()
        _embedding_cache_instance = None
