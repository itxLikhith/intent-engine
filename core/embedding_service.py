"""
Intent Engine - Shared Embedding Service

This module provides a shared embedding service to eliminate duplicate model loading
across ranker, matcher, and recommender modules.
"""

import hashlib
import logging
import threading
from typing import Optional

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Shared embedding service for encoding text to vectors.

    This singleton service loads the ML model once and shares it across
    all components (ranker, matcher, recommender) to reduce memory usage
    and improve initialization time.
    """

    _instance: Optional["EmbeddingService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.cache: dict[str, np.ndarray] = {}
        self.model = None
        self.tokenizer = None
        self.redis = None
        self.cache_ttl = 3600  # 1 hour TTL for Redis cache
        self._device = "cpu"
        self._initialized = False
        self._init_lock = threading.Lock()

    def initialize(
        self,
        use_redis: bool = True,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the embedding service with the ML model.

        Args:
            use_redis: Whether to use Redis for cross-instance caching
            model_name: Name of the sentence transformer model to load
        """
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            self._load_model(model_name)

            if use_redis:
                self._init_redis()

            self._initialized = True
            logger.info("EmbeddingService initialized successfully")

    def _init_redis(self):
        """Initialize Redis connection for cross-instance caching"""
        try:
            import os

            import redis as redis_lib

            redis_host = os.getenv("REDIS_HOST", "redis")
            redis_port = int(os.getenv("REDIS_PORT", 6379))

            self.redis = redis_lib.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=False,  # We need binary for numpy arrays
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
            )
            # Test connection
            self.redis.ping()
            logger.info(f"EmbeddingService connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"Redis not available for embeddings, using in-memory only: {e}")
            self.redis = None

    def _load_model(self, model_name: str):
        """Load the sentence transformer model"""
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            # Move model to CPU
            self.model = self.model.to(self._device)

            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning("Transformers library not available. Using mock embeddings.")
            self.tokenizer = None
            self.model = None

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return f"emb:{hashlib.md5(text.encode()).hexdigest()}"

    def _get_from_redis(self, text: str) -> np.ndarray | None:
        """Try to get embedding from Redis"""
        if self.redis is None:
            return None

        try:
            key = self._get_cache_key(text)
            cached = self.redis.get(key)
            if cached:
                # Deserialize numpy array
                return np.frombuffer(cached, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Redis get failed: {e}")

        return None

    def _set_in_redis(self, text: str, embedding: np.ndarray):
        """Store embedding in Redis"""
        if self.redis is None:
            return

        try:
            key = self._get_cache_key(text)
            # Serialize numpy array
            self.redis.setex(key, self.cache_ttl, embedding.tobytes())
        except Exception as e:
            logger.debug(f"Redis set failed: {e}")

    def encode_text(self, text: str) -> np.ndarray | None:
        """
        Encode text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as numpy array, or None if encoding failed
        """
        if not self._initialized:
            self.initialize()

        # Check local cache first
        if text in self.cache:
            return self.cache[text]

        # Check Redis cache
        redis_cached = self._get_from_redis(text)
        if redis_cached is not None:
            # Store in local cache for faster access
            self.cache[text] = redis_cached
            return redis_cached

        # Use mock embeddings if model not available
        if self.model is None or self.tokenizer is None:
            # Return deterministic hash-based vector for mock implementation
            hash_input = text.encode("utf-8")
            hash_bytes = hashlib.md5(hash_input).digest()
            # Create a 384-dim vector from hash
            result = np.zeros(384, dtype=np.float32)
            for i in range(384):
                result[i] = (hash_bytes[i % 16] + i * 17) % 100 / 100.0
            # Normalize the vector
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm

            # Cache the result
            self.cache[text] = result
            return result

        try:
            import torch

            # Tokenize the input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling to get sentence embedding
                embeddings = outputs.last_hidden_state.mean(dim=1)

            result = embeddings.cpu().numpy().flatten().astype(np.float32)

            # Cache the result in both local and Redis
            self.cache[text] = result
            self._set_in_redis(text, result)

            return result
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None

    def encode_batch(self, texts: list[str]) -> list[np.ndarray | None]:
        """
        Encode a batch of texts to embedding vectors efficiently.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors (or None if encoding failed)
        """
        if not texts:
            return []

        if not self._initialized:
            self.initialize()

        if self.model is None or self.tokenizer is None:
            # Return deterministic hash-based vectors for mock implementation
            results = []
            for text in texts:
                hash_input = text.encode("utf-8")
                hash_bytes = hashlib.md5(hash_input).digest()
                # Create a 384-dim vector from hash
                result = np.zeros(384, dtype=np.float32)
                for i in range(384):
                    result[i] = (hash_bytes[i % 16] + i * 17) % 100 / 100.0
                # Normalize the vector
                norm = np.linalg.norm(result)
                if norm > 0:
                    result = result / norm
                results.append(result)
            return results

        # Filter out cached texts (check both local and Redis cache)
        uncached_texts = []
        uncached_indices = []
        results: list[np.ndarray | None] = [None] * len(texts)

        for i, text in enumerate(texts):
            # Check local cache first
            if text in self.cache:
                results[i] = self.cache[text]
            else:
                # Check Redis cache
                redis_cached = self._get_from_redis(text)
                if redis_cached is not None:
                    results[i] = redis_cached
                    self.cache[text] = redis_cached  # Also store in local cache
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

        if not uncached_texts:
            # All texts were cached
            return results

        try:
            import torch

            # Tokenize all uncached texts in a batch
            inputs = self.tokenizer(
                uncached_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling to get sentence embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Convert to numpy and cache results
            numpy_embeddings = embeddings.cpu().numpy().astype(np.float32)

            for idx, text_idx in enumerate(uncached_indices):
                result = numpy_embeddings[idx].flatten()
                results[text_idx] = result
                text = texts[text_idx]
                # Cache in both local and Redis
                self.cache[text] = result
                self._set_in_redis(text, result)

            return results

        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            # Fallback to individual encoding
            for i, text in enumerate(texts):
                if results[i] is None:
                    results[i] = self.encode_text(text)
            return results

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return float(dot_product / (norm_vec1 * norm_vec2))

    def clear_cache(self):
        """Clear the local cache"""
        self.cache.clear()
        logger.debug("Embedding cache cleared")


# Global singleton instance
_embedding_service_instance: Optional[EmbeddingService] = None
_embedding_service_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    """
    Get thread-safe singleton instance of EmbeddingService.

    Uses double-checked locking pattern for thread safety.

    Returns:
        Shared EmbeddingService instance
    """
    global _embedding_service_instance
    if _embedding_service_instance is None:
        with _embedding_service_lock:
            if _embedding_service_instance is None:
                _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance


# Convenience functions for direct use
def encode_text(text: str) -> np.ndarray | None:
    """Encode text to embedding vector"""
    return get_embedding_service().encode_text(text)


def encode_batch(texts: list[str]) -> list[np.ndarray | None]:
    """Encode batch of texts to embedding vectors"""
    return get_embedding_service().encode_batch(texts)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return get_embedding_service().cosine_similarity(vec1, vec2)
