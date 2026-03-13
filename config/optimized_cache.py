"""
Optimized Embedding Cache with Singleton Pattern

This module provides a shared embedding cache to prevent duplicate model loading.
"""

import hashlib
import threading
from typing import Optional
import numpy as np

from core.embedding_service import get_embedding_service


class EmbeddingCache:
    """Thread-safe embedding cache using the shared embedding service"""
    
    _instance: Optional["EmbeddingCache"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._service = get_embedding_service()
        self._service.initialize()
        self._initialized = True
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding vector"""
        return self._service.encode_text(text)
    
    def encode_batch(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        """Encode batch of texts"""
        return self._service.encode_batch(texts)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return self._service.cosine_similarity(vec1, vec2)


# Singleton instance
_embedding_cache_instance: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get singleton instance of EmbeddingCache"""
    global _embedding_cache_instance
    if _embedding_cache_instance is None:
        _embedding_cache_instance = EmbeddingCache()
    return _embedding_cache_instance
