"""
Intent Engine - Core Utilities

This module contains shared utilities used across all components of the Intent Engine.
"""

import logging
import re
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for embeddings to improve performance with LRU eviction"""

    def __init__(self, maxsize: int = 1000):
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            from transformers import AutoModel, AutoTokenizer

            # Use a lightweight model optimized for CPU
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            # Move model to CPU
            self.model = self.model.to("cpu")

            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning("Transformers library not available. Using mock embeddings.")
            self.tokenizer = None
            self.model = None

    def encode_text(self, text: str) -> np.ndarray | None:
        """Encode text to embedding vector using the sentence transformer model"""
        if self.model is None or self.tokenizer is None:
            # Return random vector for mock implementation
            return np.random.rand(384).astype(np.float32)

        # Check cache first (LRU access pattern)
        if text in self.cache:
            self.hits += 1
            # Move to end to mark as recently used
            self.cache.move_to_end(text)
            return self.cache[text]

        self.misses += 1

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

            # Cache the result with LRU eviction
            if len(self.cache) >= self.maxsize:
                # Remove oldest (least recently used) item
                self.cache.popitem(last=False)
            self.cache[text] = result

            return result
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "maxsize": self.maxsize,
        }

    def clear(self):
        """Clear the cache and reset statistics"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return float(dot_product / (norm_vec1 * norm_vec2))


def extract_price_range(text: str) -> tuple | None:
    """
    Extract price range from text using regex patterns
    Returns (operator, value) tuple or None
    """
    # Pattern to match prices like "under 100", "less than 50", etc.
    patterns = [
        r"\b(under|less than|below)\s*(\d+)\s*(?:rupees|rs|₹|dollars?|usd)?\b",
        r"\b(over|more than|above)\s*(\d+)\s*(?:rupees|rs|₹|dollars?|usd)?\b",
        r"\b(above|greater than)\s*(\d+)\s*(?:rupees|rs|₹|dollars?|usd)?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            operator_word = match.group(1)
            value = int(match.group(2))

            # Map operator words to symbols
            if operator_word in ["under", "less than", "below"]:
                return ("<=", value)
            elif operator_word in ["over", "more than", "above", "greater than"]:
                return (">=", value)

    return None


def normalize_datetime(dt_str: str) -> datetime:
    """
    Normalize datetime string to a consistent format
    """
    try:
        # Handle ISO format with Z suffix
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"

        # Parse the datetime
        if "+" in dt_str or dt_str.count("-") > 2:  # Has timezone info
            return datetime.fromisoformat(dt_str)
        else:
            # Naive datetime, treat as UTC
            return datetime.fromisoformat(dt_str).replace(tzinfo=UTC)
    except ValueError:
        # If parsing fails, return current time
        return datetime.now(UTC)


def calculate_expiration_time(hours: int = 8) -> str:
    """
    Calculate expiration time (default 8 hours from now)
    """
    expires_at = (datetime.utcnow() + timedelta(hours=hours)).isoformat() + "Z"
    return expires_at


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks
    """
    # Remove potentially dangerous characters/sequences
    sanitized = re.sub(r'[<>"\';]', "", text)
    return sanitized.strip()


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format
    """
    return datetime.utcnow().isoformat() + "Z"


def fuzzy_match(text1: str, text2: str, threshold: float = 0.7) -> bool:
    """
    Perform fuzzy matching between two texts
    """
    # Simple token-based similarity
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if not union:
        return False

    similarity = len(intersection) / len(union)
    return similarity >= threshold


# Global instance for caching
_embedding_cache_instance = None


def get_embedding_cache() -> EmbeddingCache:
    """
    Get singleton instance of EmbeddingCache
    """
    global _embedding_cache_instance
    if _embedding_cache_instance is None:
        _embedding_cache_instance = EmbeddingCache()
    return _embedding_cache_instance
