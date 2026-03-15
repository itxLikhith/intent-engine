"""
Intent Engine - Embedding Service
Generates sentence embeddings for vector search using sentence-transformers
"""

import logging

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Using fallback embeddings.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service

        Args:
            model_name: Sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("Using fallback random embeddings (install sentence-transformers for real embeddings)")

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings

        Args:
            texts: List of texts to encode
            normalize: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            # Fallback: random embeddings (for testing without model)
            logger.warning("Using fallback random embeddings")
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            if normalize:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings

        try:
            embeddings = self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=False
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            # Fallback to random
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            if normalize:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query"""
        embeddings = self.encode([query])
        return embeddings[0].tolist()

    def encode_documents(self, documents: list[dict]) -> list[list[float]]:
        """
        Encode multiple documents

        Args:
            documents: List of dicts with 'text' key

        Returns:
            List of embedding vectors
        """
        texts = [doc.get("text", doc.get("content", "")) for doc in documents]
        embeddings = self.encode(texts)
        return embeddings.tolist()

    @property
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self.model is not None or not SENTENCE_TRANSFORMERS_AVAILABLE


# Global instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """Get or create embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name)
    return _embedding_service


if __name__ == "__main__":
    # Test the service
    service = get_embedding_service()

    test_texts = [
        "golang tutorial for beginners",
        "how to fix connection error in go",
        "best practices for go web development",
    ]

    embeddings = service.encode(test_texts)
    print(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape}")

    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    print(f"Similarity between text 1 and 2: {similarity[0][0]:.4f}")
