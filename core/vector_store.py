"""
Intent Engine - Vector Storage for Intent Embeddings

Uses Qdrant for storing and searching intent vectors.
Enables semantic search over intent-tagged web pages.

Features:
- Intent vector storage
- Semantic similarity search
- Intent-based filtering
- Batch operations
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from core.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class IntentVector:
    """Intent vector with metadata for storage"""

    id: str
    url: str
    embedding: list[float] = field(default_factory=list)
    intent_tags: dict[str, Any] = field(default_factory=dict)
    score: float = 0.5
    title: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "url": self.url,
            "intent_tags": self.intent_tags,
            "score": self.score,
            "title": self.title,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentVector":
        """Create from dictionary"""
        return cls(
            id=data.get("id", ""),
            url=data.get("url", ""),
            intent_tags=data.get("intent_tags", {}),
            score=data.get("score", 0.5),
            title=data.get("title"),
            description=data.get("description"),
        )


class QdrantVectorStore:
    """
    Vector storage and search using Qdrant.

    Usage:
        store = QdrantVectorStore()
        store.store_intent_vector(intent_vector)
        results = store.search_similar(query_embedding, limit=10)

    Features:
    - Store intent vectors with metadata
    - Semantic similarity search
    - Filter by intent tags
    - Batch operations
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.collection_name = self.config.get(
            "collection", "intent_index"
        )
        self.qdrant_url = self.config.get("url", os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.embedding_dim = self.config.get("embedding_dim", 384)  # sentence-transformers

        self._client = None
        self._embedding_service = None
        self._initialized = False

        logger.info(
            f"QdrantVectorStore initialized: collection={self.collection_name}, "
            f"url={self.qdrant_url}"
        )

    def _ensure_initialized(self):
        """Lazy initialization of Qdrant client"""
        if self._initialized:
            return

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams

            self._client = QdrantClient(url=self.qdrant_url, timeout=10)

            # Create collection if not exists
            self._ensure_collection()

            self._initialized = True
            logger.info(f"Qdrant client initialized successfully")

        except ImportError:
            logger.warning(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            )
            self._client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Qdrant client: {e}")
            self._client = None

    def _get_client(self) -> Optional[Any]:
        """Get Qdrant client"""
        self._ensure_initialized()
        return self._client

    def _get_embedding_service(self):
        """Get embedding service"""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def _ensure_collection(self):
        """Create Qdrant collection if not exists"""
        from qdrant_client.http.models import Distance, VectorParams

        client = self._get_client()
        if not client:
            return

        try:
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                # Create collection with specified embedding dimension
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")

                # Create payload indexes for filtering
                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="intent_tags.primary_goal",
                    field_schema="keyword",
                )
                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="intent_tags.skill_level",
                    field_schema="keyword",
                )

                logger.info(f"Created payload indexes for {self.collection_name}")

        except Exception as e:
            logger.warning(f"Failed to create collection: {e}")

    def store_intent_vector(self, intent_vector: IntentVector) -> bool:
        """
        Store an intent vector.

        Args:
            intent_vector: IntentVector to store

        Returns:
            True if successful, False otherwise
        """
        from qdrant_client.http.models import PointStruct

        client = self._get_client()
        if not client:
            logger.warning("Qdrant client not available")
            return False

        try:
            # Generate ID from URL if not provided
            if not intent_vector.id:
                intent_vector.id = self._generate_id(intent_vector.url)

            # Prepare payload
            payload = {
                "url": intent_vector.url,
                "intent_tags": intent_vector.intent_tags,
                "score": intent_vector.score,
                "title": intent_vector.title,
                "description": intent_vector.description,
            }

            # Create point
            point = PointStruct(
                id=self._hash_to_int(intent_vector.id),
                vector=intent_vector.embedding,
                payload=payload,
            )

            # Upsert point
            client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            logger.debug(f"Stored intent vector for {intent_vector.url}")
            return True

        except Exception as e:
            logger.error(f"Failed to store intent vector: {e}")
            return False

    def store_intent_vectors(self, intent_vectors: list[IntentVector]) -> int:
        """
        Store multiple intent vectors in batch.

        Args:
            intent_vectors: List of IntentVector to store

        Returns:
            Number of successfully stored vectors
        """
        from qdrant_client.http.models import PointStruct

        client = self._get_client()
        if not client:
            return 0

        try:
            points = []
            for intent_vector in intent_vectors:
                if not intent_vector.id:
                    intent_vector.id = self._generate_id(intent_vector.url)

                payload = {
                    "url": intent_vector.url,
                    "intent_tags": intent_vector.intent_tags,
                    "score": intent_vector.score,
                    "title": intent_vector.title,
                    "description": intent_vector.description,
                }

                points.append(
                    PointStruct(
                        id=self._hash_to_int(intent_vector.id),
                        vector=intent_vector.embedding,
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(f"Stored {len(points)} intent vectors")
            return len(points)

        except Exception as e:
            logger.error(f"Failed to store batch: {e}")
            return 0

    def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[IntentVector]:
        """
        Search for similar intent vectors.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filters: Optional filters (e.g., {"primary_goal": "LEARN"})

        Returns:
            List of IntentVector sorted by similarity
        """
        from qdrant_client.http.models import (
            Condition,
            FieldCondition,
            Filter,
            MatchValue,
        )

        client = self._get_client()
        if not client:
            logger.warning("Qdrant client not available")
            return []

        try:
            # Build filter
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=f"intent_tags.{key}",
                            match=MatchValue(value=value),
                        )
                    )

                if conditions:
                    search_filter = Filter(must=conditions)

            # Search
            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=search_filter,
            )

            # Convert to IntentVector
            intent_vectors = []
            for r in results:
                payload = r.payload or {}
                intent_vectors.append(
                    IntentVector(
                        id=str(r.id),
                        url=payload.get("url", ""),
                        embedding=[],  # Don't return embedding
                        intent_tags=payload.get("intent_tags", {}),
                        score=r.score,
                        title=payload.get("title"),
                        description=payload.get("description"),
                    )
                )

            logger.debug(f"Search returned {len(intent_vectors)} results")
            return intent_vectors

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by_intent(
        self,
        goal: Optional[str] = None,
        use_case: Optional[str] = None,
        skill_level: Optional[str] = None,
        limit: int = 10,
    ) -> list[IntentVector]:
        """
        Search by intent tags (filtered search).

        Args:
            goal: Filter by primary goal (e.g., "LEARN")
            use_case: Filter by use case
            skill_level: Filter by skill level
            limit: Maximum results

        Returns:
            List of IntentVector matching filters
        """
        filters = {}

        if goal:
            filters["primary_goal"] = goal
        if use_case:
            filters["use_cases"] = use_case
        if skill_level:
            filters["skill_level"] = skill_level

        # For tag-based search, we need to fetch and filter
        # This is a simplified implementation
        client = self._get_client()
        if not client:
            return []

        try:
            # Scroll through collection with filter
            results, _ = client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # Filter results
            filtered_vectors = []
            for r in results:
                payload = r.payload or {}
                intent_tags = payload.get("intent_tags", {})

                # Apply filters
                match = True
                if goal and intent_tags.get("primary_goal") != goal:
                    match = False
                if use_case and use_case not in intent_tags.get("use_cases", []):
                    match = False
                if skill_level and intent_tags.get("skill_level") != skill_level:
                    match = False

                if match:
                    filtered_vectors.append(
                        IntentVector(
                            id=str(r.id),
                            url=payload.get("url", ""),
                            intent_tags=intent_tags,
                            score=payload.get("score", 0.0),
                            title=payload.get("title"),
                            description=payload.get("description"),
                        )
                    )

            return filtered_vectors[:limit]

        except Exception as e:
            logger.error(f"Filtered search failed: {e}")
            return []

    def search_by_query(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[IntentVector]:
        """
        Search by text query (generates embedding automatically).

        Args:
            query: Search query string
            limit: Maximum results
            filters: Optional filters

        Returns:
            List of IntentVector sorted by similarity
        """
        embedding_service = self._get_embedding_service()
        if not embedding_service:
            logger.warning("Embedding service not available")
            return []

        # Generate embedding
        embedding = embedding_service.encode_text(query)
        if embedding is None:
            logger.warning("Failed to generate query embedding")
            return []

        # Search
        return self.search_similar(
            query_embedding=embedding.tolist(),
            limit=limit,
            filters=filters,
        )

    def delete_by_url(self, url: str) -> bool:
        """
        Delete intent vector by URL.

        Args:
            url: URL to delete

        Returns:
            True if successful, False otherwise
        """
        client = self._get_client()
        if not client:
            return False

        try:
            point_id = self._hash_to_int(self._generate_id(url))
            client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id],
            )
            logger.debug(f"Deleted intent vector for {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            return False

    def count(self) -> int:
        """Get total count of vectors in collection"""
        client = self._get_client()
        if not client:
            return 0

        try:
            info = client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception as e:
            logger.error(f"Failed to count: {e}")
            return 0

    def _generate_id(self, url: str) -> str:
        """Generate unique ID from URL"""
        return f"intent_{hashlib.md5(url.encode()).hexdigest()}"

    def _hash_to_int(self, id_str: str) -> int:
        """Convert string ID to integer for Qdrant"""
        # Qdrant requires integer IDs, so we hash and truncate
        return int(hashlib.md5(id_str.encode()).hexdigest()[:15], 16)


# Singleton instance
_vector_store: Optional[QdrantVectorStore] = None


def get_vector_store(config: Optional[dict[str, Any]] = None) -> QdrantVectorStore:
    """
    Get or create vector store singleton.

    Args:
        config: Optional configuration override

    Returns:
        QdrantVectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore(config)
    return _vector_store


def reset_vector_store():
    """Reset vector store singleton (useful for testing)"""
    global _vector_store
    _vector_store = None
