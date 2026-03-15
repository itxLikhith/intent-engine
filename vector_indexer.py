"""
Intent Engine - Vector Indexer Service
Indexes crawled pages into Qdrant for semantic search
"""

import logging
from dataclasses import dataclass

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document for vector search"""

    id: str
    url: str
    title: str
    content: str
    intent_goal: str
    topics: list[str]
    skill_level: str
    crawled_at: str
    vector: list[float]


class QdrantClient:
    """Simple Qdrant client for vector operations"""

    def __init__(self, host: str = "qdrant", port: int = 6333, collection: str = "intent_vectors"):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.collection = collection
        self.vector_size = 384  # Default for all-MiniLM-L6-v2
        self.distance = "Cosine"

    def ping(self) -> bool:
        """Check if Qdrant is available"""
        try:
            resp = requests.get(f"{self.base_url}/", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Qdrant ping failed: {e}")
            return False

    def create_collection(self) -> bool:
        """Create collection if it doesn't exist"""
        # Check if exists
        try:
            resp = requests.get(f"{self.base_url}/collections/{self.collection}")
            if resp.status_code == 200:
                logger.info(f"Collection {self.collection} already exists")
                return True
        except Exception as e:
            logger.error(f"Failed to check collection: {e}")

        # Create collection
        config = {
            "vectors": {"size": self.vector_size, "distance": self.distance},
            "hnsw_config": {"m": 16, "ef_construct": 100},
        }

        try:
            resp = requests.put(f"{self.base_url}/collections/{self.collection}", json=config, timeout=30)
            if resp.status_code == 200:
                logger.info(f"Created collection {self.collection}")
                return True
            else:
                logger.error(f"Failed to create collection: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def upsert(self, documents: list[VectorDocument]) -> bool:
        """Upsert documents into Qdrant"""
        if not documents:
            return True

        points = []
        for doc in documents:
            point = {
                "id": doc.id,
                "vector": doc.vector,
                "payload": {
                    "url": doc.url,
                    "title": doc.title,
                    "content": doc.content[:500],  # Truncate for payload
                    "intent_goal": doc.intent_goal,
                    "topics": doc.topics,
                    "skill_level": doc.skill_level,
                    "crawled_at": doc.crawled_at,
                },
            }
            points.append(point)

        try:
            resp = requests.put(
                f"{self.base_url}/collections/{self.collection}/points", json={"points": points}, timeout=30
            )
            if resp.status_code == 200:
                logger.info(f"Upserted {len(documents)} documents")
                return True
            else:
                logger.error(f"Failed to upsert: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to upsert: {e}")
            return False

    def search(self, query_vector: list[float], limit: int = 10) -> list[dict]:
        """Search for similar documents"""
        query = {"vector": query_vector, "limit": limit, "with_payload": True, "with_vector": False}

        try:
            resp = requests.post(f"{self.base_url}/collections/{self.collection}/points/search", json=query, timeout=10)
            if resp.status_code == 200:
                results = resp.json().get("result", [])
                logger.info(f"Found {len(results)} results")
                return results
            else:
                logger.error(f"Search failed: {resp.text}")
                return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_stats(self) -> dict:
        """Get collection statistics"""
        try:
            resp = requests.get(f"{self.base_url}/collections/{self.collection}")
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": data.get("status", "unknown"),
                    "points_count": data.get("result", {}).get("points_count", 0),
                    "vectors_count": data.get("result", {}).get("vectors_count", 0),
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
        return {"status": "error", "points_count": 0}


class VectorIndexer:
    """Indexes crawled pages into Qdrant"""

    def __init__(self, qdrant_host: str = "qdrant", qdrant_port: int = 6333):
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedding_service = None

    def initialize(self) -> bool:
        """Initialize the indexer"""
        # Import embedding service
        try:
            from embedding_service import get_embedding_service

            self.embedding_service = get_embedding_service()
            logger.info("Embedding service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            return False

        # Check Qdrant
        if not self.qdrant.ping():
            logger.error("Qdrant not available")
            return False

        # Create collection
        if not self.qdrant.create_collection():
            logger.error("Failed to create collection")
            return False

        logger.info("Vector indexer initialized successfully")
        return True

    def index_documents(self, documents: list[dict]) -> int:
        """
        Index documents into Qdrant

        Args:
            documents: List of document dicts with keys:
                - id, url, title, content, intent_goal, topics, skill_level, crawled_at

        Returns:
            Number of documents indexed
        """
        if not documents or not self.embedding_service:
            logger.warning("No documents or embedding service not available")
            return 0

        # Prepare texts for embedding
        texts = []
        vector_docs = []

        for doc in documents:
            # Create text for embedding (title + content)
            title = doc.get("title", "")
            content = doc.get("content", "")[:2000]  # Limit content length
            text = f"{title} {content}"
            texts.append(text)

            # Generate numeric ID from database ID or URL hash
            doc_id_str = doc.get("id", "")
            try:
                # Try to use numeric ID directly
                doc_id = int(doc_id_str)
            except (ValueError, TypeError):
                # Use hash of URL as fallback (convert to positive int)
                doc_id = abs(hash(doc.get("url", ""))) % (10**18)

            # Create vector document (with placeholder vector)
            vector_docs.append(
                {
                    "id": doc_id,
                    "url": doc.get("url", ""),
                    "title": title,
                    "content": content,
                    "intent_goal": doc.get("intent_goal", "learn"),
                    "topics": doc.get("topics", []),
                    "skill_level": doc.get("skill_level", "intermediate"),
                    "crawled_at": doc.get("crawled_at", ""),
                }
            )

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.embedding_service.encode(texts)

        # Add vectors to documents
        for i, doc_dict in enumerate(vector_docs):
            doc_dict["vector"] = embeddings[i].tolist()

        # Convert to VectorDocument objects
        vector_doc_objects = []
        for doc_dict in vector_docs:
            vector_doc_objects.append(VectorDocument(**doc_dict))

        # Upsert to Qdrant
        if self.qdrant.upsert(vector_doc_objects):
            logger.info(f"Successfully upserted {len(vector_doc_objects)} documents")
            return len(vector_doc_objects)

        logger.error("Failed to upsert documents")
        return 0

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search for similar documents"""
        if not self.embedding_service:
            return []

        # Encode query
        query_vector = self.embedding_service.encode_query(query)

        # Search Qdrant
        results = self.qdrant.search(query_vector, limit)

        # Format results
        formatted = []
        for result in results:
            payload = result.get("payload", {})
            formatted.append(
                {
                    "url": payload.get("url", ""),
                    "title": payload.get("title", ""),
                    "content": payload.get("content", ""),
                    "score": result.get("score", 0),
                    "intent_goal": payload.get("intent_goal", ""),
                    "topics": payload.get("topics", []),
                    "source": "vector",
                }
            )

        return formatted

    def get_stats(self) -> dict:
        """Get indexer statistics"""
        return self.qdrant.get_stats()


if __name__ == "__main__":
    # Test the indexer
    indexer = VectorIndexer()
    if indexer.initialize():
        print("Indexer initialized successfully")

        # Test search
        results = indexer.search("golang tutorial", limit=5)
        print(f"Found {len(results)} results")
        for r in results:
            print(f"  - {r['title']} (score: {r['score']:.4f})")
    else:
        print("Failed to initialize indexer")
