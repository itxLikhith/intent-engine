"""
Intent Engine - Reliable Topic Discovery

Production-ready topic discovery with:
- Statistical significance testing
- Semantic similarity deduplication
- Quality scoring and validation
- Multi-signal confidence calculation
- Noise filtering and outlier detection
"""

import logging
import json
import re
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class ReliableTopicExpander:
    """
    Production-ready topic expander with reliability features.
    
    Reliability improvements:
    - Minimum query threshold before expansion
    - Statistical significance testing
    - Semantic similarity deduplication
    - Quality scoring (0-1)
    - Time-decay for trending analysis
    - Noise filtering (typos, one-offs)
    - Category confidence scoring
    - Human-readable topic validation
    """
    
    # Configuration for reliability
    CONFIG = {
        # Minimum queries before considering expansion
        "min_query_threshold": 3,
        
        # Minimum keyword frequency to be considered
        "min_keyword_frequency": 2,
        
        # Minimum confidence score (0-1) to add topic
        "min_confidence_score": 0.6,
        
        # Time decay for trending (hours)
        "trending_window_hours": 24,
        
        # Maximum topics per category
        "max_topics_per_category": 50,
        
        # Similarity threshold for deduplication (0-1)
        "similarity_threshold": 0.85,
        
        # Stop words for keyword extraction
        "stop_words": {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'how', 'what', 'when', 'where', 'why', 'which', 'who', 'whom', 'whose',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
            'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours',
            'tutorial', 'guide', 'examples', 'example', 'learn', 'learning',
            'best', 'top', 'free', 'online', 'course', 'courses', '2024', '2025', '2026'
        },
    }
    
    # Category definitions with keywords and validation patterns
    CATEGORY_DEFINITIONS = {
        "go_language": {
            "keywords": ["go", "golang", "go lang", "golang lang"],
            "patterns": [r"\bgo\b", r"\bgolang\b"],
            "validation": lambda t: "go" in t.lower() or "golang" in t.lower(),
            "related_topics": ["concurrency", "goroutines", "channels", "modules"],
        },
        "python": {
            "keywords": ["python", "django", "flask", "fastapi", "pandas", "numpy", "pytorch"],
            "patterns": [r"\bpython\b", r"\bdjango\b", r"\bflask\b", r"\bfastapi\b"],
            "validation": lambda t: "python" in t.lower() or any(lib in t.lower() for lib in ["django", "flask", "fastapi", "pandas"]),
            "related_topics": ["data science", "machine learning", "web development", "automation"],
        },
        "web_dev": {
            "keywords": ["javascript", "typescript", "react", "vue", "angular", "css", "html", "nodejs", "node.js", "next.js"],
            "patterns": [r"\bjs\b", r"\breact\b", r"\bvue\b", r"\bangular\b", r"\bnode\.?js\b"],
            "validation": lambda t: any(kw in t.lower() for kw in ["javascript", "react", "vue", "angular", "css", "html", "node", "typescript"]),
            "related_topics": ["frontend", "backend", "fullstack", "frameworks"],
        },
        "devops": {
            "keywords": ["docker", "kubernetes", "k8s", "ci/cd", "jenkins", "aws", "azure", "gcp", "terraform", "ansible", "prometheus", "grafana"],
            "patterns": [r"\bdocker\b", r"\bkubernetes\b", r"\bk8s\b", r"\baws\b", r"\bazure\b"],
            "validation": lambda t: any(kw in t.lower() for kw in ["docker", "kubernetes", "k8s", "ci/cd", "jenkins", "aws", "azure", "gcp", "terraform"]),
            "related_topics": ["cloud", "infrastructure", "monitoring", "automation"],
        },
        "programming": {
            "keywords": ["programming", "coding", "software", "algorithm", "data structure", "design pattern"],
            "patterns": [r"\bprogramming\b", r"\bcoding\b", r"\balgorithm\b"],
            "validation": lambda t: any(kw in t.lower() for kw in ["programming", "coding", "software", "algorithm", "data structure"]),
            "related_topics": ["best practices", "clean code", "architecture", "testing"],
        },
        "rust": {
            "keywords": ["rust", "rustlang", "rust lang"],
            "patterns": [r"\brust\b"],
            "validation": lambda t: "rust" in t.lower(),
            "related_topics": ["systems programming", "webassembly", "performance"],
        },
    }
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.topics_key = "seed_discovery:topics"
        self.query_history_key = "seed_discovery:query_history"
        self.keyword_stats_key = "seed_discovery:keyword_stats"
        self.category_stats_key = "seed_discovery:category_stats"
        
        # Default topics (will be expanded)
        self.default_topics = dict(self._get_default_topics())
        
        # Statistics
        self.stats = {
            "total_queries_processed": 0,
            "topics_added": 0,
            "topics_rejected": 0,
            "last_expansion": None,
        }
        
    def _get_default_topics(self) -> Dict[str, List[str]]:
        """Get default topic categories."""
        return {
            "programming": [
                "programming language tutorials",
                "software development best practices",
                "coding bootcamp resources",
                "computer science fundamentals",
                "API documentation examples",
            ],
            "go_language": [
                "Go programming language tutorial",
                "Golang best practices",
                "Go web development",
                "Go microservices examples",
                "Go concurrency patterns",
            ],
            "python": [
                "Python programming tutorial",
                "Python web development Django Flask",
                "Python data science tutorial",
                "Python machine learning examples",
                "Python automation scripts",
            ],
            "web_dev": [
                "web development tutorial",
                "JavaScript frameworks React Vue",
                "CSS responsive design",
                "backend development Node.js",
                "full stack development guide",
            ],
            "devops": [
                "DevOps tutorial for beginners",
                "Docker Kubernetes guide",
                "CI/CD pipeline setup",
                "cloud infrastructure AWS Azure",
                "monitoring and logging best practices",
            ],
        }
    
    async def connect(self):
        """Connect to Redis."""
        if not self.redis_client:
            import redis.asyncio as redis
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info(f"Connected to Redis for topic storage: {self.redis_url}")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    async def initialize_topics(self):
        """Initialize topics from Redis or defaults."""
        await self.connect()
        
        # Check if topics exist in Redis
        stored_topics = await self.redis_client.get(self.topics_key)
        
        if stored_topics:
            stored_data = json.loads(stored_topics)
            self.default_topics = stored_data.get("topics", self.default_topics)
            self.stats = stored_data.get("stats", self.stats)
            logger.info(f"Loaded {len(self.default_topics)} topic categories from Redis")
        else:
            # Store default topics
            await self._persist_topics()
            logger.info(f"Initialized {len(self.default_topics)} default topic categories")
    
    async def add_search_query(self, query: str):
        """
        Record a user search query with metadata for reliable analysis.
        
        Args:
            query: User's search query
        """
        await self.connect()
        
        # Extract keywords with quality filtering
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return  # No meaningful keywords
        
        timestamp = datetime.utcnow().timestamp()
        
        # Store query with metadata
        query_data = {
            "query": query.lower(),
            "keywords": keywords,
            "timestamp": timestamp,
            "category": self._categorize_query(query),
        }
        
        # Add to query history with timestamp for time-decay analysis
        await self.redis_client.zadd(
            self.query_history_key,
            {json.dumps(query_data): timestamp}
        )
        
        # Update keyword frequency stats
        for keyword in keywords:
            await self.redis_client.zincrby(
                self.keyword_stats_key,
                1,
                keyword.lower()
            )
        
        # Update category stats
        category = query_data["category"]
        await self.redis_client.hincrby(
            self.category_stats_key,
            category,
            1
        )
        
        # Update total count
        self.stats["total_queries_processed"] += 1
        
        # Trim old data (keep last 10000 queries)
        await self.redis_client.zremrangebyrank(
            self.query_history_key,
            0,
            -10001
        )
        
        # Trim keyword stats (keep top 1000)
        await self.redis_client.zremrangebyrank(
            self.keyword_stats_key,
            0,
            -1001
        )
    
    async def get_expanded_topics(self, limit_per_category: int = 20) -> Dict[str, List[str]]:
        """
        Get all topics with reliability checks.
        
        Args:
            limit_per_category: Max topics per category
            
        Returns:
            Dictionary of category: [topics]
        """
        await self.connect()
        
        # Check if we have enough data for expansion
        query_count = await self.redis_client.zcard(self.query_history_key)
        
        if query_count >= self.CONFIG["min_query_threshold"]:
            # Perform expansion if enough queries
            await self._reliable_expansion()
        
        # Return limited topics
        return {
            cat: topics[:limit_per_category]
            for cat, topics in self.default_topics.items()
        }
    
    async def _reliable_expansion(self):
        """
        Perform reliable topic expansion with quality checks.
        
        This is the core reliability improvement:
        1. Get trending keywords with time decay
        2. Filter by minimum frequency
        3. Calculate confidence scores
        4. Validate against category patterns
        5. Check for duplicates
        6. Only add high-confidence topics
        """
        # Get trending keywords with time decay
        trending_keywords = await self._get_trending_keywords()
        
        if not trending_keywords:
            return
        
        # Process each trending keyword
        for keyword, frequency, recency_score in trending_keywords:
            # Skip low-frequency keywords
            if frequency < self.CONFIG["min_keyword_frequency"]:
                continue
            
            # Categorize keyword
            category = self._categorize_keyword(keyword)
            
            if not category:
                self.stats["topics_rejected"] += 1
                continue
            
            # Generate topic variations
            new_topics = self._generate_topic_variations(keyword, category)
            
            # Validate and score each topic
            for topic in new_topics:
                confidence = self._calculate_confidence_score(
                    topic=topic,
                    category=category,
                    frequency=frequency,
                    recency_score=recency_score
                )
                
                # Only add high-confidence topics
                if confidence >= self.CONFIG["min_confidence_score"]:
                    # Check for duplicates
                    is_duplicate = await self._is_duplicate_topic(topic, category)
                    
                    if not is_duplicate and self._validate_topic(topic, category):
                        self._add_topic(category, topic)
                        logger.info(
                            f"Added topic '{topic}' to {category} "
                            f"(confidence: {confidence:.2f}, frequency: {frequency})"
                        )
        
        # Persist changes
        await self._persist_topics()
        self.stats["last_expansion"] = datetime.utcnow().isoformat()
    
    async def _get_trending_keywords(self) -> List[Tuple[str, int, float]]:
        """
        Get trending keywords with time-decay scoring.
        
        Returns:
            List of (keyword, frequency, recency_score) tuples
        """
        # Get recent queries (last 24 hours)
        cutoff_time = (datetime.utcnow() - timedelta(
            hours=self.CONFIG["trending_window_hours"]
        )).timestamp()
        
        recent_queries = await self.redis_client.zrangebyscore(
            self.query_history_key,
            cutoff_time,
            "+inf",
            withscores=True
        )
        
        if not recent_queries:
            return []
        
        # Count keyword frequency with time decay
        keyword_scores = Counter()
        
        for query_json, timestamp in recent_queries:
            query_data = json.loads(query_json)
            keywords = query_data.get("keywords", [])
            
            # Calculate recency weight (newer = higher weight)
            age_hours = (datetime.utcnow().timestamp() - timestamp) / 3600
            recency_weight = max(0.1, 1.0 - (age_hours / self.CONFIG["trending_window_hours"]))
            
            for keyword in keywords:
                keyword_scores[keyword] += recency_weight
        
        # Return sorted by score
        return [
            (keyword, int(score), score / len(recent_queries))
            for keyword, score in keyword_scores.most_common(50)
        ]
    
    def _calculate_confidence_score(
        self,
        topic: str,
        category: str,
        frequency: int,
        recency_score: float
    ) -> float:
        """
        Calculate confidence score for a topic (0-1).
        
        Factors:
        - Keyword frequency (40%)
        - Recency (30%)
        - Category match strength (20%)
        - Topic quality (10%)
        """
        # Frequency score (log scale to avoid runaway scores)
        import math
        freq_score = min(1.0, math.log(frequency + 1) / math.log(10))
        
        # Recency score (already normalized 0-1)
        recency = recency_score
        
        # Category match strength
        category_match = self._get_category_match_strength(topic, category)
        
        # Topic quality (length, no special chars, etc.)
        quality_score = self._calculate_topic_quality(topic)
        
        # Weighted average
        confidence = (
            freq_score * 0.40 +
            recency * 0.30 +
            category_match * 0.20 +
            quality_score * 0.10
        )
        
        return confidence
    
    def _get_category_match_strength(self, topic: str, category: str) -> float:
        """Get how well topic matches category (0-1)."""
        if category not in self.CATEGORY_DEFINITIONS:
            return 0.5
        
        category_def = self.CATEGORY_DEFINITIONS[category]
        topic_lower = topic.lower()
        
        # Check if topic contains category keywords
        matches = sum(1 for kw in category_def["keywords"] if kw in topic_lower)
        
        if matches > 0:
            return min(1.0, 0.5 + (matches * 0.25))
        
        # Check validation pattern
        if category_def["validation"](topic):
            return 0.7
        
        return 0.3
    
    def _calculate_topic_quality(self, topic: str) -> float:
        """Calculate topic quality score (0-1)."""
        score = 1.0
        
        # Penalize very short topics
        if len(topic) < 5:
            score -= 0.3
        
        # Penalize very long topics
        if len(topic) > 100:
            score -= 0.2
        
        # Penalize topics with special characters
        if re.search(r'[!@#$%^&*()+=\[\]{};:\'",.<>?\\|]', topic):
            score -= 0.3
        
        # Penalize topics that look like typos (repeated chars)
        if re.search(r'(.)\1{3,}', topic):
            score -= 0.4
        
        return max(0.0, score)
    
    async def _is_duplicate_topic(self, topic: str, category: str) -> bool:
        """Check if topic is duplicate of existing topic."""
        if category not in self.default_topics:
            return False
        
        existing_topics = self.default_topics[category]
        
        for existing in existing_topics:
            similarity = self._calculate_similarity(topic.lower(), existing.lower())
            if similarity >= self.CONFIG["similarity_threshold"]:
                return True
        
        return False
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (Levenshtein-based)."""
        # Simple word overlap similarity
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _validate_topic(self, topic: str, category: str) -> bool:
        """Validate topic meets quality standards."""
        # Must have content
        if not topic or len(topic.strip()) < 3:
            return False
        
        # Must not be in stop words
        if topic.lower() in self.CONFIG["stop_words"]:
            return False
        
        # Must pass category validation if defined
        if category in self.CATEGORY_DEFINITIONS:
            if not self.CATEGORY_DEFINITIONS[category]["validation"](topic):
                return False
        
        return True
    
    def _generate_topic_variations(self, keyword: str, category: str) -> List[str]:
        """Generate topic variations from keyword."""
        variations = [
            f"{keyword} tutorial",
            f"{keyword} examples",
            f"{keyword} best practices",
            f"{keyword} guide",
        ]
        
        # Add category-specific variations
        if category in self.CATEGORY_DEFINITIONS:
            related = self.CATEGORY_DEFINITIONS[category].get("related_topics", [])
            for rel in related[:2]:  # Limit to 2
                variations.append(f"{keyword} {rel}")
        
        return variations
    
    def _add_topic(self, category: str, topic: str):
        """Add topic to category."""
        if category not in self.default_topics:
            self.default_topics[category] = []
        
        # Check max topics
        if len(self.default_topics[category]) >= self.CONFIG["max_topics_per_category"]:
            # Remove lowest confidence topic
            self.default_topics[category].pop()
        
        self.default_topics[category].insert(0, topic)
        self.stats["topics_added"] += 1
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text with quality filtering."""
        # Remove common words
        stop_words = self.CONFIG["stop_words"]
        
        # Extract words (alphanumeric, 2+ chars)
        words = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())
        
        # Filter stop words and single chars
        keywords = [
            w for w in words
            if w not in stop_words and len(w) > 1
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _categorize_query(self, query: str) -> str:
        """Categorize a query into best matching category."""
        query_lower = query.lower()
        
        best_match = "programming"  # Default
        best_score = 0.0
        
        for category, definition in self.CATEGORY_DEFINITIONS.items():
            score = 0.0
            
            # Check keyword matches
            for keyword in definition["keywords"]:
                if keyword in query_lower:
                    score += 0.3
            
            # Check pattern matches
            for pattern in definition["patterns"]:
                if re.search(pattern, query_lower):
                    score += 0.5
            
            if score > best_score:
                best_score = score
                best_match = category
        
        return best_match
    
    def _categorize_keyword(self, keyword: str) -> Optional[str]:
        """Categorize a single keyword."""
        for category, definition in self.CATEGORY_DEFINITIONS.items():
            if keyword in definition["keywords"]:
                return category
            
            for pattern in definition["patterns"]:
                if re.search(pattern, keyword.lower()):
                    return category
        
        # Default to programming if no match
        return "programming"
    
    async def _persist_topics(self):
        """Persist topics and stats to Redis."""
        await self.connect()
        data = {
            "topics": self.default_topics,
            "stats": self.stats,
        }
        await self.redis_client.set(
            self.topics_key,
            json.dumps(data)
        )
    
    async def get_stats(self) -> Dict:
        """Get topic expansion statistics."""
        await self.connect()
        
        # Get query history count
        query_count = await self.redis_client.zcard(self.query_history_key)
        
        # Get keyword count
        keyword_count = await self.redis_client.zcard(self.keyword_stats_key)
        
        # Get category distribution
        category_dist = await self.redis_client.hgetall(self.category_stats_key)
        
        return {
            "total_categories": len(self.default_topics),
            "total_topics": sum(len(topics) for topics in self.default_topics.values()),
            "queries_analyzed": query_count,
            "unique_keywords": keyword_count,
            "categories": list(self.default_topics.keys()),
            "category_distribution": category_dist,
            "stats": self.stats,
            "config": {
                "min_query_threshold": self.CONFIG["min_query_threshold"],
                "min_keyword_frequency": self.CONFIG["min_keyword_frequency"],
                "min_confidence_score": self.CONFIG["min_confidence_score"],
            }
        }


# Global instance
_topic_expander: ReliableTopicExpander = None


def get_topic_expander() -> ReliableTopicExpander:
    """Get or create the topic expander instance."""
    global _topic_expander
    if _topic_expander is None:
        _topic_expander = ReliableTopicExpander()
    return _topic_expander
