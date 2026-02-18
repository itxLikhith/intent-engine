"""
Intent Engine - Phase 2: Constraint Satisfaction and Result Ranking

This module implements Algorithms 2 and 3 for constraint satisfaction and intent-aligned ranking.
"""

import logging
import re
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import numpy as np

from core.schema import (
    Constraint,
    ConstraintType,
    EthicalDimension,
    Recency,
    SkillLevel,
    TemporalHorizon,
    UniversalIntent,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result candidate with metadata"""

    id: str
    title: str
    description: str
    platform: str | None = None
    provider: str | None = None
    license: str | None = None
    price: float | None = None
    tags: list[str] | None = None
    qualityScore: float | None = 0.5  # Default quality score
    recency: str | None = None  # Publication date or update date
    complexity: str | None = None  # beginner, intermediate, advanced
    compatibility: list[str] | None = None  # Compatible platforms
    privacyRating: float | None = None  # Privacy rating if available
    opensource: bool | None = None  # Whether it's open source


@dataclass
class RankedResult:
    """Represents a ranked result with alignment score and reasons"""

    result: SearchResult
    alignmentScore: float
    matchReasons: list[str]


@dataclass
class RankingRequest:
    """Request object for ranking API"""

    intent: UniversalIntent
    candidates: list[SearchResult]
    options: dict[str, Any] | None = None  # RankingOptions


@dataclass
class RankingResponse:
    """Response object for ranking API"""

    rankedResults: list[RankedResult]


class EmbeddingCache:
    """Cache for embeddings to improve performance with Redis support"""

    def __init__(self, redis_client=None):
        self.cache = {}
        self.model = None
        self.tokenizer = None
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour TTL for Redis cache
        self._load_model()

        # Initialize Redis if not provided
        if self.redis is None:
            self._init_redis()

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
            logger.info(f"EmbeddingCache connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"Redis not available for embeddings, using in-memory only: {e}")
            self.redis = None

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        import hashlib

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
        """Encode text to embedding vector using the sentence transformer model with caching"""
        if self.model is None or self.tokenizer is None:
            # Return random vector for mock implementation
            return np.random.rand(384).astype(np.float32)

        # Check local cache first
        if text in self.cache:
            return self.cache[text]

        # Check Redis cache
        redis_cached = self._get_from_redis(text)
        if redis_cached is not None:
            # Store in local cache for faster access
            self.cache[text] = redis_cached
            return redis_cached

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

        if self.model is None or self.tokenizer is None:
            # Return random vectors for mock implementation
            return [np.random.rand(384).astype(np.float32) for _ in texts]

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
            inputs = self.tokenizer(uncached_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

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
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return float(dot_product / (norm_vec1 * norm_vec2))


class ConstraintSatisfactionEngine:
    """Implements Algorithm 2: satisfiesConstraints()"""

    def __init__(self):
        self.embedding_cache = EmbeddingCache()

    def satisfies_constraints(self, result: SearchResult, constraints: list[Constraint]) -> bool:
        """
        Check if a result satisfies all hard constraints
        Implements Algorithm 2: satisfiesConstraints()
        """
        for constraint in constraints:
            if not constraint.hardFilter:
                continue  # Skip soft constraints

            if not self._satisfies_single_constraint(result, constraint):
                return False

        return True

    def _satisfies_single_constraint(self, result: SearchResult, constraint: Constraint) -> bool:
        """Check if a result satisfies a single constraint"""
        dimension = constraint.dimension
        value = constraint.value
        constraint_type = constraint.type

        if dimension == "platform":
            if constraint_type == ConstraintType.INCLUSION:
                if isinstance(value, str):
                    if result.platform != value:
                        return False
                elif isinstance(value, list) and result.platform not in value:
                    return False
            elif constraint_type == ConstraintType.EXCLUSION:
                if isinstance(value, str):
                    if result.platform == value:
                        return False
                elif isinstance(value, list) and result.platform in value:
                    return False

        elif dimension == "provider":
            if constraint_type == ConstraintType.INCLUSION:
                if isinstance(value, str):
                    if result.provider != value:
                        return False
                elif isinstance(value, list) and result.provider not in value:
                    return False
            elif constraint_type == ConstraintType.EXCLUSION:
                if isinstance(value, str):
                    if result.provider == value:
                        return False
                elif isinstance(value, list) and result.provider in value:
                    return False

        elif dimension == "license":
            if constraint_type == ConstraintType.INCLUSION:
                if isinstance(value, str):
                    if result.license != value:
                        return False
                elif isinstance(value, list) and result.license not in value:
                    return False
            elif constraint_type == ConstraintType.EXCLUSION:
                if isinstance(value, str):
                    if result.license == value:
                        return False
                elif isinstance(value, list) and result.license in value:
                    return False

        elif dimension == "price":
            if isinstance(value, str):
                # Handle price ranges like "<=100" or ">=50"
                match = re.match(r"([<>]=?)(\d+)", value)
                if match:
                    operator, price_limit = match.groups()
                    price_limit = float(price_limit)

                    if constraint_type == ConstraintType.RANGE:
                        # Check price constraints
                        violates = False
                        if operator == "<=" and result.price and result.price > price_limit:
                            violates = True
                        elif operator == ">=" and result.price and result.price < price_limit:
                            violates = True
                        elif operator == "<" and result.price and result.price >= price_limit:
                            violates = True
                        elif operator == ">" and result.price and result.price <= price_limit:
                            violates = True
                        if violates:
                            return False

        elif dimension == "format":
            # Assuming format could be file format or document type
            if constraint_type == ConstraintType.INCLUSION or constraint_type == ConstraintType.EXCLUSION:
                # This would depend on specific implementation
                pass

        elif dimension == "recency":
            # Handle recency constraints
            if constraint_type == ConstraintType.INCLUSION or constraint_type == ConstraintType.EXCLUSION:
                # This would depend on specific implementation
                pass

        return True


class IntentAlignmentEngine:
    """Implements Algorithm 3: computeIntentAlignment()"""

    def __init__(self):
        self.embedding_cache = EmbeddingCache()

    def compute_intent_alignment(self, result: SearchResult, intent: UniversalIntent) -> tuple[float, list[str]]:
        """
        Compute alignment score between a result and user intent
        Implements Algorithm 3: computeIntentAlignment()
        """
        scores = []
        reasons = []

        # 1. Query-content alignment (semantic similarity)
        query_content_score, query_reasons = self._compute_query_content_alignment(result, intent)
        scores.append(query_content_score)
        reasons.extend(query_reasons)

        # 2. Use case alignment (semantic similarity)
        use_case_score, use_case_reasons = self._compute_use_case_alignment(result, intent)
        scores.append(use_case_score)
        reasons.extend(use_case_reasons)

        # 3. Ethical signal alignment
        ethical_score, ethical_reasons = self._compute_ethical_alignment(result, intent)
        scores.append(ethical_score)
        reasons.extend(ethical_reasons)

        # 4. Skill level alignment
        skill_score, skill_reasons = self._compute_skill_alignment(result, intent)
        scores.append(skill_score)
        reasons.extend(skill_reasons)

        # 5. Temporal intent alignment
        temporal_score, temporal_reasons = self._compute_temporal_alignment(result, intent)
        scores.append(temporal_score)
        reasons.extend(temporal_reasons)

        # 6. Quality score (if available)
        quality_score = result.qualityScore or 0.5
        scores.append(quality_score)

        # Weighted combination of scores
        weights = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]  # Sum to 1.0

        # Ensure we have the right number of weights
        if len(scores) != len(weights):
            # Adjust weights if needed
            adjusted_weights = [w for w in weights[: len(scores)]]
            remaining_weight = 1.0 - sum(adjusted_weights)
            if remaining_weight > 0 and len(adjusted_weights) > 0:
                adjusted_weights[0] += remaining_weight
            elif remaining_weight > 0:
                adjusted_weights = [remaining_weight]
            weights = adjusted_weights

        alignment_score = sum(score * weight for score, weight in zip(scores, weights, strict=False))

        # Clamp the score between 0 and 1
        alignment_score = max(0.0, min(1.0, alignment_score))

        return alignment_score, reasons

    def _compute_query_content_alignment(
        self, result: SearchResult, intent: UniversalIntent
    ) -> tuple[float, list[str]]:
        """Compute alignment based on query-content similarity"""
        reasons = []

        if not intent.declared.query:
            return 0.5, reasons  # Neutral score if no query

        # Combine title and description for content
        content = f"{result.title} {result.description}".strip()

        if not content:
            return 0.0, reasons

        # Get embeddings for semantic similarity
        query_embedding = self.embedding_cache.encode_text(intent.declared.query)
        content_embedding = self.embedding_cache.encode_text(content)

        if query_embedding is not None and content_embedding is not None:
            similarity = self.embedding_cache.cosine_similarity(query_embedding, content_embedding)
            # Normalize to 0-1 range
            score = (similarity + 1) / 2  # Cosine similarity is -1 to 1, convert to 0-1

            if score > 0.3:  # Threshold for relevance
                reasons.append("query-content-match")

            return score, reasons
        else:
            # Fallback to keyword matching if embeddings fail
            query_lower = intent.declared.query.lower()
            content_lower = content.lower()

            # Count matching keywords
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            matching_words = query_words.intersection(content_words)

            if len(query_words) > 0:
                score = len(matching_words) / len(query_words)
                if score > 0.1:  # Threshold for relevance
                    reasons.append("keyword-match")
                return score, reasons
            else:
                return 0.5, reasons  # Neutral if no query words

    def _compute_use_case_alignment(self, result: SearchResult, intent: UniversalIntent) -> tuple[float, list[str]]:
        """Compute alignment based on use case similarity"""
        reasons = []

        if not intent.inferred.useCases:
            return 0.5, reasons  # Neutral score if no use cases

        if not result.tags:
            return 0.0, reasons  # No tags to match against

        score = 0.0
        tag_text = " ".join(result.tags).lower()

        for use_case in intent.inferred.useCases:
            use_case_str = use_case.value.replace("_", " ")

            # Check if use case is mentioned in tags
            if use_case_str in tag_text:
                score += 0.5  # Significant boost for direct match
                reasons.append(f"use-case-{use_case.value}-match")
            else:
                # Use embedding similarity as fallback
                use_case_embedding = self.embedding_cache.encode_text(use_case_str)
                tag_embedding = self.embedding_cache.encode_text(tag_text)

                if use_case_embedding is not None and tag_embedding is not None:
                    similarity = self.embedding_cache.cosine_similarity(use_case_embedding, tag_embedding)
                    # Normalize to 0-1 range and add to score
                    similarity_score = (similarity + 1) / 2
                    score += similarity_score * 0.3  # Smaller weight for semantic match

        # Normalize score to 0-1 range
        if intent.inferred.useCases:
            score = min(1.0, score / len(intent.inferred.useCases))

        return score, reasons

    def _compute_ethical_alignment(self, result: SearchResult, intent: UniversalIntent) -> tuple[float, list[str]]:
        """Compute alignment based on ethical signals"""
        reasons = []

        if not intent.inferred.ethicalSignals:
            return 0.5, reasons  # Neutral score if no ethical signals

        score = 0.0
        ethical_matches = 0

        for signal in intent.inferred.ethicalSignals:
            if signal.dimension == EthicalDimension.PRIVACY:
                if result.privacyRating and result.privacyRating > 0.7:
                    score += 0.5
                    ethical_matches += 1
                    reasons.append("privacy-aligned")
            elif signal.dimension == EthicalDimension.OPENNESS:
                if signal.preference == "open-source_preferred" and result.opensource:
                    score += 0.5
                    ethical_matches += 1
                    reasons.append("open-source-aligned")
            elif signal.dimension == EthicalDimension.SUSTAINABILITY:
                # Placeholder for sustainability checks
                pass
            elif signal.dimension == EthicalDimension.ETHICS:
                # Placeholder for ethics checks
                pass

        # Normalize score based on number of ethical signals
        if intent.inferred.ethicalSignals:
            score = min(1.0, score / len(intent.inferred.ethicalSignals))

        return score, reasons

    def _compute_skill_alignment(self, result: SearchResult, intent: UniversalIntent) -> tuple[float, list[str]]:
        """Compute alignment based on skill level"""
        reasons = []

        declared_skill = intent.declared.skillLevel
        result_complexity = result.complexity

        if not result_complexity:
            return 0.5, reasons  # Neutral if no complexity info

        # Map result complexity to skill levels
        complexity_to_skill = {
            "beginner": SkillLevel.BEGINNER,
            "intermediate": SkillLevel.INTERMEDIATE,
            "advanced": SkillLevel.ADVANCED,
            "expert": SkillLevel.EXPERT,
        }

        if result_complexity in complexity_to_skill:
            result_skill = complexity_to_skill[result_complexity]

            if declared_skill == result_skill:
                score = 1.0
                reasons.append(f"skill-level-match-{result_complexity}")
            elif (
                (
                    declared_skill == SkillLevel.BEGINNER
                    and result_skill in [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE]
                )
                or (
                    declared_skill == SkillLevel.INTERMEDIATE
                    and result_skill in [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED]
                )
                or (
                    declared_skill == SkillLevel.ADVANCED
                    and result_skill in [SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED, SkillLevel.EXPERT]
                )
            ):
                # Allow some flexibility in skill matching
                score = 0.8
                reasons.append(f"skill-level-compatible-{result_complexity}")
            else:
                score = 0.2
                reasons.append(f"skill-level-mismatch-{result_complexity}")
        else:
            score = 0.5  # Neutral if complexity not recognized

        return score, reasons

    def _compute_temporal_alignment(self, result: SearchResult, intent: UniversalIntent) -> tuple[float, list[str]]:
        """Compute alignment based on temporal intent"""
        reasons = []

        temporal_intent = intent.inferred.temporalIntent
        if not temporal_intent:
            return 0.5, reasons  # Neutral if no temporal intent

        score = 0.5  # Base score

        # Check recency alignment
        if temporal_intent.recency == Recency.RECENT and result.recency:
            # If user wants recent content, check if result is recent
            from datetime import datetime

            try:
                # Assume result.recency is in ISO format or similar
                if result.recency.endswith("Z"):
                    result_date = datetime.fromisoformat(result.recency.replace("Z", "+00:00"))
                elif "+" in result.recency or result.recency.count("-") > 2:  # Has timezone info
                    result_date = datetime.fromisoformat(result.recency)
                else:
                    # Naive datetime, treat as UTC
                    result_date = datetime.fromisoformat(result.recency).replace(tzinfo=UTC)

                # Make sure both datetimes have timezone info
                now = datetime.now(UTC)
                days_old = (now - result_date).days

                if days_old <= 30:  # Recent if within 30 days
                    score += 0.2
                    reasons.append("recent-content")
                elif days_old <= 180:  # Somewhat recent
                    score += 0.1
            except ValueError:
                # If date parsing fails, skip this check
                pass

        # Check horizon alignment (today, week, month, etc.)
        if temporal_intent.horizon == TemporalHorizon.TODAY and result.recency:
            # If user wants today's content, check if result is very recent
            from datetime import datetime

            try:
                # Same date parsing logic as above
                if result.recency.endswith("Z"):
                    result_date = datetime.fromisoformat(result.recency.replace("Z", "+00:00"))
                elif "+" in result.recency or result.recency.count("-") > 2:  # Has timezone info
                    result_date = datetime.fromisoformat(result.recency)
                else:
                    # Naive datetime, treat as UTC
                    result_date = datetime.fromisoformat(result.recency).replace(tzinfo=UTC)

                # Make sure both datetimes have timezone info
                now = datetime.now(UTC)
                days_old = (now - result_date).days

                if days_old <= 1:  # Within 1 day
                    score += 0.2
                    reasons.append("today-relevant")
            except ValueError:
                # If date parsing fails, skip this check
                pass

        # Clamp score between 0 and 1
        score = max(0.0, min(1.0, score))

        return score, reasons


class IntentRanker:
    """Main class that orchestrates constraint satisfaction and ranking"""

    def __init__(self):
        self.constraint_satisfaction_engine = ConstraintSatisfactionEngine()
        self.intent_alignment_engine = IntentAlignmentEngine()

    def rank_results(self, request: RankingRequest) -> RankingResponse:
        """
        Main function to rank results based on intent alignment
        Implements the complete ranking algorithm
        """
        # Step 1: Apply hard filters (constraint satisfaction)
        filtered_candidates = []
        for candidate in request.candidates:
            if self.constraint_satisfaction_engine.satisfies_constraints(
                candidate, request.intent.declared.constraints
            ):
                filtered_candidates.append(candidate)

        # Step 2: Compute intent alignment for each candidate
        ranked_results = []
        for candidate in filtered_candidates:
            alignment_score, match_reasons = self.intent_alignment_engine.compute_intent_alignment(
                candidate, request.intent
            )

            ranked_result = RankedResult(result=candidate, alignmentScore=alignment_score, matchReasons=match_reasons)
            ranked_results.append(ranked_result)

        # Step 3: Sort by alignment score (descending)
        ranked_results.sort(key=lambda x: x.alignmentScore, reverse=True)

        return RankingResponse(rankedResults=ranked_results)


# Global instance for caching
_intent_ranker_instance = None


def get_intent_ranker() -> IntentRanker:
    """
    Get singleton instance of IntentRanker with cached model
    """
    global _intent_ranker_instance
    if _intent_ranker_instance is None:
        _intent_ranker_instance = IntentRanker()
    return _intent_ranker_instance


def rank_results(request: RankingRequest) -> RankingResponse:
    """
    Main entry point for ranking results based on intent
    """
    ranker = get_intent_ranker()
    return ranker.rank_results(request)
