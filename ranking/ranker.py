"""
Intent Engine - Phase 2: Constraint Satisfaction and Result Ranking

This module implements Algorithms 2 and 3 for constraint satisfaction and intent-aligned ranking.
"""

import logging
import re
import threading
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import numpy as np

from core.embedding_service import get_embedding_service
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
    """Cache for embeddings to improve performance - uses shared EmbeddingService"""

    def __init__(self, redis_client=None):
        # Use the shared embedding service instead of loading models independently
        self._service = get_embedding_service()
        # Initialize if needed
        self._service.initialize(use_redis=(redis_client is not None))

    def encode_text(self, text: str) -> np.ndarray | None:
        """Encode text to embedding vector using the shared service"""
        return self._service.encode_text(text)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray | None]:
        """Encode batch of texts using the shared service"""
        return self._service.encode_batch(texts)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return self._service.cosine_similarity(vec1, vec2)


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

        elif dimension == "tags":
            # Handle tags constraints
            if constraint_type == ConstraintType.INCLUSION:
                if isinstance(value, str):
                    # Check if the value is in result tags
                    if result.tags is None or value not in result.tags:
                        return False
                elif isinstance(value, list):
                    # Check if any of the values are in result tags
                    if result.tags is None or not any(v in result.tags for v in value):
                        return False
            elif constraint_type == ConstraintType.EXCLUSION:
                if isinstance(value, str):
                    # Check if the value is NOT in result tags
                    if result.tags is not None and value in result.tags:
                        return False
                elif isinstance(value, list):
                    # Check if none of the values are in result tags
                    if result.tags is not None and any(v in result.tags for v in value):
                        return False

        elif dimension == "format":
            # Handle format constraints (file format, document type, content format)
            # Assumes result has a 'format' attribute or it can be inferred from tags/description
            result_format = getattr(result, "format", None)

            # Try to infer format from tags if not explicitly set
            if result_format is None and result.tags:
                format_tags = [
                    t for t in result.tags if t.lower() in ["pdf", "doc", "video", "audio", "interactive", "text"]
                ]
                if format_tags:
                    result_format = format_tags[0].lower()

            if result_format is None:
                # If no format info available, inclusion fails, exclusion passes
                if constraint_type == ConstraintType.INCLUSION:
                    return False
                else:  # EXCLUSION
                    return True

            result_format = result_format.lower()

            if constraint_type == ConstraintType.INCLUSION:
                if isinstance(value, str):
                    if value.lower() != result_format:
                        return False
                elif isinstance(value, list):
                    if not any(v.lower() == result_format for v in value):
                        return False
            elif constraint_type == ConstraintType.EXCLUSION:
                if isinstance(value, str):
                    if value.lower() == result_format:
                        return False
                elif isinstance(value, list):
                    if any(v.lower() == result_format for v in value):
                        return False

        elif dimension == "recency":
            # Handle recency constraints (content freshness)
            # Assumes result.recency is in ISO format or similar
            if not result.recency:
                # If no recency info available, inclusion fails, exclusion passes
                if constraint_type == ConstraintType.INCLUSION:
                    return False
                else:  # EXCLUSION
                    return True

            from datetime import datetime

            try:
                # Parse result recency
                recency_str = result.recency
                if recency_str.endswith("Z"):
                    result_date = datetime.fromisoformat(recency_str.replace("Z", "+00:00"))
                elif "+" in recency_str or recency_str.count("-") > 2:  # Has timezone info
                    result_date = datetime.fromisoformat(recency_str)
                else:
                    # Naive datetime, treat as UTC
                    result_date = datetime.fromisoformat(recency_str).replace(tzinfo=UTC)

                now = datetime.now(UTC)
                days_old = (now - result_date).days

                # Handle recency constraint values
                if isinstance(value, str):
                    value_lower = value.lower()

                    # Parse recency constraints like "last_7_days", "last_30_days", "this_year", etc.
                    if value_lower.startswith("last_") and value_lower.endswith("_days"):
                        try:
                            days_limit = int(value_lower.split("_")[1])
                            if days_old > days_limit:
                                return False
                        except (ValueError, IndexError):
                            pass
                    elif value_lower == "today" and days_old > 1:
                        return False
                    elif value_lower == "this_week" and days_old > 7:
                        return False
                    elif value_lower == "this_month" and days_old > 30:
                        return False
                    elif value_lower == "this_year":
                        current_year = now.year
                        result_year = result_date.year
                        if result_year != current_year:
                            return False
                    elif value_lower == "evergreen":
                        # Evergreen content is old but still relevant - no filtering
                        pass

                elif isinstance(value, dict):
                    # Handle structured recency constraints
                    if "max_days_old" in value:
                        max_days = value["max_days_old"]
                        if days_old > max_days:
                            return False
                    if "min_days_old" in value:
                        min_days = value["min_days_old"]
                        if days_old < min_days:
                            return False

            except (ValueError, AttributeError):
                # If date parsing fails, skip this check (fail open)
                if constraint_type == ConstraintType.INCLUSION:
                    return False

            # Handle exclusion constraints for recency
            if constraint_type == ConstraintType.EXCLUSION and isinstance(value, str):
                value_lower = value.lower()
                # For exclusion, filter out if matches
                if value_lower == "today" and days_old <= 1:
                    return False
                elif value_lower == "this_week" and days_old <= 7:
                    return False
                elif value_lower == "this_month" and days_old <= 30:
                    return False

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
                    and result_skill
                    in [
                        SkillLevel.BEGINNER,
                        SkillLevel.INTERMEDIATE,
                        SkillLevel.ADVANCED,
                    ]
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

            ranked_result = RankedResult(
                result=candidate,
                alignmentScore=alignment_score,
                matchReasons=match_reasons,
            )
            ranked_results.append(ranked_result)

        # Step 3: Sort by alignment score (descending)
        ranked_results.sort(key=lambda x: x.alignmentScore, reverse=True)

        return RankingResponse(rankedResults=ranked_results)


# Global instance for caching
_intent_ranker_instance = None
_ranker_lock = threading.Lock()


def get_intent_ranker() -> IntentRanker:
    """
    Get thread-safe singleton instance of IntentRanker with cached model.

    Uses double-checked locking pattern for thread safety.
    """
    global _intent_ranker_instance
    if _intent_ranker_instance is None:
        with _ranker_lock:
            if _intent_ranker_instance is None:
                _intent_ranker_instance = IntentRanker()
    return _intent_ranker_instance


def rank_results(request: RankingRequest) -> RankingResponse:
    """
    Main entry point for ranking results based on intent
    """
    ranker = get_intent_ranker()
    return ranker.rank_results(request)
