"""
Intent Engine - Phase 3: Service Recommendation and Cross-Product Routing

This module implements service recommendation logic based on user intent.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.embedding_service import get_embedding_service
from core.schema import Frequency, Recency, TemporalHorizon, UniversalIntent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServiceMetadata:
    """Metadata for a service in the workspace ecosystem"""

    id: str
    name: str
    supportedGoals: list[str]  # List of supported IntentGoal values
    primaryUseCases: list[str]  # List of primary use cases
    temporalPatterns: list[str]  # List of temporal patterns
    ethicalAlignment: list[str]  # List of ethical alignments
    description: str | None = None  # Optional description


@dataclass
class ServiceRecommendation:
    """Represents a recommended service with score and reasons"""

    service: ServiceMetadata
    serviceScore: float
    matchReasons: list[str]


@dataclass
class ServiceRecommendationRequest:
    """Request object for service recommendation API"""

    intent: UniversalIntent
    availableServices: list[ServiceMetadata]
    options: dict[str, Any] | None = None  # RecommendationOptions


@dataclass
class ServiceRecommendationResponse:
    """Response object for service recommendation API"""

    recommendations: list[ServiceRecommendation]


class EmbeddingCache:
    """Cache for embeddings to improve performance - uses shared EmbeddingService"""

    def __init__(self):
        # Use the shared embedding service instead of loading models independently
        self._service = get_embedding_service()
        self._service.initialize()

    def encode_text(self, text: str) -> np.ndarray | None:
        """Encode text to embedding vector using the shared service"""
        return self._service.encode_text(text)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return self._service.cosine_similarity(vec1, vec2)


class ServiceScoringEngine:
    """Engine for scoring services based on user intent"""

    def __init__(self):
        self.embedding_cache = EmbeddingCache()

    def compute_service_match(
        self, intent: UniversalIntent, service: ServiceMetadata
    ) -> tuple[float, list[str]]:
        """
        Compute match score between intent and service
        """
        scores = []
        reasons = []

        # 1. Goal matching
        goal_score, goal_reasons = self._compute_goal_matching(intent, service)
        scores.append(goal_score)
        reasons.extend(goal_reasons)

        # 2. Use case matching (semantic similarity)
        use_case_score, use_case_reasons = self._compute_use_case_matching(
            intent, service
        )
        scores.append(use_case_score)
        reasons.extend(use_case_reasons)

        # 3. Temporal pattern matching
        temporal_score, temporal_reasons = self._compute_temporal_matching(
            intent, service
        )
        scores.append(temporal_score)
        reasons.extend(temporal_reasons)

        # 4. Ethical alignment matching
        ethical_score, ethical_reasons = self._compute_ethical_matching(intent, service)
        scores.append(ethical_score)
        reasons.extend(ethical_reasons)

        # Weighted combination of scores
        weights = [0.40, 0.30, 0.15, 0.15]  # Sum to 1.0

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

        service_score = sum(
            score * weight for score, weight in zip(scores, weights, strict=False)
        )

        # Clamp the score between 0 and 1
        service_score = max(0.0, min(1.0, service_score))

        return service_score, reasons

    def _compute_goal_matching(
        self, intent: UniversalIntent, service: ServiceMetadata
    ) -> tuple[float, list[str]]:
        """Compute score based on goal matching"""
        reasons = []

        if not intent.declared.goal:
            return 0.5, reasons  # Neutral score if no goal

        goal_value = intent.declared.goal.value

        if goal_value in service.supportedGoals:
            score = 1.0
            reasons.append(f"goal={goal_value} supported")
            return score, reasons
        else:
            # Check for semantic similarity with supported goals
            goal_embedding = self.embedding_cache.encode_text(
                goal_value.replace("_", " ")
            )
            service_goals_text = " ".join(service.supportedGoals).replace("_", " ")
            service_goals_embedding = self.embedding_cache.encode_text(
                service_goals_text
            )

            if goal_embedding is not None and service_goals_embedding is not None:
                similarity = self.embedding_cache.cosine_similarity(
                    goal_embedding, service_goals_embedding
                )
                # Normalize to 0-1 range
                score = (similarity + 1) / 2
                if score > 0.3:  # Threshold for relevance
                    reasons.append(f"goal-semantically-matched ({goal_value})")
                return score, reasons
            else:
                # Fallback: partial match based on keywords
                goal_lower = goal_value.lower()
                service_goals_lower = [g.lower() for g in service.supportedGoals]

                for service_goal in service_goals_lower:
                    if goal_lower in service_goal or service_goal in goal_lower:
                        score = 0.7
                        reasons.append(f"goal-partially-matched ({goal_value})")
                        return score, reasons

                return 0.1, reasons  # Low score if no match

    def _compute_use_case_matching(
        self, intent: UniversalIntent, service: ServiceMetadata
    ) -> tuple[float, list[str]]:
        """Compute score based on use case matching"""
        reasons = []

        if not intent.inferred.useCases:
            return 0.5, reasons  # Neutral score if no use cases

        if not service.primaryUseCases:
            return 0.0, reasons  # No use cases to match against

        total_score = 0.0
        matches_found = 0

        for use_case in intent.inferred.useCases:
            use_case_str = use_case.value.replace("_", " ")

            # Check for direct matches
            direct_match = False
            for service_use_case in service.primaryUseCases:
                if (
                    service_use_case.lower() in use_case_str.lower()
                    or use_case_str.lower() in service_use_case.lower()
                ):
                    total_score += 1.0
                    matches_found += 1
                    reasons.append(f"use case '{service_use_case}' matched")
                    direct_match = True
                    break

            if not direct_match:
                # Use semantic similarity as fallback
                use_case_embedding = self.embedding_cache.encode_text(use_case_str)
                service_use_cases_embedding = self.embedding_cache.encode_text(
                    " ".join(service.primaryUseCases)
                )

                if (
                    use_case_embedding is not None
                    and service_use_cases_embedding is not None
                ):
                    similarity = self.embedding_cache.cosine_similarity(
                        use_case_embedding, service_use_cases_embedding
                    )
                    # Normalize to 0-1 range
                    similarity_score = (similarity + 1) / 2
                    total_score += similarity_score
                    matches_found += 1
                    if similarity_score > 0.3:
                        reasons.append(
                            f"use case semantically matched '{use_case_str}'"
                        )

        # Normalize score based on number of use cases
        if matches_found > 0:
            avg_score = total_score / len(intent.inferred.useCases)
            return min(1.0, avg_score), reasons
        else:
            return 0.0, reasons

    def _compute_temporal_matching(
        self, intent: UniversalIntent, service: ServiceMetadata
    ) -> tuple[float, list[str]]:
        """Compute score based on temporal pattern matching"""
        reasons = []

        temporal_intent = intent.inferred.temporalIntent
        if not temporal_intent:
            return 0.5, reasons  # Neutral score if no temporal intent

        score = 0.5  # Base score

        # Check horizon matching
        if temporal_intent.horizon == TemporalHorizon.TODAY:
            if (
                "short_session" in service.temporalPatterns
                or "quick_access" in service.temporalPatterns
            ):
                score += 0.2
                reasons.append("temporal-horizon-today-matched")
        elif temporal_intent.horizon in [
            TemporalHorizon.WEEK,
            TemporalHorizon.LONGTERM,
        ]:
            if (
                "long_session" in service.temporalPatterns
                or "extended_work" in service.temporalPatterns
            ):
                score += 0.2
                reasons.append("temporal-horizon-longterm-matched")

        # Check frequency matching
        if temporal_intent.frequency == Frequency.RECURRING:
            if (
                "recurring_edit" in service.temporalPatterns
                or "regular_updates" in service.temporalPatterns
            ):
                score += 0.15
                reasons.append("temporal-frequency-recurring-matched")
        elif temporal_intent.frequency == Frequency.ONEOFF:
            if (
                "one_time_task" in service.temporalPatterns
                or "quick_completion" in service.temporalPatterns
            ):
                score += 0.15
                reasons.append("temporal-frequency-oneoff-matched")

        # Check recency matching
        if temporal_intent.recency == Recency.RECENT:
            if (
                "real_time" in service.temporalPatterns
                or "live_updates" in service.temporalPatterns
            ):
                score += 0.1
                reasons.append("temporal-recency-recent-matched")

        # Clamp score between 0 and 1
        score = max(0.0, min(1.0, score))

        return score, reasons

    def _compute_ethical_matching(
        self, intent: UniversalIntent, service: ServiceMetadata
    ) -> tuple[float, list[str]]:
        """Compute score based on ethical alignment matching"""
        reasons = []

        if not intent.inferred.ethicalSignals:
            return 0.5, reasons  # Neutral score if no ethical signals

        score = 0.0
        matches_found = 0

        for signal in intent.inferred.ethicalSignals:
            # Check dimension and preference alignment
            signal_dimension = signal.dimension.value
            signal_preference = signal.preference

            # Look for matches in service ethical alignment
            for eth_align in service.ethicalAlignment:
                if (
                    signal_dimension.lower() in eth_align.lower()
                    or signal_preference.lower() in eth_align.lower()
                ):
                    score += 0.5
                    matches_found += 1
                    reasons.append(f"ethical alignment: {eth_align}")
                    break

        # Normalize score based on number of ethical signals
        if matches_found > 0:
            avg_score = score / len(intent.inferred.ethicalSignals)
            return min(1.0, avg_score), reasons
        else:
            return 0.0, reasons


class ServiceRecommender:
    """Main class that orchestrates service recommendation"""

    def __init__(self):
        self.scoring_engine = ServiceScoringEngine()

    def recommend_services(
        self, request: ServiceRecommendationRequest
    ) -> ServiceRecommendationResponse:
        """
        Main function to recommend services based on user intent
        """
        recommendations = []

        for service in request.availableServices:
            service_score, match_reasons = self.scoring_engine.compute_service_match(
                request.intent, service
            )

            # Only include recommendations above a minimum threshold
            if service_score > 0.1:  # Minimum relevance threshold
                recommendation = ServiceRecommendation(
                    service=service,
                    serviceScore=service_score,
                    matchReasons=match_reasons,
                )
                recommendations.append(recommendation)

        # Sort by service score (descending)
        recommendations.sort(key=lambda x: x.serviceScore, reverse=True)

        return ServiceRecommendationResponse(recommendations=recommendations)


# Global instance for caching
_service_recommender_instance = None
_recommender_lock = threading.Lock()


def get_service_recommender() -> ServiceRecommender:
    """
    Get thread-safe singleton instance of ServiceRecommender with cached model.

    Uses double-checked locking pattern for thread safety.
    """
    global _service_recommender_instance
    if _service_recommender_instance is None:
        with _recommender_lock:
            if _service_recommender_instance is None:
                _service_recommender_instance = ServiceRecommender()
    return _service_recommender_instance


def recommend_services(
    request: ServiceRecommendationRequest,
) -> ServiceRecommendationResponse:
    """
    Main entry point for recommending services based on intent
    """
    recommender = get_service_recommender()
    return recommender.recommend_services(request)
