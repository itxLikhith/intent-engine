"""
Intent Engine - Phase 4: Privacy-First Ad Matching

This module implements ethical ad matching based on user intent without tracking.
"""

import logging
from dataclasses import dataclass
from datetime import timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.schema import (
    Constraint,
    ConstraintType,
    DeclaredIntent,
    EthicalDimension,
    EthicalSignal,
    InferredIntent,
    IntentGoal,
    UniversalIntent,
    UseCase,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AdMetadata:
    """Metadata for an advertisement"""

    id: str
    title: str
    description: str
    targetingConstraints: Dict[str, List[str]]  # Platform, provider, license, etc.
    forbiddenDimensions: List[str]  # Must be empty or only safe fields
    qualityScore: float  # Quality score of the ad
    ethicalTags: List[str]  # Privacy, open_source, no_tracking, etc.
    advertiser: Optional[str] = None  # Name of advertiser
    category: Optional[str] = None  # Category of the ad
    creative_format: Optional[str] = None  # Creative format of the ad (banner, native, video, etc.)


@dataclass
class MatchedAd:
    """Represents a matched ad with relevance score and reasons"""

    ad: AdMetadata
    adRelevanceScore: float
    matchReasons: List[str]


@dataclass
class AdMatchingRequest:
    """Request object for ad matching API"""

    intent: UniversalIntent
    adInventory: List[AdMetadata]
    config: Optional[Dict[str, Any]] = None  # MatchingConfig


@dataclass
class AdMatchingResponse:
    """Response object for ad matching API"""

    matchedAds: List[MatchedAd]
    metrics: Dict[str, int]


class EmbeddingCache:
    """Cache for embeddings to improve performance"""

    def __init__(self):
        self.cache = {}
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            import torch
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

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding vector using the sentence transformer model"""
        if self.model is None or self.tokenizer is None:
            # Return random vector for mock implementation
            return np.random.rand(384).astype(np.float32)

        # Check cache first
        if text in self.cache:
            return self.cache[text]

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

            # Cache the result
            self.cache[text] = result

            return result
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return float(dot_product / (norm_vec1 * norm_vec2))


class AdFairnessChecker:
    """Checks if ads comply with fairness and privacy rules"""

    def __init__(self):
        # Define forbidden dimensions that violate privacy
        self.forbidden_dimensions = {
            "age",
            "gender",
            "income",
            "race",
            "ethnicity",
            "political_affiliation",
            "religious_belief",
            "health_condition",
            "sexual_orientation",
            "location",
            "behavior",
            "interests",
            "purchasing_history",
            "browsing_history",
            "device_fingerprint",
            "ip_address",
        }

    def validate_advertiser_constraints(self, ad: AdMetadata) -> Tuple[bool, str]:
        """
        Validate that an ad doesn't use forbidden targeting dimensions
        Returns (isValid, reason)
        """
        for forbidden_dim in ad.forbiddenDimensions:
            if forbidden_dim.lower() in self.forbidden_dimensions:
                return False, f"Forbidden dimension '{forbidden_dim}' used for targeting"

        return True, "Valid"


class AdConstraintMatcher:
    """Matches ads against user constraints"""

    def __init__(self):
        pass

    def satisfies_user_constraints(self, ad: AdMetadata, constraints: List[Constraint]) -> Tuple[bool, List[str]]:
        """
        Check if an ad satisfies user constraints
        Returns (satisfies, list_of_reasons)
        """
        reasons = []

        for constraint in constraints:
            if not constraint.hardFilter:
                continue  # Skip soft constraints

            dimension = constraint.dimension
            value = constraint.value
            constraint_type = constraint.type

            # Check if ad has targeting constraints for this dimension
            if dimension in ad.targetingConstraints:
                ad_values = ad.targetingConstraints[dimension]

                if constraint_type == ConstraintType.INCLUSION:
                    # Ad must include at least one of the allowed values
                    if isinstance(value, str):
                        if value not in ad_values:
                            return False, [f"Ad does not target required {dimension}: {value}"]
                        else:
                            reasons.append(f"{dimension}={value} satisfied")
                    elif isinstance(value, list):
                        # At least one value from the constraint must be in ad's targets
                        if not any(v in ad_values for v in value):
                            return False, [f"Ad does not target any of required {dimension}s: {value}"]
                        else:
                            reasons.append(f"{dimension} satisfied: {value}")

                elif constraint_type == ConstraintType.EXCLUSION:
                    # Ad must NOT target any of the excluded values
                    if isinstance(value, str):
                        if value in ad_values:
                            return False, [f"Ad targets excluded {dimension}: {value}"]
                    elif isinstance(value, list):
                        if any(v in ad_values for v in value):
                            return False, [f"Ad targets excluded {dimension}s: {value}"]
                    reasons.append(f"Excluded {dimension} constraint satisfied")
            else:
                # If ad doesn't specify targeting for this dimension,
                # inclusion constraints fail, exclusion constraints pass
                if constraint_type == ConstraintType.INCLUSION:
                    return False, [f"Ad does not specify targeting for required dimension: {dimension}"]
                elif constraint_type == ConstraintType.EXCLUSION:
                    reasons.append(f"Excluded {dimension} constraint satisfied (not targeted)")

        return True, reasons


class AdRelevanceScorer:
    """Scores ad relevance based on user intent"""

    def __init__(self):
        self.embedding_cache = EmbeddingCache()

    def compute_ad_relevance(self, ad: AdMetadata, intent: UniversalIntent) -> Tuple[float, List[str]]:
        """
        Compute relevance score between ad and user intent
        Returns (relevance_score, list_of_reasons)
        """
        scores = []
        reasons = []

        # FIX: Add null safety for intent.declared and intent.inferred
        declared = intent.declared if intent.declared else DeclaredIntent()
        inferred = intent.inferred if intent.inferred else InferredIntent()

        # 1. Semantic similarity between ad and query
        semantic_score, semantic_reasons = self._compute_semantic_similarity(ad, intent, declared)
        scores.append(semantic_score)
        reasons.extend(semantic_reasons)

        # 2. Ethical signal alignment
        ethical_score, ethical_reasons = self._compute_ethical_alignment(ad, intent, inferred)
        scores.append(ethical_score)
        reasons.extend(ethical_reasons)

        # 3. Goal alignment
        goal_score, goal_reasons = self._compute_goal_alignment(ad, intent, declared)
        scores.append(goal_score)
        reasons.extend(goal_reasons)

        # 4. Quality score factor
        scores.append(ad.qualityScore)

        # Weighted combination of scores
        weights = [0.40, 0.30, 0.20, 0.10]  # Sum to 1.0

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

        relevance_score = sum(score * weight for score, weight in zip(scores, weights))

        # Clamp the score between 0 and 1
        relevance_score = max(0.0, min(1.0, relevance_score))

        return relevance_score, reasons

    def _compute_semantic_similarity(
        self, ad: AdMetadata, intent: UniversalIntent, declared: DeclaredIntent
    ) -> Tuple[float, List[str]]:
        """Compute semantic similarity between ad and user query"""
        reasons = []

        # FIX: Add null safety for declared.query
        query = declared.query if declared else None
        if not query:
            return 0.5, reasons

        # Combine ad title and description for content
        ad_content = f"{ad.title} {ad.description}".strip()

        if not ad_content:
            return 0.0, reasons

        # Get embeddings for semantic similarity
        query_embedding = self.embedding_cache.encode_text(query)
        ad_embedding = self.embedding_cache.encode_text(ad_content)

        if query_embedding is not None and ad_embedding is not None:
            similarity = self.embedding_cache.cosine_similarity(query_embedding, ad_embedding)
            # Normalize to 0-1 range
            score = (similarity + 1) / 2  # Cosine similarity is -1 to 1, convert to 0-1

            if score > 0.3:  # Threshold for relevance
                reasons.append(f"query semantic match: '{query[:30]}...'")

            return score, reasons
        else:
            # Fallback to keyword matching if embeddings fail
            query_lower = query.lower()
            ad_lower = ad_content.lower()

            # Count matching keywords
            query_words = set(query_lower.split())
            ad_words = set(ad_lower.split())
            matching_words = query_words.intersection(ad_words)

            if len(query_words) > 0:
                score = len(matching_words) / len(query_words)
                if score > 0.1:  # Threshold for relevance
                    reasons.append("keyword match")
                return score, reasons
            else:
                return 0.5, reasons  # Neutral if no query words

    def _compute_ethical_alignment(
        self, ad: AdMetadata, intent: UniversalIntent, inferred: InferredIntent
    ) -> Tuple[float, List[str]]:
        """Compute alignment based on ethical signals"""
        reasons = []

        # FIX: Add null safety for inferred.ethicalSignals
        ethical_signals = inferred.ethicalSignals if inferred else []
        if not ethical_signals:
            return 0.5, reasons

        score = 0.0
        matches_found = 0

        for signal in ethical_signals:
            signal_dimension = signal.dimension.value
            signal_preference = signal.preference

            # Look for matches in ad ethical tags
            for eth_tag in ad.ethicalTags:
                if signal_dimension.lower() in eth_tag.lower() or signal_preference.lower() in eth_tag.lower():
                    score += 0.5
                    matches_found += 1
                    reasons.append(f"ethical alignment: {eth_tag}")
                    break

        # Normalize score based on number of ethical signals
        if matches_found > 0 and ethical_signals:
            avg_score = score / len(ethical_signals)
            return min(1.0, avg_score), reasons
        else:
            return 0.0, reasons

    def _compute_goal_alignment(
        self, ad: AdMetadata, intent: UniversalIntent, declared: DeclaredIntent
    ) -> Tuple[float, List[str]]:
        """Compute alignment based on user goal"""
        reasons = []

        # FIX: Add null safety for declared.goal
        goal = declared.goal if declared else None
        if not goal:
            return 0.5, reasons

        goal_value = goal.value.upper()

        # Simple heuristic: check if ad content relates to common goals
        ad_content = f"{ad.title} {ad.description}".lower()

        goal_indicators = {
            "LEARN": ["guide", "tutorial", "how to", "learn", "explain", "setup", "configure"],
            "PURCHASE": ["buy", "purchase", "price", "cost", "deal", "offer", "sale"],
            "COMPARISON": ["compare", "versus", "vs", "difference", "alternative", "best"],
            "TROUBLESHOOTING": ["fix", "solve", "problem", "issue", "troubleshoot", "error"],
        }

        if goal_value in goal_indicators:
            indicators = goal_indicators[goal_value]
            matches = [indicator for indicator in indicators if indicator in ad_content]
            if matches:
                score = min(1.0, len(matches) * 0.3)  # Boost for each match
                reasons.append(f"goal alignment: {goal_value.lower()} related terms found")
                return score, reasons

        # Fallback: use embedding similarity between goal and ad
        goal_embedding = self.embedding_cache.encode_text(goal_value.replace("_", " "))
        ad_embedding = self.embedding_cache.encode_text(ad_content)

        if goal_embedding is not None and ad_embedding is not None:
            similarity = self.embedding_cache.cosine_similarity(goal_embedding, ad_embedding)
            # Normalize to 0-1 range
            score = (similarity + 1) / 2
            if score > 0.2:
                reasons.append(f"goal semantically aligned: {goal_value.lower()}")
            return score, reasons

        return 0.2, reasons  # Low default if no clear alignment


class AdMatcher:
    """Main class that orchestrates ad matching"""

    def __init__(self):
        self.fairness_checker = AdFairnessChecker()
        self.constraint_matcher = AdConstraintMatcher()
        self.relevance_scorer = AdRelevanceScorer()

    def match_ads(self, request: AdMatchingRequest) -> AdMatchingResponse:
        """
        Main function to match ads to user intent
        Implements the ad matching algorithm from the technical reference
        """
        matched = []
        total_ads_evaluated = len(request.adInventory)
        ads_passed_fairness = 0
        ads_passed_constraints = 0
        ads_passed_relevance = 0

        # FIX: Initialize min_threshold before the loop
        min_threshold = request.config.get("minThreshold", 0.3) if request.config else 0.3

        # Log intent summary for debugging
        declared = request.intent.declared if request.intent.declared else DeclaredIntent()
        logger.info(
            f"Ad matching started: {total_ads_evaluated} ads, query='{declared.query[:50] if declared.query else 'N/A'}...'"
        )

        for idx, ad in enumerate(request.adInventory):
            # Filter 1: Fairness check (NO discriminatory targeting)
            is_valid, fairness_reason = self.fairness_checker.validate_advertiser_constraints(ad)
            if not is_valid:
                logger.debug(f"Ad {idx} ({ad.id}) rejected at fairness check: {fairness_reason}")
                continue
            ads_passed_fairness += 1

            # Filter 2: User constraints
            # FIX: Add null safety for intent.declared.constraints
            declared = request.intent.declared if request.intent.declared else DeclaredIntent()
            constraints = declared.constraints if declared else []
            satisfies_constraints, constraint_reasons = self.constraint_matcher.satisfies_user_constraints(
                ad, constraints
            )
            if not satisfies_constraints:
                logger.debug(f"Ad {idx} ({ad.id}) rejected at constraint check: {constraint_reasons[:2]}")
                continue
            ads_passed_constraints += 1

            # Filter 3: Relevance scoring
            relevance_score, relevance_reasons = self.relevance_scorer.compute_ad_relevance(ad, request.intent)

            # Log relevance scores for debugging
            logger.debug(f"Ad {idx} ({ad.id}) relevance score: {relevance_score:.3f}")

            # Apply minimum threshold (initialized at start of function)
            if relevance_score > min_threshold:
                ads_passed_relevance += 1
                # Combine all reasons
                all_reasons = constraint_reasons + relevance_reasons

                matched_ad = MatchedAd(ad=ad, adRelevanceScore=relevance_score, matchReasons=all_reasons)
                matched.append(matched_ad)
            else:
                logger.debug(f"Ad {idx} ({ad.id}) below threshold {min_threshold}: {relevance_score:.3f}")

        # Sort by relevance (descending)
        matched.sort(key=lambda x: x.adRelevanceScore, reverse=True)

        # Limit to top K (default 5)
        k = request.config.get("topK", 5) if request.config else 5
        matched = matched[:k]

        # Log comprehensive metrics
        logger.info(
            f"Ad matching completed: {len(matched)}/{total_ads_evaluated} matched | "
            f"Fairness: {ads_passed_fairness}/{total_ads_evaluated} | "
            f"Constraints: {ads_passed_constraints}/{ads_passed_fairness} | "
            f"Relevance: {ads_passed_relevance}/{ads_passed_constraints} above {min_threshold}"
        )

        return AdMatchingResponse(
            matchedAds=matched,
            metrics={
                "totalAdsEvaluated": total_ads_evaluated,
                "adsPassedFairness": ads_passed_fairness,
                "adsPassedConstraints": ads_passed_constraints,
                "adsPassedRelevance": ads_passed_relevance,
                "adsFiltered": total_ads_evaluated - len(matched),
            },
        )


# Global instance for caching
_ad_matcher_instance = None


def get_ad_matcher() -> AdMatcher:
    """
    Get singleton instance of AdMatcher with cached model
    """
    global _ad_matcher_instance
    if _ad_matcher_instance is None:
        _ad_matcher_instance = AdMatcher()
    return _ad_matcher_instance


def match_ads(request: AdMatchingRequest) -> AdMatchingResponse:
    """
    Main entry point for matching ads to user intent
    """
    matcher = get_ad_matcher()
    return matcher.match_ads(request)
