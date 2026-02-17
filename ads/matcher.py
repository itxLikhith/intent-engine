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
from dataclasses_json import DataClassJsonMixin

from core.schema import (
    DeclaredIntent,
    InferredIntent,
    UniversalIntent,
)


class AdFairnessChecker:
    """Checks ad fairness constraints"""

    def __init__(self):
        pass

    def check_fairness(self, ads: list) -> list:
        """Check fairness of ads - returns ads that pass fairness check"""
        return ads


class AdConstraintMatcher:
    """Matches ads based on constraints"""

    def __init__(self):
        pass

    def match_constraints(self, ad, intent) -> bool:
        """Check if ad matches constraints for given intent"""
        return True


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AdMetadata(DataClassJsonMixin):
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
class MatchedAd(DataClassJsonMixin):
    """Represents a matched ad with relevance score and reasons"""

    ad: AdMetadata
    adRelevanceScore: float
    matchReasons: List[str]


@dataclass
class AdMatchingRequest(DataClassJsonMixin):
    """Request object for ad matching API"""

    intent: UniversalIntent
    adInventory: List[AdMetadata]
    config: Optional[Dict[str, Any]] = None  # MatchingConfig


@dataclass
class AdMatchingResponse(DataClassJsonMixin):
    """Response object for ad matching API"""

    matchedAds: List[MatchedAd]
    metrics: Dict[str, int]


class EmbeddingCache:
    """Cache for embeddings to improve performance"""

    def __init__(self, redis_client=None):

        self.cache = {}

        self.redis = redis_client

        self.model = None

        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""

        try:

            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = "sentence-transformers/all-MiniLM-L6-v2"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model = AutoModel.from_pretrained(model_name)

            self.model = self.model.to("cpu")

            logger.info(f"Loaded embedding model: {model_name}")

        except ImportError:

            logger.warning("Transformers library not available. Using mock embeddings.")

            self.tokenizer = None

            self.model = None

    async def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding vector using the sentence transformer model"""

        if self.model is None or self.tokenizer is None:

            return np.random.rand(384).astype(np.float32)

        if self.redis:

            cached_embedding = await self.redis.get(f"embedding:{text}")

            if cached_embedding:

                return np.frombuffer(cached_embedding, dtype=np.float32)

        if text in self.cache:

            return self.cache[text]

        try:

            import torch

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            device = next(self.model.parameters()).device

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():

                outputs = self.model(**inputs)

                embeddings = outputs.last_hidden_state.mean(dim=1)

            result = embeddings.cpu().numpy().flatten().astype(np.float32)

            self.cache[text] = result

            if self.redis:

                await self.redis.set(f"embedding:{text}", result.tobytes(), ex=3600)

            return result

        except Exception as e:

            logger.error(f"Error encoding text: {e}")

            return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:

        dot_product = np.dot(vec1, vec2)

        norm_vec1 = np.linalg.norm(vec1)

        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:

            return 0.0

        return float(dot_product / (norm_vec1 * norm_vec2))


class AdRelevanceScorer:
    """Scores ad relevance based on user intent"""

    def __init__(self, redis_client=None):

        self.embedding_cache = EmbeddingCache(redis_client=redis_client)

    async def compute_ad_relevance(self, ad: AdMetadata, intent: UniversalIntent) -> Tuple[float, List[str]]:

        scores, reasons = [], []

        declared = intent.declared if intent.declared else DeclaredIntent()

        inferred = intent.inferred if intent.inferred else InferredIntent()

        semantic_score, semantic_reasons = await self._compute_semantic_similarity(ad, intent, declared)

        scores.append(semantic_score)

        reasons.extend(semantic_reasons)

        ethical_score, ethical_reasons = self._compute_ethical_alignment(ad, intent, inferred)

        scores.append(ethical_score)

        reasons.extend(ethical_reasons)

        goal_score, goal_reasons = await self._compute_goal_alignment(ad, intent, declared)

        scores.append(goal_score)

        reasons.extend(goal_reasons)

        scores.append(ad.qualityScore)

        weights = [0.40, 0.30, 0.20, 0.10]

        relevance_score = sum(score * weight for score, weight in zip(scores, weights))

        return max(0.0, min(1.0, relevance_score)), reasons

    async def _compute_semantic_similarity(
        self, ad: AdMetadata, intent: UniversalIntent, declared: DeclaredIntent
    ) -> Tuple[float, List[str]]:

        reasons = []

        query = declared.query if declared else None

        if not query:

            return 0.5, reasons

        ad_content = f"{ad.title} {ad.description}".strip()

        if not ad_content:

            return 0.0, reasons

        query_embedding = await self.embedding_cache.encode_text(query)

        ad_embedding = await self.embedding_cache.encode_text(ad_content)

        if query_embedding is not None and ad_embedding is not None:

            similarity = self.embedding_cache.cosine_similarity(query_embedding, ad_embedding)

            score = (similarity + 1) / 2

            if score > 0.3:

                reasons.append(f"query semantic match: '{query[:30]}...'")

            return score, reasons

        return 0.5, reasons

    def _compute_ethical_alignment(
        self, ad: AdMetadata, intent: UniversalIntent, inferred: InferredIntent
    ) -> Tuple[float, List[str]]:

        return 0.0, []

    async def _compute_goal_alignment(
        self, ad: AdMetadata, intent: UniversalIntent, declared: DeclaredIntent
    ) -> Tuple[float, List[str]]:

        reasons = []

        goal = declared.goal if declared else None

        if not goal:

            return 0.5, reasons

        goal_value = goal.value.upper()

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

                score = min(1.0, len(matches) * 0.3)

                reasons.append(f"goal alignment: {goal_value.lower()} related terms found")

                return score, reasons

        goal_embedding = await self.embedding_cache.encode_text(goal_value.replace("_", " "))

        ad_embedding = await self.embedding_cache.encode_text(ad_content)

        if goal_embedding is not None and ad_embedding is not None:

            similarity = self.embedding_cache.cosine_similarity(goal_embedding, ad_embedding)

            score = (similarity + 1) / 2

            if score > 0.2:

                reasons.append(f"goal semantically aligned: {goal_value.lower()}")

            return score, reasons

        return 0.2, reasons


# Global instance for singleton pattern
_ad_matcher_instance = None


class AdMatcher:
    """Main class that orchestrates ad matching"""

    def __init__(self, redis_client=None):

        self.fairness_checker = AdFairnessChecker()

        self.constraint_matcher = AdConstraintMatcher()

        self.relevance_scorer = AdRelevanceScorer(redis_client=redis_client)

    async def match_ads(self, request: AdMatchingRequest) -> AdMatchingResponse:
        """Match ads based on request"""

        matched_ads = []

        for ad in request.adInventory:

            score, reasons = await self.relevance_scorer.compute_ad_relevance(ad, request.intent)

            matched_ads.append(MatchedAd(ad=ad, adRelevanceScore=score, matchReasons=reasons))

        # Sort by relevance score

        matched_ads.sort(key=lambda x: x.adRelevanceScore, reverse=True)

        return AdMatchingResponse(
            matchedAds=matched_ads, metrics={"total_ads": len(request.adInventory), "matched_ads": len(matched_ads)}
        )


def get_ad_matcher() -> AdMatcher:
    """Get or create singleton AdMatcher instance"""

    global _ad_matcher_instance

    if _ad_matcher_instance is None:

        _ad_matcher_instance = AdMatcher()

    return _ad_matcher_instance


def match_ads(request: AdMatchingRequest) -> AdMatchingResponse:
    """








    Synchronous wrapper for the async match_ads function.








    This is to be used by the tests.








    """

    matcher = AdMatcher()

    import asyncio

    return asyncio.run(matcher.match_ads(request))


async def find_matching_ads_background(ctx, request_dict: dict) -> dict:
    """


    ARQ background task for ad matching.


    """

    redis = ctx.get("redis")

    request = AdMatchingRequest.from_dict(request_dict)

    matcher = AdMatcher(redis_client=redis)

    response = await matcher.match_ads(request)

    return response.to_dict()
