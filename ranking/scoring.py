"""
Intent Engine - Scoring Module

This module contains scoring algorithms for alignment, quality, and ethical evaluation.
"""

from ..core.schema import (
    EthicalDimension,
    UniversalIntent,
)
from ..core.utils import get_embedding_cache


class AlignmentScorer:
    """Scores alignment between user intent and results"""

    def __init__(self):
        self.embedding_cache = get_embedding_cache()

    def compute_alignment_score(self, text: str, intent: UniversalIntent) -> float:
        """
        Compute alignment score between text and user intent
        """
        score = 0.0

        # Query-content alignment
        if intent.declared.query:
            query_embedding = self.embedding_cache.encode_text(intent.declared.query)
            text_embedding = self.embedding_cache.encode_text(text)

            if query_embedding is not None and text_embedding is not None:
                similarity = self.embedding_cache.cosine_similarity(query_embedding, text_embedding)
                # Normalize to 0-1 range
                score += (similarity + 1) / 2 * 0.5  # 50% weight for query alignment

        # Use case alignment
        if intent.inferred.useCases:
            use_case_text = " ".join([uc.value.replace("_", " ") for uc in intent.inferred.useCases])
            use_case_embedding = self.embedding_cache.encode_text(use_case_text)
            text_embedding = self.embedding_cache.encode_text(text)

            if use_case_embedding is not None and text_embedding is not None:
                similarity = self.embedding_cache.cosine_similarity(use_case_embedding, text_embedding)
                score += (similarity + 1) / 2 * 0.3  # 30% weight for use case alignment

        # Ethical alignment
        if intent.inferred.ethicalSignals:
            ethical_text = " ".join([es.preference for es in intent.inferred.ethicalSignals])
            ethical_embedding = self.embedding_cache.encode_text(ethical_text)
            text_embedding = self.embedding_cache.encode_text(text)

            if ethical_embedding is not None and text_embedding is not None:
                similarity = self.embedding_cache.cosine_similarity(ethical_embedding, text_embedding)
                score += (similarity + 1) / 2 * 0.2  # 20% weight for ethical alignment

        return min(1.0, max(0.0, score))


class QualityScorer:
    """Scores quality of results based on various factors"""

    @staticmethod
    def compute_quality_score(relevance: float, freshness: float, authority: float, engagement: float) -> float:
        """
        Compute overall quality score from multiple factors
        """
        # Weighted average of quality factors
        weights = [0.4, 0.2, 0.2, 0.2]  # [relevance, freshness, authority, engagement]
        scores = [relevance, freshness, authority, engagement]

        quality_score = sum(w * s for w, s in zip(weights, scores, strict=False))
        return min(1.0, max(0.0, quality_score))


class EthicalScorer:
    """Scores ethical alignment of results"""

    def compute_ethical_score(self, result_metadata: dict, intent: UniversalIntent) -> tuple[float, list[str]]:
        """
        Compute ethical score based on user preferences and result properties
        Returns (score, list_of_reasons)
        """
        score = 0.5  # Base neutral score
        reasons = []

        for signal in intent.inferred.ethicalSignals:
            if signal.dimension == EthicalDimension.PRIVACY:
                # Check if result has privacy-friendly properties
                privacy_rating = result_metadata.get("privacy_rating", 0.0)
                if privacy_rating > 0.7:
                    score += 0.2
                    reasons.append("high_privacy_rating")
                elif privacy_rating > 0.5:
                    score += 0.1
                    reasons.append("medium_privacy_rating")

            elif signal.dimension == EthicalDimension.OPENNESS:
                # Check if result is open source or transparent
                is_opensource = result_metadata.get("opensource", False)
                if is_opensource:
                    score += 0.2
                    reasons.append("open_source")

            elif signal.dimension == EthicalDimension.SUSTAINABILITY:
                # Check if result has sustainability properties
                sustainability_score = result_metadata.get("sustainability_score", 0.0)
                if sustainability_score > 0.7:
                    score += 0.15
                    reasons.append("high_sustainability")

        # Normalize score to 0-1 range
        score = min(1.0, max(0.0, score))

        return score, reasons
