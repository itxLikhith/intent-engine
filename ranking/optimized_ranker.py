"""
Intent Engine - Optimized Ranking Module

This module implements improved constraint satisfaction and intent-aligned ranking
with better accuracy and efficiency.

Improvements:
- Optimized embedding cache with persistence
- Better relevance scoring with semantic matching
- Result deduplication
- Batch processing for efficiency
- Configurable ranking weights
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from config.optimized_cache import get_embedding_cache
from config.query_cache import get_ranking_cache

from core.schema import (
    Complexity,
    Constraint,
    ConstraintType,
    DeclaredIntent,
    EthicalDimension,
    Frequency,
    InferredIntent,
    Recency,
    ResultType,
    SkillLevel,
    TemporalHorizon,
    UniversalIntent,
    UseCase,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result candidate with metadata"""

    id: str
    title: str
    description: str
    platform: Optional[str] = None
    provider: Optional[str] = None
    license: Optional[str] = None
    price: Optional[float] = None
    tags: Optional[List[str]] = None
    qualityScore: Optional[float] = 0.5
    recency: Optional[str] = None
    complexity: Optional[str] = None
    compatibility: Optional[List[str]] = None
    privacyRating: Optional[float] = None
    opensource: Optional[bool] = None


@dataclass
class RankedResult:
    """Represents a ranked result with alignment score and reasons"""

    result: SearchResult
    alignmentScore: float
    matchReasons: List[str]


@dataclass
class RankingRequest:
    """Request object for ranking API"""

    intent: UniversalIntent
    candidates: List[SearchResult]
    options: Optional[Dict[str, Any]] = None


@dataclass
class RankingResponse:
    """Response object for ranking API"""

    rankedResults: List[RankedResult]
    cache_hit: bool = False
    processing_time_ms: float = 0.0


class OptimizedConstraintSatisfactionEngine:
    """Improved constraint satisfaction with better performance"""

    def __init__(self):
        self.embedding_cache = get_embedding_cache()

    def satisfies_constraints(self, result: SearchResult, constraints: List[Constraint]) -> bool:
        """Check if a result satisfies all hard constraints"""
        for constraint in constraints:
            if not constraint.hardFilter:
                continue

            if not self._satisfies_single_constraint(result, constraint):
                return False

        return True

    def _satisfies_single_constraint(self, result: SearchResult, constraint: Constraint) -> bool:
        """Check if a result satisfies a single constraint with semantic matching"""
        dimension = constraint.dimension
        value = constraint.value
        constraint_type = constraint.type

        result_value = getattr(result, dimension, None)

        if constraint_type == ConstraintType.INCLUSION:
            if isinstance(value, str):
                if result_value != value:
                    # Try semantic matching for partial matches
                    return self._semantic_match(result_value, value)
            elif isinstance(value, list):
                if result_value not in value:
                    # Check if any value in the list matches semantically
                    return any(self._semantic_match(result_value, v) for v in value)

        elif constraint_type == ConstraintType.EXCLUSION:
            if isinstance(value, str):
                if result_value == value:
                    return False
                # Check semantic exclusion
                if self._semantic_match(result_value, value):
                    return False
            elif isinstance(value, list):
                if result_value in value:
                    return False
                # Check if any excluded value matches semantically
                if any(self._semantic_match(result_value, v) for v in value):
                    return False

        elif constraint_type == ConstraintType.RANGE:
            if dimension == "price" and result_value is not None:
                return self._check_price_range(result_value, value)

        return True

    def _semantic_match(self, value1: Optional[str], value2: str) -> bool:
        """Check if two values match semantically"""
        if not value1 or not value2:
            return False

        # Direct match
        if value1.lower() == value2.lower():
            return True

        # Check if one contains the other
        v1_lower = value1.lower()
        v2_lower = value2.lower()

        if v2_lower in v1_lower or v1_lower in v2_lower:
            return True

        # Use embeddings for semantic similarity
        try:
            emb1 = self.embedding_cache.encode_text(value1)
            emb2 = self.embedding_cache.encode_text(value2)

            if emb1 is not None and emb2 is not None:
                similarity = self.embedding_cache.cosine_similarity(emb1, emb2)
                return similarity > 0.7  # High similarity threshold
        except Exception:
            pass

        return False

    def _check_price_range(self, price: float, range_spec: str) -> bool:
        """Check if price satisfies range specification"""
        # Parse range specification
        # Formats: "<=100", ">=50", "<100", ">50", "budget"
        range_spec = str(range_spec).lower().strip()

        if range_spec == "budget" or range_spec == "free" or range_spec == "0":
            return price == 0

        match = re.match(r"([<>]=?)(\d+)", range_spec)
        if match:
            operator, limit = match.groups()
            limit = float(limit)

            if operator == "<=":
                return price <= limit
            elif operator == ">=":
                return price >= limit
            elif operator == "<":
                return price < limit
            elif operator == ">":
                return price > limit

        return True


class OptimizedIntentAlignmentEngine:
    """Improved intent alignment with better scoring"""

    def __init__(self):
        self.embedding_cache = get_embedding_cache()

    def compute_intent_alignment(self, result: SearchResult, intent: UniversalIntent) -> Tuple[float, List[str]]:
        """Compute alignment score with improved weighting"""
        scores = []
        reasons = []

        # FIX: Add null safety for intent.declared and intent.inferred
        declared = intent.declared or DeclaredIntent()
        inferred = intent.inferred or InferredIntent()

        # 1. Query-content alignment (semantic similarity) - 35%
        query_score, query_reasons = self._compute_query_content_alignment(result, intent, declared)
        scores.append((query_score, 0.35))
        reasons.extend(query_reasons)

        # 2. Use case alignment - 20%
        use_case_score, use_case_reasons = self._compute_use_case_alignment(result, intent, inferred)
        scores.append((use_case_score, 0.20))
        reasons.extend(use_case_reasons)

        # 3. Ethical signal alignment - 15%
        ethical_score, ethical_reasons = self._compute_ethical_alignment(result, intent, inferred)
        scores.append((ethical_score, 0.15))
        reasons.extend(ethical_reasons)

        # 4. Skill level alignment - 15%
        skill_score, skill_reasons = self._compute_skill_alignment(result, intent, declared)
        scores.append((skill_score, 0.15))
        reasons.extend(skill_reasons)

        # 5. Temporal intent alignment - 10%
        temporal_score, temporal_reasons = self._compute_temporal_alignment(result, intent, inferred)
        scores.append((temporal_score, 0.10))
        reasons.extend(temporal_reasons)

        # 6. Quality score - 5%
        quality_score = result.qualityScore or 0.5
        scores.append((quality_score, 0.05))

        # Calculate weighted average
        alignment_score = sum(score * weight for score, weight in scores)

        # Normalize to 0-1 range
        alignment_score = max(0.0, min(1.0, alignment_score))

        return alignment_score, list(set(reasons))  # Remove duplicate reasons

    def _compute_query_content_alignment(
        self, result: SearchResult, intent: UniversalIntent, declared: DeclaredIntent
    ) -> Tuple[float, List[str]]:
        """Compute semantic similarity between query and result"""
        reasons = []

        # FIX: Add null safety for declared.query
        query = declared.query if declared else None
        if not query:
            return 0.5, reasons

        content = f"{result.title} {result.description}".strip()
        if not content:
            return 0.0, reasons

        # Get embeddings
        query_emb = self.embedding_cache.encode_text(query)
        content_emb = self.embedding_cache.encode_text(content)

        if query_emb is not None and content_emb is not None:
            similarity = self.embedding_cache.cosine_similarity(query_emb, content_emb)
            score = (similarity + 1) / 2  # Normalize to 0-1

            if score > 0.5:
                reasons.append("high-semantic-match")
            elif score > 0.3:
                reasons.append("semantic-match")

            # Boost for exact keyword matches
            keyword_score = self._keyword_match_score(query, content)
            score = score * 0.7 + keyword_score * 0.3  # Combine semantic and keyword

            return score, reasons

        # Fallback to keyword matching
        return self._keyword_match_score(query, content), ["keyword-match"]

    def _keyword_match_score(self, query: str, content: str) -> float:
        """Calculate keyword match score"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.5

        # Calculate weighted match
        matches = query_words.intersection(content_words)

        # Boost for title matches
        title_bonus = 0
        if hasattr(self, "_last_result_title"):
            title_words = set(self._last_result_title.lower().split())
            title_matches = query_words.intersection(title_words)
            title_bonus = len(title_matches) / len(query_words) * 0.2

        return (len(matches) / len(query_words)) * 0.8 + title_bonus

    def _compute_use_case_alignment(
        self, result: SearchResult, intent: UniversalIntent, inferred: InferredIntent
    ) -> Tuple[float, List[str]]:
        """Compute use case alignment"""
        reasons = []

        # FIX: Add null safety for inferred.useCases
        use_cases = inferred.useCases if inferred else []
        if not use_cases or not result.tags:
            return 0.5, reasons

        score = 0.0
        tag_text = " ".join(result.tags).lower()

        # Encode tags once
        tag_emb = self.embedding_cache.encode_text(tag_text)

        for use_case in use_cases:
            use_case_str = use_case.value.replace("_", " ")

            # Direct match
            if use_case_str in tag_text:
                score += 0.5
                reasons.append(f"use-case-{use_case.value}-direct")
            elif tag_emb is not None:
                # Semantic match
                use_case_emb = self.embedding_cache.encode_text(use_case_str)
                if use_case_emb is not None:
                    similarity = self.embedding_cache.cosine_similarity(use_case_emb, tag_emb)
                    if similarity > 0.5:
                        score += similarity * 0.3
                        reasons.append(f"use-case-{use_case.value}-semantic")

        # Normalize
        if use_cases:
            score = min(1.0, score / len(use_cases))

        return score, reasons

    def _compute_ethical_alignment(
        self, result: SearchResult, intent: UniversalIntent, inferred: InferredIntent
    ) -> Tuple[float, List[str]]:
        """Compute ethical signal alignment"""
        reasons = []

        # FIX: Add null safety for inferred.ethicalSignals
        ethical_signals = inferred.ethicalSignals if inferred else []
        if not ethical_signals:
            return 0.5, reasons

        score = 0.0

        for signal in ethical_signals:
            if signal.dimension == EthicalDimension.PRIVACY:
                if result.privacyRating and result.privacyRating > 0.7:
                    score += 0.5
                    reasons.append("privacy-aligned")
                # Check for privacy keywords in description
                privacy_keywords = ["privacy", "encrypted", "secure", "private"]
                if any(kw in result.description.lower() for kw in privacy_keywords):
                    score += 0.2

            elif signal.dimension == EthicalDimension.OPENNESS:
                if result.opensource:
                    score += 0.5
                    reasons.append("open-source-aligned")
                # Check for open source keywords
                oss_keywords = ["open source", "foss", "free software"]
                if any(kw in result.description.lower() for kw in oss_keywords):
                    score += 0.2

        # Normalize
        if ethical_signals:
            score = min(1.0, score / len(ethical_signals))

        return score, reasons

    def _compute_skill_alignment(
        self, result: SearchResult, intent: UniversalIntent, declared: DeclaredIntent
    ) -> Tuple[float, List[str]]:
        """Compute skill level alignment"""
        reasons = []

        # FIX: Add null safety for declared.skillLevel
        declared_skill = declared.skillLevel if declared else SkillLevel.INTERMEDIATE
        result_complexity = result.complexity

        if not result_complexity:
            return 0.5, reasons

        # Map complexity to skill levels
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
                reasons.append(f"exact-skill-match-{result_complexity}")
            elif abs(declared_skill.value - result_skill.value) == 1:
                score = 0.8
                reasons.append(f"adjacent-skill-level-{result_complexity}")
            else:
                score = 0.3
        else:
            score = 0.5

        return score, reasons

    def _compute_temporal_alignment(
        self, result: SearchResult, intent: UniversalIntent, inferred: InferredIntent
    ) -> Tuple[float, List[str]]:
        """Compute temporal intent alignment"""
        reasons = []
        score = 0.5

        # FIX: Add null safety for inferred.temporalIntent
        temporal_intent = inferred.temporalIntent if inferred else None
        if not temporal_intent or not result.recency:
            return score, reasons

        try:
            # Parse recency date
            result_date = self._parse_date(result.recency)
            if result_date:
                now = datetime.now(timezone.utc)
                days_old = (now - result_date).days

                # Check recency preferences
                if temporal_intent.recency == Recency.RECENT:
                    if days_old <= 30:
                        score += 0.3
                        reasons.append("recent-content")
                    elif days_old <= 90:
                        score += 0.1

                # Check horizon preferences
                if temporal_intent.horizon == TemporalHorizon.TODAY and days_old <= 1:
                    score += 0.2
                    reasons.append("today-relevant")
                elif temporal_intent.horizon == TemporalHorizon.WEEK and days_old <= 7:
                    score += 0.15
                    reasons.append("week-relevant")

        except Exception:
            pass

        return max(0.0, min(1.0, score)), reasons

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime"""
        try:
            if date_str.endswith("Z"):
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            elif "+" in date_str or date_str.count("-") > 2:
                return datetime.fromisoformat(date_str)
            else:
                return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        except:
            return None


class ResultDeduplicator:
    """Deduplicate results based on content similarity"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.embedding_cache = get_embedding_cache()

    def deduplicate(self, results: List[RankedResult]) -> List[RankedResult]:
        """Remove duplicate results based on content similarity"""
        if not results:
            return results

        unique_results = []
        seen_embeddings = []

        for result in results:
            content = f"{result.result.title} {result.result.description}"
            embedding = self.embedding_cache.encode_text(content)

            if embedding is None:
                unique_results.append(result)
                continue

            # Check similarity with all seen results
            is_duplicate = False
            for seen_emb in seen_embeddings:
                similarity = self.embedding_cache.cosine_similarity(embedding, seen_emb)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)
                seen_embeddings.append(embedding)

        return unique_results


class OptimizedIntentRanker:
    """Main ranking class with optimizations"""

    def __init__(self):
        self.constraint_engine = OptimizedConstraintSatisfactionEngine()
        self.alignment_engine = OptimizedIntentAlignmentEngine()
        self.deduplicator = ResultDeduplicator()
        self.cache = get_ranking_cache()

    def rank_results(self, request: RankingRequest) -> RankingResponse:
        """Rank results with caching and optimizations"""
        import time

        start_time = time.time()

        # Create cache key
        cache_key = self._make_cache_key(request)

        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            cached_result.cache_hit = True
            cached_result.processing_time_ms = 0
            return cached_result

        # Filter candidates
        filtered_candidates = self._filter_candidates(request)

        # Compute alignment scores
        ranked_results = []
        for candidate in filtered_candidates:
            score, reasons = self.alignment_engine.compute_intent_alignment(candidate, request.intent)
            ranked_results.append(RankedResult(result=candidate, alignmentScore=score, matchReasons=reasons))

        # Deduplicate
        ranked_results = self.deduplicator.deduplicate(ranked_results)

        # Sort by score
        ranked_results.sort(key=lambda x: x.alignmentScore, reverse=True)

        # Create response
        processing_time = (time.time() - start_time) * 1000
        response = RankingResponse(rankedResults=ranked_results, cache_hit=False, processing_time_ms=processing_time)

        # Cache result
        self.cache.set(cache_key, response)

        return response

    def _filter_candidates(self, request: RankingRequest) -> List[SearchResult]:
        """Filter candidates based on constraints"""
        filtered = []
        for candidate in request.candidates:
            if self.constraint_engine.satisfies_constraints(candidate, request.intent.declared.constraints):
                filtered.append(candidate)
        return filtered

    def _make_cache_key(self, request: RankingRequest) -> str:
        """Create cache key for request"""
        # Hash intent query and candidate IDs
        intent_str = request.intent.declared.query if request.intent.declared.query else ""
        candidate_ids = sorted([c.id for c in request.candidates])
        key_data = f"{intent_str}:{','.join(candidate_ids)}"
        return hashlib.md5(key_data.encode()).hexdigest()


# Global instance
_ranker_instance = None


def get_intent_ranker() -> OptimizedIntentRanker:
    """Get singleton instance"""
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = OptimizedIntentRanker()
    return _ranker_instance


def rank_results(request: RankingRequest) -> RankingResponse:
    """Main entry point"""
    ranker = get_intent_ranker()
    return ranker.rank_results(request)
