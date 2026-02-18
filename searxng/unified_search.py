"""
Unified Search Service

Combines SearXNG privacy search with Intent Engine ranking to provide
privacy-focused, intent-aware search results.

Flow:
1. Extract intent from user query
2. Search SearXNG for results
3. Rank results based on intent alignment
4. Return privacy-enhanced, ranked results
"""

import hashlib
import logging
import time
from datetime import datetime
from functools import lru_cache
from typing import Any

from core.schema import UniversalIntent
from extraction.extractor import IntentExtractionRequest, extract_intent
from models import (
    ExtractedIntent,
    RankedSearchResult,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from ranking.url_ranker import URLRankingRequest, rank_urls
from searxng.client import SearXNGResult, get_searxng_client

logger = logging.getLogger(__name__)


class UnifiedSearchService:
    """
    Service for unified privacy search with intent ranking.

    Features:
    - Privacy-first search (no tracking via SearXNG)
    - Intent extraction from queries
    - Intent-aware result ranking
    - Privacy score calculation
    - Ethical alignment scoring
    """

    def __init__(self):
        self.searxng_client = get_searxng_client()
        logger.info("Unified Search Service initialized")

    async def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        """
        Perform unified search with intent extraction and ranking.

        Args:
            request: Unified search request with query and options

        Returns:
            UnifiedSearchResponse with ranked results
        """
        import asyncio

        start_time = time.time()
        logger.info(
            f"Unified search: query='{request.query}', "
            f"extract_intent={request.extract_intent}, rank_results={request.rank_results}"
        )

        # Step 1 & 2: Extract intent and Search SearXNG in PARALLEL
        # These two operations are independent and can run concurrently
        extracted_intent = None
        universal_intent = None

        # Build task list for parallel execution
        tasks = []
        intent_task_idx = None

        if request.extract_intent:
            intent_task_idx = len(tasks)
            # Run intent extraction in thread pool (it's CPU-bound)
            tasks.append(asyncio.to_thread(self._extract_intent_with_error_handling, request.query))

        # Always run SearXNG search (it's I/O-bound)
        tasks.append(self._search_searxng(request))

        # Execute all tasks in parallel
        logger.debug(f"Running {len(tasks)} tasks in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process intent extraction result
        if intent_task_idx is not None:
            intent_result = results[intent_task_idx]
            if isinstance(intent_result, Exception):
                logger.warning(f"Intent extraction failed: {intent_result}")
                universal_intent = None
            else:
                logger.debug(f"intent_response: {intent_result is not None}")
                if intent_result and hasattr(intent_result, "intent"):
                    universal_intent = intent_result.intent
                    logger.debug(f"universal_intent extracted: {universal_intent is not None}")
                    if universal_intent:
                        extracted_intent = self._convert_to_extracted_intent(universal_intent)
                        logger.debug(
                            f"extracted_intent converted: goal={extracted_intent.goal if extracted_intent else 'None'}"
                        )
                else:
                    logger.warning("Intent extraction returned None or missing 'intent' attribute")

        # Process SearXNG result (last task)
        searxng_result = results[-1]
        if isinstance(searxng_result, Exception):
            logger.error(f"SearXNG search failed: {searxng_result}")
            searxng_results = []
        else:
            searxng_results = searxng_result

        logger.info(f"SearXNG returned {len(searxng_results)} results")

        # Step 3: Rank results based on intent
        if request.rank_results and universal_intent:
            logger.debug("Ranking with intent enabled")
            try:
                ranked_results = await self._rank_with_intent(searxng_results, universal_intent, request)
            except Exception as e:
                logger.error(f"Ranking failed: {e}")
                import traceback

                logger.error(traceback.format_exc())
                ranked_results = self._convert_to_ranked_results(searxng_results)
        else:
            # Use original SearXNG ranking
            logger.debug("Using original SearXNG ranking (no intent ranking)")
            ranked_results = self._convert_to_ranked_results(searxng_results)

        # Step 4: Apply privacy filters
        if request.min_privacy_score or request.exclude_big_tech:
            logger.debug("Applying privacy filters")
            ranked_results = self._apply_privacy_filters(ranked_results, request)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Build response - defensive handling of engines_used
        try:
            engines_used = list(set(r.engine for r in searxng_results if r.engine))
        except Exception as e:
            logger.warning(f"Error extracting engines: {e}")
            engines_used = []

        response = UnifiedSearchResponse(
            query=request.query,
            results=ranked_results,
            total_results=len(ranked_results),
            processing_time_ms=processing_time_ms,
            extracted_intent=extracted_intent,
            engines_used=engines_used,
            categories_searched=request.categories or ["general"],
            ranking_applied=request.rank_results and universal_intent is not None,
            results_ranked=len([r for r in ranked_results if r.ranked_score != r.original_score]),
            privacy_enhanced=True,
            tracking_blocked=True,
        )

        logger.info(f"Unified search complete: {len(response.results)} results in {processing_time_ms:.2f}ms")
        return response

    def _extract_intent(self, query: str) -> Any:
        """Extract intent from search query."""
        intent_request = IntentExtractionRequest(
            product="search",
            input={"text": query},
            context={
                "sessionId": f"search_{datetime.utcnow().timestamp()}",
                "userLocale": "en-US",
            },
        )
        return extract_intent(intent_request)

    def _extract_intent_with_error_handling(self, query: str) -> Any:
        """Extract intent with error handling for parallel execution."""
        try:
            return self._extract_intent(query)
        except Exception as e:
            logger.warning(f"Intent extraction error in thread: {e}")
            raise

    def _convert_to_extracted_intent(self, universal_intent: UniversalIntent) -> ExtractedIntent:
        """Convert UniversalIntent to ExtractedIntent for API response."""
        # Defensive: handle None values in inferred intent
        inferred = universal_intent.inferred if universal_intent.inferred else None
        declared = universal_intent.declared if universal_intent.declared else None

        # Handle use_cases - might be None or empty list
        use_cases_list = getattr(inferred, "useCases", []) if inferred else []
        if use_cases_list is None:
            use_cases_list = []

        # Handle constraints - might be None or empty list
        constraints_list = getattr(declared, "constraints", []) if declared else []
        if constraints_list is None:
            constraints_list = []

        return ExtractedIntent(
            goal=(declared.goal.value if declared and declared.goal else "unknown"),
            constraints=[
                {
                    "type": c.type.value if hasattr(c.type, "value") else str(c.type),
                    "dimension": c.dimension,
                    "value": c.value,
                }
                for c in constraints_list
            ],
            use_cases=[uc.value for uc in use_cases_list],
            result_type=(inferred.resultType.value if inferred and inferred.resultType else "unknown"),
            complexity=(inferred.complexity.value if inferred and inferred.complexity else "moderate"),
            confidence=0.8,  # Default confidence
        )

    async def _search_searxng(self, request: UnifiedSearchRequest) -> list[SearXNGResult]:
        """Search SearXNG and return results."""
        try:
            logger.debug(f"_search_searxng: calling SearXNG client with query='{request.query}'")
            response = await self.searxng_client.search(
                query=request.query,
                categories=request.categories,
                engines=request.engines,
                language=request.language,
                safe_search=request.safe_search,
                time_range=request.time_range,
            )
            logger.debug(
                f"_search_searxng: got {len(response.results) if response else 0} results from SearXNGResponse"
            )
            return response.results if response else []
        except Exception as e:
            logger.error(f"SearXNG search failed: {e}")
            # Return empty results on error
            return []

    async def _rank_with_intent(
        self,
        results: list[SearXNGResult],
        universal_intent: UniversalIntent,
        request: UnifiedSearchRequest,
    ) -> list[RankedSearchResult]:
        """
        Rank search results based on intent alignment.

        Uses the URL ranker to score and rank results based on:
        - Query relevance
        - Intent alignment
        - Privacy compliance
        - Ethical alignment
        """
        logger.debug(f"_rank_with_intent: {len(results)} results, intent={universal_intent is not None}")

        # Convert SearXNG results to URL ranking format (list of URL strings)
        urls_to_rank = [r.url for r in results]
        logger.debug(f"urls_to_rank: {len(urls_to_rank)} URLs")

        # Build URL ranking request
        ranking_request = URLRankingRequest(
            query=request.query,
            urls=urls_to_rank,
            intent=universal_intent,
            options={
                "weights": request.weights,
                "min_privacy_score": request.min_privacy_score,
                "exclude_big_tech": request.exclude_big_tech,
            },
        )

        # Rank URLs
        logger.debug("Calling rank_urls...")
        ranking_response = await rank_urls(ranking_request)
        logger.debug(f"ranking_response: {ranking_response is not None}")

        # Defensive: check if ranking_response is None
        if ranking_response is None:
            logger.error("rank_urls returned None, returning original results")
            return self._convert_to_ranked_results(results)

        ranked_map = {r.url: r for r in ranking_response.ranked_urls}
        logger.debug(f"ranked_map built with {len(ranked_map)} entries")

        ranked_results = []
        for idx, ranked in enumerate(ranking_response.ranked_urls):
            # Find original SearXNG result
            original = next((r for r in results if r.url == ranked.url), None)

            ranked_result = RankedSearchResult(
                url=ranked.url,
                title=ranked.title or (original.title if original else ""),
                content=ranked.description or (original.content if original else ""),
                engine=original.engine if original else "unknown",
                original_score=(original.score if original and original.score is not None else 0.0),
                ranked_score=ranked.final_score,
                rank=idx + 1,
                category=original.category if original else "general",
                thumbnail=original.thumbnail if original else None,
                published_date=original.published_date if original else None,
                intent_goal=(
                    universal_intent.declared.goal.value
                    if universal_intent.declared and universal_intent.declared.goal
                    else None
                ),
                match_reasons=self._generate_match_reasons(ranked, universal_intent),
                privacy_score=ranked.privacy_score,
                ethical_alignment=ranked.privacy_score,  # Use privacy as proxy for ethics
            )
            ranked_results.append(ranked_result)

        logger.debug(f"_rank_with_intent returning {len(ranked_results)} results")
        return ranked_results

    def _convert_to_ranked_results(self, results: list[SearXNGResult]) -> list[RankedSearchResult]:
        """Convert SearXNG results to RankedSearchResult without intent ranking."""
        return [
            RankedSearchResult(
                url=r.url,
                title=r.title,
                content=r.content,
                engine=r.engine,
                original_score=r.score if r.score is not None else 0.0,
                ranked_score=r.score if r.score is not None else 0.0,
                rank=r.position,
                category=r.category,
                thumbnail=r.thumbnail,
                published_date=r.published_date,
                intent_goal=None,
                match_reasons=[],
                privacy_score=None,
                ethical_alignment=None,
            )
            for r in results
        ]

    def _apply_privacy_filters(
        self, results: list[RankedSearchResult], request: UnifiedSearchRequest
    ) -> list[RankedSearchResult]:
        """Apply privacy-based filtering to results."""
        filtered = []

        big_tech_domains = [
            "google.com",
            "facebook.com",
            "amazon.com",
            "microsoft.com",
            "apple.com",
            "twitter.com",
            "instagram.com",
            "linkedin.com",
            "youtube.com",
            "tiktok.com",
        ]

        for result in results:
            # Filter by privacy score
            if request.min_privacy_score and result.privacy_score:
                if result.privacy_score < request.min_privacy_score:
                    continue

            # Filter big tech
            if request.exclude_big_tech:
                domain = result.url.split("/")[2].lower() if "/" in result.url else result.url.lower()
                if any(bt in domain for bt in big_tech_domains):
                    continue

            filtered.append(result)

        # Re-number ranks
        for idx, result in enumerate(filtered):
            result.rank = idx + 1

        return filtered

    def _generate_match_reasons(self, ranked_result: Any, universal_intent: UniversalIntent) -> list[str]:
        """Generate human-readable match reasons for a result."""
        reasons = []

        # Defensive: check if universal_intent or its components are None
        if not universal_intent:
            return reasons

        declared = universal_intent.declared if universal_intent.declared else None
        inferred = universal_intent.inferred if universal_intent.inferred else None

        # Intent goal match
        if declared and declared.goal:
            reasons.append(f"Matches {declared.goal.value} intent")

        # Use case match
        if inferred and inferred.useCases:
            use_case = inferred.useCases[0]
            reasons.append(f"Suitable for {use_case.value}")

        # Privacy alignment
        if hasattr(ranked_result, "privacy_score") and ranked_result.privacy_score:
            if ranked_result.privacy_score > 0.8:
                reasons.append("High privacy rating")
            elif ranked_result.privacy_score > 0.5:
                reasons.append("Good privacy rating")

        # Ethical signals
        if inferred and inferred.ethicalSignals:
            for signal in inferred.ethicalSignals:
                reasons.append(f"Aligns with {signal.dimension.value} values")

        return reasons[:3]  # Limit to top 3 reasons


# Singleton instance
_unified_search_service: UnifiedSearchService | None = None


def get_unified_search_service() -> UnifiedSearchService:
    """Get or create unified search service singleton."""
    global _unified_search_service

    if _unified_search_service is None:
        _unified_search_service = UnifiedSearchService()

    return _unified_search_service


# Module-level cache for intent extraction results
# Cache up to 1000 recent intent extractions
@lru_cache(maxsize=1000)
def _cached_extract_intent(query_hash: str, query: str):
    """
    Cached version of intent extraction.

    Args:
        query_hash: MD5 hash of normalized query (for cache key)
        query: Original query string

    Returns:
        Intent extraction result or None if failed
    """
    try:
        intent_request = IntentExtractionRequest(
            product="search",
            input={"text": query},
            context={
                "sessionId": f"search_cached_{datetime.utcnow().timestamp()}",
                "userLocale": "en-US",
            },
        )
        return extract_intent(intent_request)
    except Exception as e:
        logger.warning(f"Cached intent extraction failed: {e}")
        return None


def extract_intent_cached(query: str) -> Any:
    """
    Extract intent with LRU caching for repeated queries.

    Args:
        query: Search query string

    Returns:
        Intent extraction result or None
    """
    # Normalize query for consistent caching
    normalized_query = query.lower().strip()
    query_hash = hashlib.md5(normalized_query.encode()).hexdigest()

    return _cached_extract_intent(query_hash, normalized_query)
