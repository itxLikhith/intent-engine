"""
Intent Engine - Result Aggregation and Deduplication

Aggregates search results from multiple backends, removes duplicates,
and normalizes scores for consistent ranking.

Features:
- URL-based deduplication (with normalization)
- Score normalization across backends
- Source attribution
- Result merging
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from searxng.query_router import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class AggregatedResult:
    """Result after aggregation and deduplication"""

    url: str
    url_hash: str
    title: str
    content: str
    sources: list[str] = field(default_factory=list)
    best_score: float = 0.0
    avg_score: float = 0.0
    result_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "url": self.url,
            "url_hash": self.url_hash,
            "title": self.title,
            "content": self.content,
            "sources": self.sources,
            "best_score": self.best_score,
            "avg_score": self.avg_score,
            "result_count": self.result_count,
            "metadata": self.metadata,
        }


class ResultAggregator:
    """
    Aggregates results from multiple backends.

    Features:
    - URL-based deduplication (with normalization to remove tracking params)
    - Score normalization across different backend scoring systems
    - Source attribution (which backends returned each result)
    - Result merging (combine duplicate results)

    Usage:
        aggregator = ResultAggregator()
        aggregated = aggregator.aggregate(search_results)
    """

    def __init__(self, dedup_threshold: float = 0.95):
        """
        Initialize result aggregator.

        Args:
            dedup_threshold: Similarity threshold for deduplication (0.0-1.0)
                            Higher = more aggressive deduplication
        """
        self.dedup_threshold = dedup_threshold
        logger.info(f"ResultAggregator initialized with dedup_threshold={dedup_threshold}")

    def aggregate(self, results: list[SearchResult]) -> list[AggregatedResult]:
        """
        Aggregate and deduplicate results.

        Algorithm:
        1. Normalize URLs (remove tracking parameters)
        2. Group by normalized URL hash
        3. Merge duplicate entries
        4. Normalize scores across backends
        5. Sort by final score

        Args:
            results: List of SearchResult from multiple backends

        Returns:
            List of AggregatedResult, deduplicated and sorted
        """
        logger.info(f"Aggregating {len(results)} results from multiple backends")

        # Group by URL hash
        url_groups: dict[str, list[SearchResult]] = {}

        for result in results:
            url_key = self._normalize_url(result.url)
            self._hash_url(url_key)

            if url_key not in url_groups:
                url_groups[url_key] = []
            url_groups[url_key].append(result)

        logger.debug(f"Grouped into {len(url_groups)} unique URLs")

        # Merge duplicates
        aggregated = []
        for url_key, group in url_groups.items():
            merged = self._merge_results(url_key, group)
            aggregated.append(merged)

        # Sort by best score (descending)
        aggregated.sort(key=lambda x: x.best_score, reverse=True)

        # Calculate statistics
        source_counts = self._count_by_backend(results)
        logger.info(
            f"Aggregation complete: {len(aggregated)} unique results "
            f"(from {len(results)} total), distribution: {source_counts}"
        )

        return aggregated

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for deduplication.

        Removes common tracking parameters:
        - UTM parameters (utm_source, utm_medium, utm_campaign)
        - Google Click ID (gclid)
        - Facebook Click ID (fbclid)
        - Referral parameters (ref, source)
        - Session IDs

        Args:
            url: Raw URL

        Returns:
            Normalized URL without tracking parameters
        """
        try:
            parsed = urlparse(url)

            # Parse query parameters
            query_params = parse_qs(parsed.query)

            # Remove common tracking parameters
            tracking_params = {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
                "gclid",
                "fbclid",
                "ref",
                "source",
                "medium",
                "campaign",
                "session",
                "sessionid",
                "sid",
                "pk_campaign",
                "pk_kwd",
            }

            filtered_params = {k: v for k, v in query_params.items() if k.lower() not in tracking_params}

            # Rebuild URL
            normalized = parsed._replace(query=urlencode(filtered_params, doseq=True))

            # Remove fragment (often used for tracking)
            normalized = normalized._replace(fragment="")

            # Ensure consistent trailing slash
            path = normalized.path
            if not path.endswith("/") and not path.endswith((".html", ".htm", ".php", ".asp", ".aspx")):
                path = path + "/"
                normalized = normalized._replace(path=path)

            normalized_url = normalized.geturl()

            # Log if we removed tracking parameters
            if len(query_params) != len(filtered_params):
                removed = set(query_params.keys()) - set(filtered_params.keys())
                logger.debug(f"Removed tracking params from URL: {removed}")

            return normalized_url

        except Exception as e:
            logger.warning(f"Failed to normalize URL {url}: {e}")
            # Return original URL if normalization fails
            return url

    def _hash_url(self, url: str) -> str:
        """Generate consistent hash for URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def _merge_results(self, url_key: str, results: list[SearchResult]) -> AggregatedResult:
        """
        Merge multiple results for the same URL.

        Strategy:
        - Use longest title (most descriptive)
        - Use longest content (most informative)
        - Combine sources
        - Use best score
        - Merge metadata

        Args:
            url_key: Normalized URL
            results: List of SearchResult for this URL

        Returns:
            AggregatedResult with merged data
        """
        if not results:
            raise ValueError("Cannot merge empty results")

        # Use best result as base (highest score)
        max(results, key=lambda r: r.score if r.score else 0.0)

        # Find longest title and content (most informative)
        longest_title_result = max(results, key=lambda r: len(r.title))
        longest_content_result = max(results, key=lambda r: len(r.content))

        # Combine sources (which backends returned this URL)
        sources = list({str(r.source.value) for r in results})

        # Calculate scores
        valid_scores = [r.score for r in results if r.score is not None]
        best_score = max(valid_scores) if valid_scores else 0.0
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # Merge metadata
        merged_metadata = {}
        for r in results:
            if r.metadata:
                merged_metadata.update(r.metadata)

        # Add source-specific metadata
        merged_metadata["source_details"] = {}
        for r in results:
            merged_metadata["source_details"][r.source.value] = {
                "engine": r.engine,
                "original_score": r.score,
            }

        return AggregatedResult(
            url=url_key,
            url_hash=self._hash_url(url_key),
            title=longest_title_result.title,
            content=longest_content_result.content,
            sources=sources,
            best_score=best_score,
            avg_score=avg_score,
            result_count=len(results),
            metadata=merged_metadata,
        )

    def _count_by_backend(self, results: list[SearchResult]) -> dict[str, int]:
        """Count results per backend"""
        counts: dict[str, int] = {}
        for result in results:
            source = str(result.source.value)
            counts[source] = counts.get(source, 0) + 1
        return counts

    def deduplicate_by_content(
        self, results: list[SearchResult], similarity_threshold: float = 0.9
    ) -> list[SearchResult]:
        """
        Deduplicate results by content similarity.

        Uses simple Jaccard similarity on word sets.
        For production, consider using MinHash or embeddings.

        Args:
            results: List of SearchResult
            similarity_threshold: Threshold for considering duplicates (0.0-1.0)

        Returns:
            Deduplicated list of SearchResult
        """
        if len(results) <= 1:
            return results

        unique_results = []
        seen_content_sets = []

        for result in results:
            # Tokenize content
            content_words = set(result.content.lower().split())

            # Check similarity against seen results
            is_duplicate = False
            for seen_set in seen_content_sets:
                similarity = self._jaccard_similarity(content_words, seen_set)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"Filtered duplicate result (similarity={similarity:.2f}): {result.url}")
                    break

            if not is_duplicate:
                unique_results.append(result)
                seen_content_sets.append(content_words)

        logger.info(
            f"Content deduplication: {len(results)} → {len(unique_results)} "
            f"(filtered {len(results) - len(unique_results)} duplicates)"
        )

        return unique_results

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def normalize_scores(self, results: list[AggregatedResult], method: str = "minmax") -> list[AggregatedResult]:
        """
        Normalize scores across results.

        Args:
            results: List of AggregatedResult
            method: Normalization method ('minmax', 'zscore', 'sigmoid')

        Returns:
            Results with normalized scores (0.0-1.0)
        """
        if not results:
            return results

        scores = [r.best_score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if method == "minmax":
            # Scale to 0-1 range
            for result in results:
                if max_score - min_score > 0:
                    result.best_score = (result.best_score - min_score) / (max_score - min_score)
                else:
                    result.best_score = 1.0 if result.best_score > 0 else 0.0

        elif method == "zscore":
            # Z-score normalization
            import statistics

            mean = statistics.mean(scores)
            stdev = statistics.stdev(scores) if len(scores) > 1 else 1.0

            for result in results:
                result.best_score = (result.best_score - mean) / stdev if stdev > 0 else 0.0
                # Clamp to 0-1
                result.best_score = max(0.0, min(1.0, result.best_score))

        elif method == "sigmoid":
            # Sigmoid normalization
            import math

            mean = sum(scores) / len(scores)

            for result in results:
                x = result.best_score - mean
                result.best_score = 1 / (1 + math.exp(-x))

        logger.debug(f"Score normalization ({method}): min={min(scores):.3f}, max={max(scores):.3f}")
        return results


# Singleton instance
_aggregator_instance: ResultAggregator | None = None


def get_result_aggregator(dedup_threshold: float = 0.95) -> ResultAggregator:
    """
    Get or create result aggregator singleton.

    Args:
        dedup_threshold: Deduplication threshold

    Returns:
        ResultAggregator instance
    """
    global _aggregator_instance
    if _aggregator_instance is None:
        _aggregator_instance = ResultAggregator(dedup_threshold)
    return _aggregator_instance
