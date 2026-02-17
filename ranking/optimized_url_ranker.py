"""
Intent Engine - Optimized URL Ranking Module

This module provides an efficient URL ranking system with:
- Parallel processing with asyncio and thread pools
- Persistent caching for URL metadata
- Batch embedding computation
- Better privacy scoring
- Configurable ranking weights
"""

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np
from config.optimized_cache import get_embedding_cache
from config.query_cache import get_url_analysis_cache

from core.schema import (
    Constraint,
    ConstraintType,
    DeclaredIntent,
    EthicalDimension,
    Frequency,
    InferredIntent,
    Recency,
    SkillLevel,
    TemporalHorizon,
    UniversalIntent,
    UseCase,
)

logger = logging.getLogger(__name__)


@dataclass
class URLResult:
    """Represents a URL with its metadata and scores"""

    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    favicon: Optional[str] = None

    # Privacy-focused attributes
    privacy_score: float = 0.5
    tracker_count: int = 0
    cookie_policy: Optional[str] = None
    data_retention_days: Optional[int] = None
    encryption_enabled: bool = True

    # Content attributes
    content_type: Optional[str] = None
    language: Optional[str] = None
    recency: Optional[str] = None

    # Ethics and quality
    is_open_source: bool = False
    is_non_profit: bool = False
    has_terms_of_service: bool = True

    # Scores
    relevance_score: float = 0.0
    final_score: float = 0.0

    # Processing metadata
    processing_time_ms: float = 0.0
    cache_hit: bool = False


@dataclass
class URLRankingRequest:
    """Request for URL ranking"""

    query: str
    urls: List[str]
    intent: Optional[UniversalIntent] = None
    options: Optional[Dict[str, Any]] = None


@dataclass
class URLRankingResponse:
    """Response from URL ranking"""

    query: str
    ranked_urls: List[URLResult]
    processing_time_ms: float
    total_urls: int
    filtered_count: int = 0
    cache_hit_rate: float = 0.0


class PrivacyDatabase:
    """Database of known privacy-friendly and big-tech domains"""

    # Privacy-focused domains (high scores)
    PRIVACY_FRIENDLY_DOMAINS = {
        # Email services
        "protonmail.com",
        "tutanota.com",
        "mailbox.org",
        "startmail.com",
        # Search engines
        "duckduckgo.com",
        "startpage.com",
        "searx.space",
        "qwant.com",
        # Browsers
        "firefox.com",
        "mozilla.org",
        "brave.com",
        "torproject.org",
        # VPN/Security
        "mullvad.net",
        "ivpn.net",
        "protonvpn.com",
        "nordvpn.com",
        # Privacy tools
        "privacytools.io",
        "privacyguides.org",
        "ssd.eff.org",
        # Open source platforms
        "github.com",
        "gitlab.com",
        "codeberg.org",
        "sourcehut.org",
        # Knowledge bases
        "wikipedia.org",
        "wikimedia.org",
        # Linux distributions
        "ubuntu.com",
        "debian.org",
        "archlinux.org",
        "fedoraproject.org",
    }

    # Big Tech domains (lower scores)
    BIG_TECH_DOMAINS = {
        # Google
        "google.com",
        "google.co",
        "youtube.com",
        "gmail.com",
        "android.com",
        "chrome.com",
        "googleapis.com",
        "googleusercontent.com",
        # Microsoft
        "microsoft.com",
        "windows.com",
        "office.com",
        "outlook.com",
        "hotmail.com",
        "live.com",
        "msn.com",
        "bing.com",
        # Apple
        "apple.com",
        "icloud.com",
        "me.com",
        "mac.com",
        # Meta
        "facebook.com",
        "instagram.com",
        "whatsapp.com",
        "messenger.com",
        # Amazon
        "amazon.com",
        "amazon.co",
        "aws.amazon.com",
        "alexa.com",
        # Twitter/X
        "twitter.com",
        "x.com",
        "tweetdeck.com",
        # LinkedIn (Microsoft)
        "linkedin.com",
        "licdn.com",
        # TikTok
        "tiktok.com",
        "tiktokv.com",
    }

    # Known tracker domains
    TRACKER_DOMAINS = {
        "google-analytics.com",
        "googletagmanager.com",
        "googleadservices.com",
        "doubleclick.net",
        "googlesyndication.com",
        "googleapis.com",
        "facebook.net",
        "facebook.com",
        "fbcdn.net",
        "amazon-adsystem.com",
        "amazon.com",
        "twitter.com",
        "twimg.com",
        "linkedin.com",
        "licdn.com",
        "outbrain.com",
        "taboola.com",
        "criteo.com",
        "criteo.net",
        "adnxs.com",
        "appnexus.com",
        "pubmatic.com",
        "rubiconproject.com",
        "openx.net",
        "openx.com",
        "scorecardresearch.com",
        "comscore.com",
        "quantserve.com",
        "quantcount.com",
        "moatads.com",
        "moatpixel.com",
    }

    @classmethod
    def get_privacy_score(cls, domain: str) -> float:
        """Get privacy score for a domain (0-1)"""
        domain_lower = domain.lower()

        # Check exact match
        if domain_lower in cls.PRIVACY_FRIENDLY_DOMAINS:
            return 0.95

        if domain_lower in cls.BIG_TECH_DOMAINS:
            return 0.3

        # Check subdomains
        for privacy_domain in cls.PRIVACY_FRIENDLY_DOMAINS:
            if domain_lower.endswith("." + privacy_domain):
                return 0.9

        for big_tech in cls.BIG_TECH_DOMAINS:
            if domain_lower.endswith("." + big_tech):
                return 0.25

        # Check for tracker subdomains
        for tracker in cls.TRACKER_DOMAINS:
            if tracker in domain_lower:
                return 0.1

        # Default score
        return 0.5

    @classmethod
    def is_big_tech(cls, domain: str) -> bool:
        """Check if domain belongs to Big Tech"""
        domain_lower = domain.lower()

        if domain_lower in cls.BIG_TECH_DOMAINS:
            return True

        for big_tech in cls.BIG_TECH_DOMAINS:
            if domain_lower.endswith("." + big_tech):
                return True

        return False

    @classmethod
    def count_trackers(cls, domain: str) -> int:
        """Count known trackers in domain"""
        domain_lower = domain.lower()
        count = 0

        for tracker in cls.TRACKER_DOMAINS:
            if tracker in domain_lower:
                count += 1

        return count


class URLAnalyzer:
    """Analyzes URLs for privacy and content metadata"""

    def __init__(self):
        self.cache = get_url_analysis_cache()
        self.embedding_cache = get_embedding_cache()

    def analyze_url(self, url: str, use_cache: bool = True) -> URLResult:
        """Analyze a URL and return metadata"""
        # Check cache
        if use_cache:
            cached = self.cache.get(url)
            if cached is not None:
                result = cached
                result.cache_hit = True
                return result

        # Parse URL
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Create result
        result = URLResult(
            url=url,
            domain=domain,
            title=self._extract_title(domain),
            description=f"Web resource from {domain}",
            privacy_score=PrivacyDatabase.get_privacy_score(domain),
            tracker_count=PrivacyDatabase.count_trackers(domain),
            encryption_enabled=url.startswith("https"),
            content_type=self._detect_content_type(url),
            is_open_source=self._is_open_source(domain),
            is_non_profit=self._is_non_profit(domain),
            cache_hit=False,
        )

        # Cache result
        if use_cache:
            self.cache.set(url, result, ttl_seconds=3600)  # 1 hour TTL

        return result

    def _extract_title(self, domain: str) -> str:
        """Extract a readable title from domain"""
        # Remove TLD and common prefixes
        parts = domain.split(".")
        if len(parts) >= 2:
            name = parts[-2]  # Second-level domain
            # Clean up common patterns
            name = name.replace("-", " ").replace("_", " ")
            return name.title()
        return domain

    def _detect_content_type(self, url: str) -> str:
        """Detect content type from URL patterns"""
        url_lower = url.lower()

        patterns = {
            "documentation": ["/docs/", "/documentation/", "/wiki/", "/help/", "/guide/", "/manual/"],
            "article": ["/blog/", "/article/", "/post/", "/news/", "/story/"],
            "forum": ["/forum/", "/community/", "/discuss/", "/thread/", "/topic/"],
            "video": ["/video/", "/watch/", "/v/", "/youtube.com/", "/vimeo.com/"],
            "image": ["/image/", "/photo/", "/gallery/", "/img/", "/pic/"],
            "tool": ["/tool/", "/app/", "/calculator/", "/converter/", "/generator/"],
            "download": ["/download/", "/file/", "/software/", "/install/"],
        }

        for content_type, patterns_list in patterns.items():
            if any(pattern in url_lower for pattern in patterns_list):
                return content_type

        return "general"

    def _is_open_source(self, domain: str) -> bool:
        """Check if domain is likely open source"""
        open_source_indicators = [
            "github.com",
            "gitlab.com",
            "codeberg.org",
            "sourcehut.org",
            "apache.org",
            "mozilla.org",
            "gnu.org",
            "linux.org",
            "debian.org",
            "ubuntu.com",
            "archlinux.org",
            "fedoraproject.org",
        ]
        return any(indicator in domain for indicator in open_source_indicators)

    def _is_non_profit(self, domain: str) -> bool:
        """Check if domain is likely non-profit"""
        non_profit_indicators = [
            ".org",
            "wikipedia.org",
            "wikimedia.org",
            "eff.org",
            "mozilla.org",
            "apache.org",
            "gnu.org",
        ]
        return any(indicator in domain for indicator in non_profit_indicators)


class URLRanker:
    """Main URL ranking class with parallel processing"""

    def __init__(self):
        self.analyzer = URLAnalyzer()
        self.embedding_cache = get_embedding_cache()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.default_weights = {"relevance": 0.40, "privacy": 0.30, "quality": 0.20, "ethics": 0.10}

    async def rank_urls(self, request: URLRankingRequest) -> URLRankingResponse:
        """Rank URLs with parallel processing"""
        start_time = time.time()

        # Get options
        options = request.options or {}
        weights = options.get("weights", self.default_weights)
        min_privacy_score = options.get("min_privacy_score", 0.0)
        exclude_big_tech = options.get("exclude_big_tech", False)

        # Analyze all URLs in parallel
        url_results = await self._analyze_urls_parallel(request.urls)

        # Calculate cache hit rate
        cache_hits = sum(1 for r in url_results if r.cache_hit)
        cache_hit_rate = cache_hits / len(url_results) if url_results else 0

        # Filter URLs
        filtered_results = self._filter_urls(url_results, min_privacy_score, exclude_big_tech)
        filtered_count = len(url_results) - len(filtered_results)

        # Calculate relevance scores
        await self._calculate_relevance_scores(filtered_results, request.query)

        # Calculate final scores
        self._calculate_final_scores(filtered_results, weights, request.intent)

        # Sort by final score
        filtered_results.sort(key=lambda x: x.final_score, reverse=True)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        return URLRankingResponse(
            query=request.query,
            ranked_urls=filtered_results,
            processing_time_ms=processing_time,
            total_urls=len(request.urls),
            filtered_count=filtered_count,
            cache_hit_rate=cache_hit_rate,
        )

    async def _analyze_urls_parallel(self, urls: List[str]) -> List[URLResult]:
        """Analyze multiple URLs in parallel"""
        # Create tasks for parallel execution
        tasks = [
            asyncio.get_running_loop().run_in_executor(self.executor, self.analyzer.analyze_url, url) for url in urls
        ]

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to analyze URL {urls[i]}: {result}")
                # Create fallback result
                valid_results.append(URLResult(url=urls[i]))
            else:
                valid_results.append(result)

        return valid_results

    def _filter_urls(
        self, results: List[URLResult], min_privacy_score: float, exclude_big_tech: bool
    ) -> List[URLResult]:
        """Filter URLs based on criteria"""
        filtered = []

        for result in results:
            # Check minimum privacy score (handle None case)
            if result.privacy_score is not None and result.privacy_score < min_privacy_score:
                continue

            # Check Big Tech exclusion
            if exclude_big_tech and PrivacyDatabase.is_big_tech(result.domain):
                continue

            filtered.append(result)

        return filtered

    async def _calculate_relevance_scores(self, results: List[URLResult], query: str) -> None:
        """Calculate semantic relevance scores for all URLs"""
        if not results or not query:
            return

        # Get query embedding
        query_emb = self.embedding_cache.encode_text(query)
        if query_emb is None:
            # Fallback to simple keyword matching
            for result in results:
                result.relevance_score = self._keyword_relevance(query, result)
            return

        # Prepare content texts
        contents = [f"{result.title} {result.description} {result.domain}" for result in results]

        # Get embeddings in batch
        content_embs = self.embedding_cache.encode_batch(contents)

        # Calculate similarities
        for i, (result, content_emb) in enumerate(zip(results, content_embs)):
            if content_emb is not None:
                similarity = self.embedding_cache.cosine_similarity(query_emb, content_emb)
                result.relevance_score = (similarity + 1) / 2  # Normalize to 0-1
            else:
                result.relevance_score = self._keyword_relevance(query, result)

    def _keyword_relevance(self, query: str, result: URLResult) -> float:
        """Calculate keyword-based relevance"""
        query_words = set(query.lower().split())
        content = f"{result.title} {result.description} {result.domain}".lower()

        if not query_words:
            return 0.5

        matches = query_words.intersection(set(content.split()))
        return len(matches) / len(query_words)

    def _calculate_final_scores(
        self, results: List[URLResult], weights: Dict[str, float], intent: Optional[UniversalIntent]
    ) -> None:
        """Calculate final weighted scores"""
        for result in results:
            # Base scores
            scores = {
                "relevance": result.relevance_score,
                "privacy": result.privacy_score,
                "quality": 0.7 if result.encryption_enabled else 0.3,
                "ethics": self._calculate_ethics_score(result),
            }

            # Apply intent-based boosts
            if intent:
                scores = self._apply_intent_boosts(scores, result, intent)

            # Calculate weighted average
            total_weight = sum(weights.values())
            result.final_score = (
                sum(scores[key] * weights.get(key, 0) for key in scores) / total_weight if total_weight > 0 else 0
            )

    def _calculate_ethics_score(self, result: URLResult) -> float:
        """Calculate ethics score"""
        score = 0.5

        if result.is_open_source:
            score += 0.3

        if result.is_non_profit:
            score += 0.1

        if result.privacy_score > 0.7:
            score += 0.1

        return min(1.0, score)

    def _apply_intent_boosts(
        self, scores: Dict[str, float], result: URLResult, intent: UniversalIntent
    ) -> Dict[str, float]:
        """Apply intent-based score boosts"""
        # Check for privacy-focused queries
        if intent.declared.query:
            query_lower = intent.declared.query.lower()
            privacy_keywords = ["privacy", "secure", "encrypted", "anonymous"]

            if any(kw in query_lower for kw in privacy_keywords):
                # Boost privacy score for privacy queries
                scores["privacy"] = min(1.0, scores["privacy"] * 1.2)

        # Check ethical signals
        if intent.inferred.ethicalSignals:
            for signal in intent.inferred.ethicalSignals:
                if signal.dimension == EthicalDimension.PRIVACY:
                    scores["privacy"] = min(1.0, scores["privacy"] * 1.1)
                elif signal.dimension == EthicalDimension.OPENNESS:
                    if result.is_open_source:
                        scores["ethics"] = min(1.0, scores["ethics"] * 1.2)

        return scores


# Global instance
_url_ranker_instance = None


def get_url_ranker() -> URLRanker:
    """Get singleton instance"""
    global _url_ranker_instance
    if _url_ranker_instance is None:
        _url_ranker_instance = URLRanker()
    return _url_ranker_instance


async def rank_urls(request: URLRankingRequest) -> URLRankingResponse:
    """Main entry point"""
    ranker = get_url_ranker()
    return await ranker.rank_urls(request)
