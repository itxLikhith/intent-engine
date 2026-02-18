"""
Intent Engine - URL Ranking Module

This module implements an efficient URL ranking system for a privacy-focused
search engine. It processes URLs in parallel, scores them based on relevance,
privacy compliance, and ethical alignment.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import numpy as np

from core.schema import Constraint, ConstraintType, EthicalDimension, UniversalIntent
from ranking.ranker import EmbeddingCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class URLResult:
    """Represents a URL with its metadata and scores"""

    url: str
    title: str | None = None
    description: str | None = None
    domain: str | None = None
    favicon: str | None = None

    # Privacy-focused attributes
    privacy_score: float = 0.5  # 0-1: How privacy-friendly
    tracker_count: int = 0  # Number of known trackers
    cookie_policy: str | None = None  # "minimal", "strict", "lenient"
    data_retention_days: int | None = None
    encryption_enabled: bool = True  # HTTPS support

    # Content attributes
    content_type: str | None = None  # "article", "tool", "forum", "docs", etc.
    language: str | None = None
    recency: str | None = None  # ISO date string

    # Quality attributes
    quality_score: float = 0.5
    authority_score: float = 0.5  # Domain authority
    engagement_score: float = 0.5

    # Ethical attributes
    is_open_source: bool = False
    is_non_profit: bool = False
    advertising_policy: str | None = None  # "no_ads", "minimal_ads", "ad_supported"
    ethical_tags: list[str] = field(default_factory=list)

    # Computed scores
    relevance_score: float = 0.0
    final_score: float = 0.0


@dataclass
class URLRankingRequest:
    """Request for ranking URLs"""

    query: str  # The search query
    urls: list[str]  # List of URLs to rank
    intent: UniversalIntent | None = None  # Optional structured intent
    options: dict[str, Any] | None = None  # Ranking options


@dataclass
class URLRankingResponse:
    """Response with ranked URLs"""

    query: str
    ranked_urls: list[URLResult]
    processing_time_ms: float
    total_urls: int
    filtered_count: int = 0


@dataclass
class URLAnalysisResult:
    """Result of analyzing a single URL"""

    url: str
    domain: str
    title: str
    description: str
    privacy_score: float
    content_type: str
    quality_indicators: dict[str, Any]
    error: str | None = None


class PrivacyDatabase:
    """Database of known privacy characteristics for domains"""

    # Known privacy-friendly domains (open source, privacy-focused, non-profit)
    PRIVACY_FRIENDLY_DOMAINS = {
        # Privacy-focused email/services
        "protonmail.com": {
            "privacy_score": 0.95,
            "tracker_count": 0,
            "encryption": True,
            "type": "service",
            "open_source": True,
        },
        "proton.me": {
            "privacy_score": 0.95,
            "tracker_count": 0,
            "encryption": True,
            "type": "service",
            "open_source": True,
        },
        "tutanota.com": {
            "privacy_score": 0.95,
            "tracker_count": 0,
            "encryption": True,
            "type": "service",
            "open_source": True,
        },
        "tuta.io": {
            "privacy_score": 0.95,
            "tracker_count": 0,
            "encryption": True,
            "type": "service",
            "open_source": True,
        },
        # Privacy-focused search
        "duckduckgo.com": {
            "privacy_score": 0.9,
            "tracker_count": 0,
            "encryption": True,
            "type": "search",
            "open_source": False,
        },
        "startpage.com": {
            "privacy_score": 0.85,
            "tracker_count": 1,
            "encryption": True,
            "type": "search",
            "open_source": False,
        },
        "searx.org": {
            "privacy_score": 0.95,
            "tracker_count": 0,
            "encryption": True,
            "type": "search",
            "open_source": True,
        },
        "searxng.org": {
            "privacy_score": 0.95,
            "tracker_count": 0,
            "encryption": True,
            "type": "search",
            "open_source": True,
        },
        # Open source platforms
        "github.com": {
            "privacy_score": 0.8,
            "tracker_count": 1,
            "encryption": True,
            "type": "code",
            "open_source": True,
        },
        "gitlab.com": {
            "privacy_score": 0.85,
            "tracker_count": 0,
            "encryption": True,
            "type": "code",
            "open_source": True,
        },
        "sourceforge.net": {
            "privacy_score": 0.75,
            "tracker_count": 2,
            "encryption": True,
            "type": "code",
            "open_source": True,
        },
        # Privacy-focused browsers
        "brave.com": {
            "privacy_score": 0.9,
            "tracker_count": 0,
            "encryption": True,
            "type": "software",
            "open_source": True,
        },
        "mozilla.org": {
            "privacy_score": 0.85,
            "tracker_count": 1,
            "encryption": True,
            "type": "software",
            "open_source": True,
        },
        # Wikipedia and educational
        "wikipedia.org": {
            "privacy_score": 0.9,
            "tracker_count": 0,
            "encryption": True,
            "type": "encyclopedia",
            "open_source": True,
            "non_profit": True,
        },
        "wikimedia.org": {
            "privacy_score": 0.9,
            "tracker_count": 0,
            "encryption": True,
            "type": "encyclopedia",
            "open_source": True,
            "non_profit": True,
        },
        # Privacy-focused VPNs and tools
        "mullvad.net": {
            "privacy_score": 0.95,
            "tracker_count": 0,
            "encryption": True,
            "type": "vpn",
            "open_source": True,
        },
        "nordvpn.com": {
            "privacy_score": 0.8,
            "tracker_count": 1,
            "encryption": True,
            "type": "vpn",
            "open_source": False,
        },
        # FOSS alternatives
        "fsf.org": {
            "privacy_score": 0.9,
            "tracker_count": 0,
            "encryption": True,
            "type": "organization",
            "open_source": True,
            "non_profit": True,
        },
        "opensource.org": {
            "privacy_score": 0.9,
            "tracker_count": 0,
            "encryption": True,
            "type": "organization",
            "open_source": True,
            "non_profit": True,
        },
    }

    # Known tracker domains
    TRACKER_DOMAINS = {
        "google.com",
        "facebook.com",
        "twitter.com",
        "linkedin.com",
        "doubleclick.net",
        "google-analytics.com",
        "googletagmanager.com",
        "facebook.net",
        "hotjar.com",
        "segment.io",
        "mixpanel.com",
        "criteo.com",
        "taboola.com",
        "outbrain.com",
        "amazon-adsystem.com",
    }

    # Big tech domains (often excluded in privacy-focused searches)
    BIG_TECH_DOMAINS = {
        "google.com",
        "googleusercontent.com",
        "googledrive.com",
        "facebook.com",
        "instagram.com",
        "whatsapp.com",
        "messenger.com",
        "microsoft.com",
        "outlook.com",
        "live.com",
        "onedrive.com",
        "amazon.com",
        "aws.amazon.com",
        "apple.com",
        "icloud.com",
        "twitter.com",
        "x.com",
        "tiktok.com",
        "bytedance.com",
    }

    # Content type patterns
    CONTENT_TYPE_PATTERNS = {
        "article": ["blog", "news", "article", "post", "story"],
        "documentation": ["docs", "documentation", "wiki", "manual", "guide"],
        "tool": ["tool", "generator", "converter", "calculator", "checker"],
        "forum": ["forum", "discussion", "community", "board"],
        "code": ["github", "gitlab", "sourceforge", "code", "repository"],
        "download": ["download", "release", "file"],
    }

    def get_domain_info(self, url: str) -> dict[str, Any]:
        """Get privacy and content info for a domain"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Check exact match first
        if domain in self.PRIVACY_FRIENDLY_DOMAINS:
            return self.PRIVACY_FRIENDLY_DOMAINS[domain].copy()

        # Check for partial matches (subdomains)
        for known_domain, info in self.PRIVACY_FRIENDLY_DOMAINS.items():
            if domain.endswith("." + known_domain):
                result = info.copy()
                result["subdomain"] = True
                return result

        # Check if it's a tracker domain
        is_tracker = any(
            domain.endswith("." + td) or domain == td for td in self.TRACKER_DOMAINS
        )

        # Check if it's big tech
        is_big_tech = any(
            domain.endswith("." + bt) or domain == bt for bt in self.BIG_TECH_DOMAINS
        )

        # Default values based on domain analysis
        privacy_score = 0.3 if is_big_tech else 0.5
        tracker_count = 3 if is_big_tech else (1 if is_tracker else 0)

        return {
            "privacy_score": privacy_score,
            "tracker_count": tracker_count,
            "encryption": True,  # Assume HTTPS
            "type": "unknown",
            "open_source": False,
            "is_big_tech": is_big_tech,
            "is_tracker": is_tracker,
        }


class URLAnalyzer:
    """Analyzes individual URLs for privacy and content characteristics"""

    def __init__(self, embedding_cache: EmbeddingCache | None = None):
        self.privacy_db = PrivacyDatabase()
        self.embedding_cache = embedding_cache or EmbeddingCache()

    def analyze_url(self, url: str) -> URLAnalysisResult:
        """Analyze a single URL and return its characteristics"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Get domain info from privacy database
            domain_info = self.privacy_db.get_domain_info(url)

            # Determine content type from URL path
            path = parsed.path.lower()
            content_type = self._determine_content_type(path, domain)

            # Extract title from path (fallback)
            title = self._extract_title_from_path(path, domain)

            # Generate description from URL structure
            description = self._generate_description(url, content_type)

            return URLAnalysisResult(
                url=url,
                domain=domain,
                title=title,
                description=description,
                privacy_score=domain_info.get("privacy_score", 0.5),
                content_type=content_type,
                quality_indicators={
                    "encryption": domain_info.get("encryption", True),
                    "tracker_count": domain_info.get("tracker_count", 0),
                    "is_open_source": domain_info.get("open_source", False),
                    "is_non_profit": domain_info.get("non_profit", False),
                    "is_big_tech": domain_info.get("is_big_tech", False),
                },
            )
        except Exception as e:
            logger.error(f"Error analyzing URL {url}: {e}")
            return URLAnalysisResult(
                url=url,
                domain="unknown",
                title=url,
                description="Error analyzing URL",
                privacy_score=0.5,
                content_type="unknown",
                quality_indicators={},
                error=str(e),
            )

    def _determine_content_type(self, path: str, domain: str) -> str:
        """Determine content type from URL path"""
        combined = f"{domain} {path}"

        for content_type, patterns in PrivacyDatabase.CONTENT_TYPE_PATTERNS.items():
            if any(p in combined for p in patterns):
                return content_type

        # Default based on domain
        if "blog" in domain or "news" in domain:
            return "article"
        elif "docs" in domain or "wiki" in domain:
            return "documentation"
        elif "forum" in domain or "community" in domain:
            return "forum"

        return "general"

    def _extract_title_from_path(self, path: str, domain: str) -> str:
        """Extract a readable title from URL path"""
        # Remove leading/trailing slashes and split
        parts = [p for p in path.strip("/").split("/") if p]

        if not parts:
            return domain

        # Use last meaningful part
        last_part = parts[-1] if parts else domain

        # Clean up: replace hyphens/underscores with spaces, capitalize
        title = re.sub(r"[-_]", " ", last_part)
        title = re.sub(r"\b\w\b", "", title)  # Remove single chars
        title = title.strip()

        if not title:
            return domain

        return title.title()

    def _generate_description(self, url: str, content_type: str) -> str:
        """Generate a description based on URL and content type"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        type_descriptions = {
            "article": "Article and blog content",
            "documentation": "Documentation and guides",
            "tool": "Online tool and utilities",
            "forum": "Community discussion forum",
            "code": "Source code repository",
            "download": "Software downloads",
            "general": "Web resource",
        }

        return f"{type_descriptions.get(content_type, 'Web resource')} from {domain}"


class URLRanker:
    """
    Main URL ranking engine for privacy-focused search.

    Processes URLs efficiently using parallel processing and scores them based on:
    - Query relevance (semantic similarity)
    - Privacy compliance (minimal tracking, encryption)
    - Ethical alignment (open source, non-profit)
    - Content quality (authority, recency)
    """

    # Default scoring weights
    DEFAULT_WEIGHTS = {
        "relevance": 0.35,  # Query-content similarity
        "privacy": 0.25,  # Privacy-friendly
        "quality": 0.20,  # Content quality
        "ethics": 0.20,  # Ethical alignment
    }

    def __init__(self):
        self.embedding_cache = EmbeddingCache()
        self.url_analyzer = URLAnalyzer(self.embedding_cache)

    async def rank_urls(self, request: URLRankingRequest) -> URLRankingResponse:
        """
        Rank URLs based on query and optional intent.

        Processes URLs in parallel for efficiency.
        """
        start_time = time.time()

        urls = request.urls
        query = request.query
        intent = request.intent

        # Get ranking options
        options = request.options or {}
        weights = options.get("weights", self.DEFAULT_WEIGHTS)
        min_privacy_score = options.get("min_privacy_score", 0.0)
        exclude_big_tech = options.get("exclude_big_tech", False)

        # Parse intent if provided
        privacy_preference = None
        ethical_signals = []
        constraints = []

        if intent:
            if intent.inferred and intent.inferred.ethicalSignals:
                ethical_signals = intent.inferred.ethicalSignals
            if intent.declared:
                constraints = intent.declared.constraints or []
                if intent.declared.negativePreferences:
                    privacy_preference = intent.declared.negativePreferences

        # Step 1: Analyze all URLs in parallel
        analyzed_urls = await self._analyze_urls_parallel(urls)

        # Step 2: Filter URLs based on constraints
        filtered_urls = self._apply_filters(
            analyzed_urls,
            constraints,
            min_privacy_score,
            exclude_big_tech,
            privacy_preference,
        )

        # Step 3: Score remaining URLs
        scored_urls = await self._score_urls(
            filtered_urls, query, weights, ethical_signals
        )

        # Step 4: Sort by final score (handle None values)
        scored_urls.sort(
            key=lambda x: x.final_score if x.final_score is not None else 0.0,
            reverse=True,
        )

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return URLRankingResponse(
            query=query,
            ranked_urls=scored_urls,
            processing_time_ms=processing_time,
            total_urls=len(urls),
            filtered_count=len(urls) - len(scored_urls),
        )

    async def _analyze_urls_parallel(self, urls: list[str]) -> list[URLResult]:
        """Analyze URLs in parallel using asyncio to_thread"""

        async def analyze_single(url: str) -> URLResult:
            """Analyze a single URL in a thread pool"""
            analysis = await asyncio.to_thread(self.url_analyzer.analyze_url, url)

            return URLResult(
                url=analysis.url,
                title=analysis.title,
                description=analysis.description,
                domain=analysis.domain,
                privacy_score=analysis.privacy_score,
                tracker_count=analysis.quality_indicators.get("tracker_count", 0),
                encryption_enabled=analysis.quality_indicators.get("encryption", True),
                content_type=analysis.content_type,
                is_open_source=analysis.quality_indicators.get("is_open_source", False),
                is_non_profit=analysis.quality_indicators.get("is_non_profit", False),
                quality_score=0.5,
                authority_score=self._calculate_authority(analysis.domain),
            )

        # Run all URL analyses concurrently
        tasks = [analyze_single(url) for url in urls]
        results = await asyncio.gather(*tasks)

        return list(results)

    def _apply_filters(
        self,
        urls: list[URLResult],
        constraints: list[Constraint],
        min_privacy_score: float,
        exclude_big_tech: bool,
        negative_preferences: list[str] | None,
    ) -> list[URLResult]:
        """Apply filtering constraints to URLs"""
        filtered = []

        for url_result in urls:
            # Apply hard constraints
            if not self._satisfies_constraints(url_result, constraints):
                continue

            # Apply minimum privacy score
            if (
                min_privacy_score is not None
                and url_result.privacy_score < min_privacy_score
            ):
                continue

            # Exclude big tech if requested
            if exclude_big_tech:
                domain_info = PrivacyDatabase().get_domain_info(url_result.url)
                if domain_info.get("is_big_tech", False):
                    continue

            # Apply negative preferences (e.g., "no google")
            if negative_preferences:
                if not self._satisfies_negative_preferences(
                    url_result, negative_preferences
                ):
                    continue

            filtered.append(url_result)

        return filtered

    def _satisfies_constraints(
        self, url_result: URLResult, constraints: list[Constraint]
    ) -> bool:
        """Check if URL satisfies hard constraints"""
        for constraint in constraints:
            if not constraint.hardFilter:
                continue

            dimension = constraint.dimension
            value = constraint.value

            if dimension == "domain" or dimension == "provider":
                if constraint.type == ConstraintType.INCLUSION:
                    if isinstance(value, list):
                        if not any(
                            v.lower() in url_result.domain.lower() for v in value
                        ):
                            return False
                    elif isinstance(value, str):
                        if value.lower() not in url_result.domain.lower():
                            return False
                elif constraint.type == ConstraintType.EXCLUSION:
                    if isinstance(value, list):
                        if any(v.lower() in url_result.domain.lower() for v in value):
                            return False
                    elif isinstance(value, str):
                        if value.lower() in url_result.domain.lower():
                            return False

            elif dimension == "privacy":
                if constraint.type == ConstraintType.INCLUSION:
                    if isinstance(value, str) and url_result.privacy_score is not None:
                        violates = False
                        if "high" in value.lower() and url_result.privacy_score < 0.7:
                            violates = True
                        elif (
                            "medium" in value.lower() and url_result.privacy_score < 0.5
                        ):
                            violates = True
                        if violates:
                            return False

        return True

    def _satisfies_negative_preferences(
        self, url_result: URLResult, preferences: list[str]
    ) -> bool:
        """Check if URL satisfies negative preferences like 'no google'"""
        for pref in preferences:
            pref_lower = pref.lower().replace("no ", "").replace("not ", "")

            # Check against big tech domains
            privacy_db = PrivacyDatabase()
            for big_tech in privacy_db.BIG_TECH_DOMAINS:
                if big_tech.replace(".com", "").replace(".", " ") in pref_lower:
                    if big_tech in url_result.domain or url_result.domain.endswith(
                        "." + big_tech
                    ):
                        return False

        return True

    async def _score_urls(
        self,
        urls: list[URLResult],
        query: str,
        weights: dict[str, float],
        ethical_signals: list[Any],
    ) -> list[URLResult]:
        """Score URLs based on multiple factors using batch processing for efficiency"""
        import asyncio

        # Get query embedding for relevance calculation
        query_embedding = self.embedding_cache.encode_text(query)

        # Prepare all content texts for batch embedding
        if query_embedding is not None:
            contents = []
            url_indices = []
            for i, url in enumerate(urls):
                content = f"{url.title or ''} {url.description or ''} {url.content_type or ''}"
                if content.strip():
                    contents.append(content)
                    url_indices.append(i)

            # Batch encode all contents at once (much faster than individual encoding)
            if contents:
                content_embeddings = await asyncio.to_thread(
                    self.embedding_cache.encode_batch, contents
                )

                # Calculate similarities for all URLs
                for idx, url_idx in enumerate(url_indices):
                    embedding = content_embeddings[idx]
                    if embedding is not None:
                        similarity = self.embedding_cache.cosine_similarity(
                            query_embedding, embedding
                        )
                        urls[url_idx].relevance_score = (similarity + 1) / 2
                    else:
                        # Fallback to keyword matching
                        urls[url_idx].relevance_score = (
                            self._calculate_keyword_relevance(urls[url_idx], query)
                        )

        # Ensure weights is not None
        if weights is None:
            weights = self.DEFAULT_WEIGHTS

        # Process remaining URLs that weren't batch encoded
        for url in urls:
            if url.relevance_score == 0.0:  # Not yet scored
                url.relevance_score = self._calculate_relevance(
                    url, query, query_embedding
                )

            # 2. Privacy score (already computed during analysis)
            privacy = url.privacy_score if url.privacy_score is not None else 0.5

            # 3. Quality score
            quality = self._calculate_quality(url)

            # 4. Ethical score
            ethics = self._calculate_ethics(url, ethical_signals)

            # Calculate final weighted score
            url.final_score = (
                url.relevance_score * weights.get("relevance", 0.35)
                + privacy * weights.get("privacy", 0.25)
                + quality * weights.get("quality", 0.20)
                + ethics * weights.get("ethics", 0.20)
            )

        return urls

    def _calculate_keyword_relevance(self, url: URLResult, query: str) -> float:
        """Calculate relevance using keyword matching only"""
        content = f"{url.title or ''} {url.description or ''} {url.content_type or ''}"

        if not content.strip():
            return 0.3

        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.5

        matches = query_words.intersection(content_words)
        return min(1.0, len(matches) / len(query_words)) * 0.5 + 0.3

    def _calculate_relevance(
        self, url: URLResult, query: str, query_embedding: np.ndarray | None
    ) -> float:
        """Calculate relevance score between query and URL"""
        # Combine title and description for content
        content = f"{url.title or ''} {url.description or ''} {url.content_type or ''}"

        if not content.strip():
            return 0.3

        # Use embeddings if available
        if query_embedding is not None:
            content_embedding = self.embedding_cache.encode_text(content)
            if content_embedding is not None:
                similarity = self.embedding_cache.cosine_similarity(
                    query_embedding, content_embedding
                )
                # Normalize from [-1, 1] to [0, 1]
                return (similarity + 1) / 2

        # Fallback to keyword matching
        return self._calculate_keyword_relevance(url, query)

    def _calculate_quality(self, url: URLResult) -> float:
        """Calculate content quality score"""
        score = 0.5  # Base score

        # Boost for encryption
        if url.encryption_enabled:
            score += 0.1

        # Boost for good privacy score
        if url.privacy_score is not None and url.privacy_score > 0.7:
            score += 0.15
        elif url.privacy_score is not None and url.privacy_score > 0.5:
            score += 0.05

        # Boost for low tracker count
        if url.tracker_count == 0:
            score += 0.15
        elif url.tracker_count <= 2:
            score += 0.05

        # Penalize high tracker count
        if url.tracker_count > 5:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _calculate_ethics(self, url: URLResult, ethical_signals: list[Any]) -> float:
        """Calculate ethical alignment score"""
        score = 0.5  # Base score

        # Check ethical signals from intent
        if ethical_signals:
            for signal in ethical_signals:
                if signal.dimension == EthicalDimension.PRIVACY:
                    if url.privacy_score is not None and url.privacy_score > 0.7:
                        score += 0.2
                elif signal.dimension == EthicalDimension.OPENNESS:
                    if url.is_open_source:
                        score += 0.2

        # Boost for open source
        if url.is_open_source:
            score += 0.15

        # Boost for non-profit
        if url.is_non_profit:
            score += 0.1

        # Penalize ad-heavy domains
        if url.advertising_policy == "ad_supported":
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _calculate_authority(self, domain: str) -> float:
        """Calculate domain authority score"""
        # Known high-authority domains
        authority_domains = {
            "wikipedia.org": 0.95,
            "github.com": 0.9,
            "mozilla.org": 0.85,
            "stackoverflow.com": 0.85,
            "medium.com": 0.75,
            "dev.to": 0.7,
        }

        for auth_domain, score in authority_domains.items():
            if domain == auth_domain or domain.endswith("." + auth_domain):
                return score

        return 0.5  # Default


# Global instance for caching
_url_ranker_instance = None


def get_url_ranker() -> URLRanker:
    """Get singleton instance of URLRanker"""
    global _url_ranker_instance
    if _url_ranker_instance is None:
        _url_ranker_instance = URLRanker()
    return _url_ranker_instance


async def rank_urls(request: URLRankingRequest) -> URLRankingResponse:
    """Main entry point for URL ranking"""
    ranker = get_url_ranker()
    return await ranker.rank_urls(request)
