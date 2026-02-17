"""
Unit tests for the URL ranking module
"""

import asyncio
import unittest

from core.schema import (
    Constraint,
    ConstraintType,
    DeclaredIntent,
    EthicalDimension,
    EthicalSignal,
    InferredIntent,
    UniversalIntent,
)
from ranking.url_ranker import PrivacyDatabase, URLAnalyzer, URLRanker, URLRankingRequest, URLResult


class TestPrivacyDatabase(unittest.TestCase):
    """Test the PrivacyDatabase class"""

    def setUp(self):
        self.db = PrivacyDatabase()

    def test_privacy_friendly_domain(self):
        """Test that privacy-friendly domains are identified"""
        info = self.db.get_domain_info("https://protonmail.com")
        self.assertEqual(info["privacy_score"], 0.95)
        self.assertEqual(info["tracker_count"], 0)
        self.assertTrue(info["open_source"])

    def test_big_tech_domain(self):
        """Test that big tech domains are identified"""
        info = self.db.get_domain_info("https://google.com")
        self.assertLess(info["privacy_score"], 0.5)
        self.assertGreater(info["tracker_count"], 0)
        self.assertTrue(info["is_big_tech"])

    def test_tracker_domain(self):
        """Test that tracker domains are identified"""
        info = self.db.get_domain_info("https://doubleclick.net")
        self.assertTrue(info["is_tracker"])

    def test_wikipedia_domain(self):
        """Test Wikipedia is identified as non-profit"""
        info = self.db.get_domain_info("https://wikipedia.org")
        self.assertTrue(info["non_profit"])
        self.assertTrue(info["open_source"])

    def test_subdomain_matching(self):
        """Test that subdomains are matched correctly"""
        info = self.db.get_domain_info("https://mail.protonmail.com")
        self.assertGreater(info["privacy_score"], 0.8)

    def test_unknown_domain_defaults(self):
        """Test default values for unknown domains"""
        info = self.db.get_domain_info("https://example.com")
        self.assertEqual(info["privacy_score"], 0.5)
        self.assertFalse(info.get("is_big_tech", False))


class TestURLAnalyzer(unittest.TestCase):
    """Test the URLAnalyzer class"""

    def setUp(self):
        self.analyzer = URLAnalyzer()

    def test_analyze_protonmail(self):
        """Test analyzing ProtonMail URL"""
        result = self.analyzer.analyze_url("https://protonmail.com")
        self.assertEqual(result.domain, "protonmail.com")
        self.assertGreater(result.privacy_score, 0.9)
        self.assertEqual(result.quality_indicators.get("tracker_count", 0), 0)

    def test_analyze_google(self):
        """Test analyzing Google URL"""
        result = self.analyzer.analyze_url("https://google.com")
        self.assertEqual(result.domain, "google.com")
        self.assertLess(result.privacy_score, 0.5)
        self.assertGreater(result.quality_indicators.get("tracker_count", 0), 0)

    def test_content_type_detection(self):
        """Test content type detection from URL"""
        # Test documentation type
        result = self.analyzer.analyze_url("https://docs.example.com")
        self.assertEqual(result.content_type, "documentation")

        # Test code type
        result = self.analyzer.analyze_url("https://github.com/user/repo")
        self.assertEqual(result.content_type, "code")

        # Test forum type
        result = self.analyzer.analyze_url("https://forum.example.com")
        self.assertEqual(result.content_type, "forum")

    def test_title_extraction(self):
        """Test title extraction from URL path"""
        result = self.analyzer.analyze_url("https://example.com/blog/setting-up-email")
        self.assertIn("Setting Up Email", result.title)

    def test_www_prefix_removal(self):
        """Test that www prefix is removed from domain"""
        result = self.analyzer.analyze_url("https://www.protonmail.com")
        self.assertEqual(result.domain, "protonmail.com")


class TestURLRanker(unittest.TestCase):
    """Test the URLRanker class"""

    def setUp(self):
        self.ranker = URLRanker()

    def test_rank_25_urls(self):
        """Test ranking 25 URLs efficiently"""
        # Create 25 diverse test URLs
        urls = [
            "https://protonmail.com",
            "https://tutanota.com",
            "https://duckduckgo.com",
            "https://searx.org",
            "https://github.com",
            "https://gitlab.com",
            "https://brave.com",
            "https://mozilla.org",
            "https://wikipedia.org",
            "https://mullvad.net",
            "https://google.com",
            "https://facebook.com",
            "https://microsoft.com",
            "https://amazon.com",
            "https://apple.com",
            "https://twitter.com",
            "https://medium.com",
            "https://stackoverflow.com",
            "https://dev.to",
            "https://sourceforge.net",
            "https://reddit.com",
            "https://youtube.com",
            "https://linkedin.com",
            "https://instagram.com",
            "https://wikipedia.org/wiki/Test",
        ]

        request = URLRankingRequest(query="privacy focused email service", urls=urls)

        # Run the async ranking
        response = asyncio.run(self.ranker.rank_urls(request))

        # Verify results
        self.assertEqual(len(response.ranked_urls), 25)
        self.assertEqual(response.total_urls, 25)

        # Privacy-friendly domains should rank higher
        top_5_domains = [r.domain for r in response.ranked_urls[:5]]

        # At least some privacy domains should be in top 5
        privacy_domains_in_top = sum(
            1 for d in top_5_domains if d in ["protonmail.com", "tutanota.com", "duckduckgo.com", "searx.org"]
        )
        self.assertGreater(privacy_domains_in_top, 0)

        # Processing should be fast
        self.assertLess(response.processing_time_ms, 5000)

    def test_exclude_big_tech(self):
        """Test excluding big tech domains"""
        urls = [
            "https://protonmail.com",
            "https://google.com",
            "https://facebook.com",
            "https://tutanota.com",
        ]

        request = URLRankingRequest(query="email service", urls=urls, options={"exclude_big_tech": True})

        response = asyncio.run(self.ranker.rank_urls(request))

        # Big tech should be filtered out
        domains = [r.domain for r in response.ranked_urls]
        self.assertNotIn("google.com", domains)
        self.assertNotIn("facebook.com", domains)
        self.assertEqual(response.filtered_count, 2)

    def test_min_privacy_score_filter(self):
        """Test minimum privacy score filtering"""
        urls = [
            "https://protonmail.com",
            "https://google.com",
        ]

        request = URLRankingRequest(query="privacy", urls=urls, options={"min_privacy_score": 0.8})

        response = asyncio.run(self.ranker.rank_urls(request))

        # Only high privacy domains should remain
        self.assertEqual(len(response.ranked_urls), 1)
        self.assertEqual(response.ranked_urls[0].domain, "protonmail.com")

    def test_constraint_filtering(self):
        """Test constraint-based filtering"""
        intent = UniversalIntent(
            intentId="test-123",
            context={"product": "search"},
            declared=DeclaredIntent(
                query="code repository",
                constraints=[
                    Constraint(type=ConstraintType.INCLUSION, dimension="domain", value="github.com", hardFilter=True)
                ],
            ),
            inferred=InferredIntent(),
        )

        urls = [
            "https://github.com/user/project",
            "https://gitlab.com/user/project",
            "https://bitbucket.org/user/project",
        ]

        request = URLRankingRequest(query="code repository", urls=urls, intent=intent)

        response = asyncio.run(self.ranker.rank_urls(request))

        # Only github should remain
        self.assertEqual(len(response.ranked_urls), 1)
        self.assertEqual(response.ranked_urls[0].domain, "github.com")

    def test_exclusion_constraint(self):
        """Test exclusion constraints"""
        intent = UniversalIntent(
            intentId="test-123",
            context={"product": "search"},
            declared=DeclaredIntent(
                query="search engine",
                constraints=[
                    Constraint(type=ConstraintType.EXCLUSION, dimension="domain", value="google.com", hardFilter=True)
                ],
            ),
            inferred=InferredIntent(),
        )

        urls = [
            "https://duckduckgo.com",
            "https://google.com",
            "https://bing.com",
        ]

        request = URLRankingRequest(query="search engine", urls=urls, intent=intent)

        response = asyncio.run(self.ranker.rank_urls(request))

        # Google should be excluded
        domains = [r.domain for r in response.ranked_urls]
        self.assertNotIn("google.com", domains)
        self.assertEqual(len(response.ranked_urls), 2)

    def test_ethical_signals_boost(self):
        """Test that ethical signals boost privacy-friendly domains"""
        intent = UniversalIntent(
            intentId="test-123",
            context={"product": "search"},
            declared=DeclaredIntent(query="open source software"),
            inferred=InferredIntent(
                ethicalSignals=[
                    EthicalSignal(dimension=EthicalDimension.OPENNESS, preference="open-source_preferred"),
                    EthicalSignal(dimension=EthicalDimension.PRIVACY, preference="privacy-first"),
                ]
            ),
        )

        urls = [
            "https://github.com",
            "https://wikipedia.org",
            "https://medium.com",
        ]

        request = URLRankingRequest(query="open source software", urls=urls, intent=intent)

        response = asyncio.run(self.ranker.rank_urls(request))

        self.assertEqual(len(response.ranked_urls), 3)

    def test_scoring_weights(self):
        """Test custom scoring weights"""
        urls = [
            "https://protonmail.com",
            "https://google.com",
        ]

        request = URLRankingRequest(
            query="email",
            urls=urls,
            options={"weights": {"relevance": 0.1, "privacy": 0.6, "quality": 0.15, "ethics": 0.15}},
        )

        response = asyncio.run(self.ranker.rank_urls(request))

        # ProtonMail should rank higher due to privacy weight
        self.assertEqual(response.ranked_urls[0].domain, "protonmail.com")

    def test_empty_url_list(self):
        """Test with empty URL list"""
        request = URLRankingRequest(query="test", urls=[])

        response = asyncio.run(self.ranker.rank_urls(request))

        self.assertEqual(len(response.ranked_urls), 0)
        self.assertEqual(response.total_urls, 0)

    def test_single_url(self):
        """Test with single URL"""
        request = URLRankingRequest(query="privacy email", urls=["https://protonmail.com"])

        response = asyncio.run(self.ranker.rank_urls(request))

        self.assertEqual(len(response.ranked_urls), 1)
        self.assertEqual(response.ranked_urls[0].url, "https://protonmail.com")

    def test_scores_are_normalized(self):
        """Test that scores are properly normalized to 0-1 range"""
        urls = [
            "https://protonmail.com",
            "https://google.com",
            "https://wikipedia.org",
        ]

        request = URLRankingRequest(query="privacy", urls=urls)

        response = asyncio.run(self.ranker.rank_urls(request))

        for result in response.ranked_urls:
            self.assertGreaterEqual(result.final_score, 0.0)
            self.assertLessEqual(result.final_score, 1.0)
            self.assertGreaterEqual(result.relevance_score, 0.0)
            self.assertLessEqual(result.relevance_score, 1.0)
            self.assertGreaterEqual(result.privacy_score, 0.0)
            self.assertLessEqual(result.privacy_score, 1.0)

    def test_processing_time_tracked(self):
        """Test that processing time is tracked"""
        urls = [f"https://example{i}.com" for i in range(10)]

        request = URLRankingRequest(query="test", urls=urls)

        response = asyncio.run(self.ranker.rank_urls(request))

        self.assertGreater(response.processing_time_ms, 0)

    def test_negative_preferences(self):
        """Test negative preferences like 'no google'"""
        intent = UniversalIntent(
            intentId="test-123",
            context={"product": "search"},
            declared=DeclaredIntent(query="search", negativePreferences=["no google", "no facebook"]),
            inferred=InferredIntent(),
        )

        urls = [
            "https://duckduckgo.com",
            "https://google.com",
            "https://facebook.com",
            "https://startpage.com",
        ]

        request = URLRankingRequest(query="search", urls=urls, intent=intent)

        response = asyncio.run(self.ranker.rank_urls(request))

        # Google and Facebook should be filtered out
        domains = [r.domain for r in response.ranked_urls]
        self.assertNotIn("google.com", domains)
        self.assertNotIn("facebook.com", domains)


class TestURLResult(unittest.TestCase):
    """Test the URLResult dataclass"""

    def test_default_values(self):
        """Test default values for URLResult"""
        result = URLResult(url="https://example.com")

        self.assertEqual(result.url, "https://example.com")
        self.assertEqual(result.privacy_score, 0.5)
        self.assertEqual(result.quality_score, 0.5)
        self.assertEqual(result.tracker_count, 0)
        self.assertTrue(result.encryption_enabled)
        self.assertEqual(result.relevance_score, 0.0)
        self.assertEqual(result.final_score, 0.0)

    def test_custom_values(self):
        """Test custom values for URLResult"""
        result = URLResult(
            url="https://protonmail.com",
            title="ProtonMail",
            description="Secure email",
            domain="protonmail.com",
            privacy_score=0.95,
            tracker_count=0,
            is_open_source=True,
            final_score=0.85,
        )

        self.assertEqual(result.title, "ProtonMail")
        self.assertEqual(result.privacy_score, 0.95)
        self.assertTrue(result.is_open_source)
        self.assertEqual(result.final_score, 0.85)


if __name__ == "__main__":
    unittest.main()
