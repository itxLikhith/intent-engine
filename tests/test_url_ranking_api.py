"""
Integration tests for the /rank-urls API endpoint using httpx ASGITransport
"""

import asyncio
import unittest

import httpx

from main_api import app


def run_async(coro):
    """Helper to run async code in sync tests"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestURLRankingAPI(unittest.TestCase):
    """Integration tests for the /rank-urls endpoint"""

    @classmethod
    def setUpClass(cls):
        cls.transport = httpx.ASGITransport(app=app)

    async def _post(self, url, json):
        async with httpx.AsyncClient(transport=self.transport, base_url="http://test") as client:
            return await client.post(url, json=json)

    def post(self, url, json):
        return run_async(self._post(url, json))

    def test_basic_ranking(self):
        """Test basic URL ranking with a simple query and URLs"""
        payload = {
            "query": "privacy focused email service",
            "urls": [
                "https://protonmail.com",
                "https://tutanota.com",
                "https://google.com",
                "https://duckduckgo.com",
                "https://github.com",
            ],
        }

        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["query"], "privacy focused email service")
        self.assertEqual(data["total_urls"], 5)
        self.assertEqual(len(data["ranked_urls"]), 5)
        self.assertGreater(data["processing_time_ms"], 0)

        first = data["ranked_urls"][0]
        self.assertIn("url", first)
        self.assertIn("privacy_score", first)
        self.assertIn("relevance_score", first)
        self.assertIn("final_score", first)
        self.assertIn("domain", first)

    def test_25_urls(self):
        """Test ranking 25 URLs as specified in the requirements"""
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
            "https://wikipedia.org/wiki/Privacy",
        ]

        payload = {"query": "privacy tools", "urls": urls}
        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["total_urls"], 25)
        self.assertEqual(len(data["ranked_urls"]), 25)
        self.assertLess(data["processing_time_ms"], 10000)

    def test_exclude_big_tech(self):
        """Test the exclude_big_tech option filters correctly"""
        payload = {
            "query": "email service",
            "urls": [
                "https://protonmail.com",
                "https://google.com",
                "https://facebook.com",
                "https://tutanota.com",
            ],
            "options": {"exclude_big_tech": True},
        }

        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        domains = [r["domain"] for r in data["ranked_urls"]]
        self.assertNotIn("google.com", domains)
        self.assertNotIn("facebook.com", domains)
        self.assertEqual(data["filtered_count"], 2)

    def test_min_privacy_score(self):
        """Test the min_privacy_score filter"""
        payload = {
            "query": "privacy",
            "urls": [
                "https://protonmail.com",
                "https://google.com",
            ],
            "options": {"min_privacy_score": 0.8},
        }

        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(len(data["ranked_urls"]), 1)
        self.assertEqual(data["ranked_urls"][0]["domain"], "protonmail.com")

    def test_custom_weights(self):
        """Test custom scoring weights"""
        payload = {
            "query": "email",
            "urls": [
                "https://protonmail.com",
                "https://google.com",
            ],
            "options": {"weights": {"relevance": 0.1, "privacy": 0.6, "quality": 0.15, "ethics": 0.15}},
        }

        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["ranked_urls"][0]["domain"], "protonmail.com")

    def test_with_intent(self):
        """Test ranking with a full UniversalIntent object"""
        payload = {
            "query": "open source code hosting",
            "urls": [
                "https://github.com",
                "https://gitlab.com",
                "https://google.com",
            ],
            "intent": {
                "intentId": "test-api-1",
                "context": {"product": "search"},
                "declared": {"query": "open source code hosting"},
                "inferred": {"ethicalSignals": [{"dimension": "openness", "preference": "open-source_preferred"}]},
            },
        }

        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(len(data["ranked_urls"]), 3)

        # Open source domains should rank above non-open-source
        top_domain = data["ranked_urls"][0]["domain"]
        self.assertIn(top_domain, ["github.com", "gitlab.com"])

    def test_with_negative_preferences(self):
        """Test ranking with negative preferences via intent"""
        payload = {
            "query": "search engine",
            "urls": [
                "https://duckduckgo.com",
                "https://google.com",
                "https://facebook.com",
                "https://startpage.com",
            ],
            "intent": {
                "intentId": "test-api-2",
                "context": {"product": "search"},
                "declared": {"query": "search engine", "negativePreferences": ["no google", "no facebook"]},
                "inferred": {},
            },
        }

        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        domains = [r["domain"] for r in data["ranked_urls"]]
        self.assertNotIn("google.com", domains)
        self.assertNotIn("facebook.com", domains)

    def test_empty_urls(self):
        """Test with empty URL list"""
        payload = {"query": "test", "urls": []}

        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(len(data["ranked_urls"]), 0)
        self.assertEqual(data["total_urls"], 0)

    def test_response_schema(self):
        """Test that the response schema matches URLRankingAPIResponse"""
        payload = {"query": "privacy", "urls": ["https://protonmail.com"]}

        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertIn("query", data)
        self.assertIn("ranked_urls", data)
        self.assertIn("processing_time_ms", data)
        self.assertIn("total_urls", data)
        self.assertIn("filtered_count", data)

        result = data["ranked_urls"][0]
        expected_fields = [
            "url",
            "title",
            "description",
            "domain",
            "privacy_score",
            "tracker_count",
            "encryption_enabled",
            "content_type",
            "is_open_source",
            "is_non_profit",
            "relevance_score",
            "final_score",
        ]
        for field in expected_fields:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_scores_normalized(self):
        """Test that all scores are in the 0-1 range"""
        payload = {
            "query": "privacy email",
            "urls": [
                "https://protonmail.com",
                "https://google.com",
                "https://wikipedia.org",
            ],
        }

        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 200)

        for result in response.json()["ranked_urls"]:
            self.assertGreaterEqual(result["final_score"], 0.0)
            self.assertLessEqual(result["final_score"], 1.0)
            self.assertGreaterEqual(result["relevance_score"], 0.0)
            self.assertLessEqual(result["relevance_score"], 1.0)
            self.assertGreaterEqual(result["privacy_score"], 0.0)
            self.assertLessEqual(result["privacy_score"], 1.0)

    def test_missing_query_returns_422(self):
        """Test that missing required field returns 422"""
        payload = {"urls": ["https://example.com"]}
        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 422)

    def test_missing_urls_returns_422(self):
        """Test that missing urls field returns 422"""
        payload = {"query": "test"}
        response = self.post("/rank-urls", payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
