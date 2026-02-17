"""
Intent Engine - Comprehensive Query Test Suite

This module tests the search engine with diverse queries across multiple domains.
NOT just email-related queries - covers shopping, health, finance, travel, etc.
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import requests

BASE_URL = "http://localhost:8000"


class QueryCategory(Enum):
    """Categories for diverse test queries"""

    TECHNOLOGY = "technology"
    SHOPPING = "shopping"
    HEALTH_WELLNESS = "health_wellness"
    FINANCE = "finance"
    TRAVEL = "travel"
    EDUCATION = "education"
    PRODUCTIVITY = "productivity"
    PRIVACY_SECURITY = "privacy_security"
    ENTERTAINMENT = "entertainment"
    HOME_LIFESTYLE = "home_lifestyle"


@dataclass
class TestQuery:
    """Test query with expected outcomes"""

    query: str
    category: QueryCategory
    expected_goal: str
    expected_use_cases: List[str]
    expected_complexity: str
    expected_constraints: List[str]


# DIVERSE TEST QUERIES - NOT JUST EMAIL!
DIVERSE_TEST_QUERIES = {
    QueryCategory.TECHNOLOGY: [
        TestQuery(
            query="best laptop for programming under $1500",
            category=QueryCategory.TECHNOLOGY,
            expected_goal="compare",
            expected_use_cases=["comparison", "learning"],
            expected_complexity="moderate",
            expected_constraints=["price", "platform"],
        ),
        TestQuery(
            query="how to setup kubernetes cluster for microservices",
            category=QueryCategory.TECHNOLOGY,
            expected_goal="learn",
            expected_use_cases=["learning", "troubleshooting"],
            expected_complexity="complex",
            expected_constraints=[],
        ),
        TestQuery(
            query="compare Python vs Go for backend development performance",
            category=QueryCategory.TECHNOLOGY,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="advanced",
            expected_constraints=[],
        ),
        TestQuery(
            query="docker compose tutorial for beginners step by step",
            category=QueryCategory.TECHNOLOGY,
            expected_goal="learn",
            expected_use_cases=["learning"],
            expected_complexity="beginner",
            expected_constraints=[],
        ),
        TestQuery(
            query="fix database connection timeout error in PostgreSQL",
            category=QueryCategory.TECHNOLOGY,
            expected_goal="troubleshoot",
            expected_use_cases=["troubleshooting"],
            expected_complexity="complex",
            expected_constraints=["platform"],
        ),
    ],
    QueryCategory.SHOPPING: [
        TestQuery(
            query="best wireless headphones under 100 dollars noise cancelling",
            category=QueryCategory.SHOPPING,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["price", "feature"],
        ),
        TestQuery(
            query="compare iPhone 15 vs Samsung Galaxy S24 camera quality",
            category=QueryCategory.SHOPPING,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
        TestQuery(
            query="where to buy sustainable clothing brands ethical fashion",
            category=QueryCategory.SHOPPING,
            expected_goal="purchase",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="mechanical keyboard reviews 2024 reddit recommendations",
            category=QueryCategory.SHOPPING,
            expected_goal="compare",
            expected_use_cases=["comparison", "verification"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
        TestQuery(
            query="smart home devices privacy concerns which to avoid",
            category=QueryCategory.SHOPPING,
            expected_goal="compare",
            expected_use_cases=["comparison", "verification"],
            expected_complexity="moderate",
            expected_constraints=["feature"],
        ),
    ],
    QueryCategory.HEALTH_WELLNESS: [
        TestQuery(
            query="how to improve sleep quality naturally without medication",
            category=QueryCategory.HEALTH_WELLNESS,
            expected_goal="learn",
            expected_use_cases=["learning"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
        TestQuery(
            query="meditation apps privacy comparison headspace vs calm",
            category=QueryCategory.HEALTH_WELLNESS,
            expected_goal="compare",
            expected_use_cases=["comparison", "verification"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="home workout routines without equipment for beginners",
            category=QueryCategory.HEALTH_WELLNESS,
            expected_goal="learn",
            expected_use_cases=["learning"],
            expected_complexity="beginner",
            expected_constraints=[],
        ),
        TestQuery(
            query="nutrition tracking apps without data sharing privacy",
            category=QueryCategory.HEALTH_WELLNESS,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["feature"],
        ),
        TestQuery(
            query="yoga for beginners tutorial at home 30 minutes",
            category=QueryCategory.HEALTH_WELLNESS,
            expected_goal="learn",
            expected_use_cases=["learning"],
            expected_complexity="beginner",
            expected_constraints=[],
        ),
    ],
    QueryCategory.FINANCE: [
        TestQuery(
            query="best budgeting apps without tracking privacy focused",
            category=QueryCategory.FINANCE,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["feature"],
        ),
        TestQuery(
            query="how to invest in index funds for beginners guide",
            category=QueryCategory.FINANCE,
            expected_goal="learn",
            expected_use_cases=["learning"],
            expected_complexity="beginner",
            expected_constraints=[],
        ),
        TestQuery(
            query="cryptocurrency wallet security best practices 2024",
            category=QueryCategory.FINANCE,
            expected_goal="learn",
            expected_use_cases=["learning", "verification"],
            expected_complexity="advanced",
            expected_constraints=[],
        ),
        TestQuery(
            query="compare credit card rewards cash back vs travel points",
            category=QueryCategory.FINANCE,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
        TestQuery(
            query="tax preparation software privacy concerns turboTax alternatives",
            category=QueryCategory.FINANCE,
            expected_goal="compare",
            expected_use_cases=["comparison", "verification"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
    ],
    QueryCategory.TRAVEL: [
        TestQuery(
            query="best travel booking sites privacy Expedia vs Booking",
            category=QueryCategory.TRAVEL,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="how to find cheap flights without tracking cookies",
            category=QueryCategory.TRAVEL,
            expected_goal="learn",
            expected_use_cases=["learning"],
            expected_complexity="moderate",
            expected_constraints=["feature"],
        ),
        TestQuery(
            query="solo travel safety tips for female travelers",
            category=QueryCategory.TRAVEL,
            expected_goal="learn",
            expected_use_cases=["learning", "verification"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
        TestQuery(
            query="digital nomad destinations 2024 cost of living",
            category=QueryCategory.TRAVEL,
            expected_goal="compare",
            expected_use_cases=["comparison", "exploration"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
        TestQuery(
            query="offline maps for travel without internet Google Maps alternatives",
            category=QueryCategory.TRAVEL,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
    ],
    QueryCategory.EDUCATION: [
        TestQuery(
            query="best online courses for programming python certification",
            category=QueryCategory.EDUCATION,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
        TestQuery(
            query="how to learn data science from scratch roadmap",
            category=QueryCategory.EDUCATION,
            expected_goal="learn",
            expected_use_cases=["learning"],
            expected_complexity="complex",
            expected_constraints=[],
        ),
        TestQuery(
            query="language learning apps comparison Duolingo vs Babbel privacy",
            category=QueryCategory.EDUCATION,
            expected_goal="compare",
            expected_use_cases=["comparison", "verification"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="free educational resources online Khan Academy alternatives",
            category=QueryCategory.EDUCATION,
            expected_goal="compare",
            expected_use_cases=["comparison", "exploration"],
            expected_complexity="moderate",
            expected_constraints=["price"],
        ),
        TestQuery(
            query="how to study effectively scientific techniques memory",
            category=QueryCategory.EDUCATION,
            expected_goal="learn",
            expected_use_cases=["learning"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
    ],
    QueryCategory.PRODUCTIVITY: [
        TestQuery(
            query="best task management apps privacy Todoist vs TickTick",
            category=QueryCategory.PRODUCTIVITY,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="how to organize digital files folder structure best practices",
            category=QueryCategory.PRODUCTIVITY,
            expected_goal="learn",
            expected_use_cases=["learning"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
        TestQuery(
            query="note taking apps comparison Obsidian vs Notion privacy",
            category=QueryCategory.PRODUCTIVITY,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="calendar apps without Google privacy focused alternatives",
            category=QueryCategory.PRODUCTIVITY,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="focus techniques for deep work Pomodoro method alternatives",
            category=QueryCategory.PRODUCTIVITY,
            expected_goal="learn",
            expected_use_cases=["learning"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
    ],
    QueryCategory.PRIVACY_SECURITY: [
        TestQuery(
            query="best password manager comparison LastPass vs Bitwarden",
            category=QueryCategory.PRIVACY_SECURITY,
            expected_goal="compare",
            expected_use_cases=["comparison", "verification"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="how to browse anonymously Tor browser vs VPN",
            category=QueryCategory.PRIVACY_SECURITY,
            expected_goal="compare",
            expected_use_cases=["comparison", "learning"],
            expected_complexity="advanced",
            expected_constraints=[],
        ),
        TestQuery(
            query="VPN services privacy review Mullvad vs NordVPN logs",
            category=QueryCategory.PRIVACY_SECURITY,
            expected_goal="compare",
            expected_use_cases=["comparison", "verification"],
            expected_complexity="advanced",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="secure messaging apps comparison Signal vs Telegram privacy",
            category=QueryCategory.PRIVACY_SECURITY,
            expected_goal="compare",
            expected_use_cases=["comparison", "verification"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="privacy focused browser alternatives to Chrome Firefox vs Brave",
            category=QueryCategory.PRIVACY_SECURITY,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
    ],
    QueryCategory.ENTERTAINMENT: [
        TestQuery(
            query="best streaming services comparison Netflix vs Disney privacy",
            category=QueryCategory.ENTERTAINMENT,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="music streaming without tracking Spotify alternatives privacy",
            category=QueryCategory.ENTERTAINMENT,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="podcast apps privacy focused without tracking",
            category=QueryCategory.ENTERTAINMENT,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["feature"],
        ),
        TestQuery(
            query="ebook readers comparison Kindle vs Kobo privacy",
            category=QueryCategory.ENTERTAINMENT,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=[],
        ),
        TestQuery(
            query="gaming platforms privacy Steam vs Epic vs GOG",
            category=QueryCategory.ENTERTAINMENT,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
    ],
    QueryCategory.HOME_LIFESTYLE: [
        TestQuery(
            query="smart home devices privacy Alexa vs Google Home concerns",
            category=QueryCategory.HOME_LIFESTYLE,
            expected_goal="compare",
            expected_use_cases=["comparison", "verification"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
        TestQuery(
            query="home security cameras privacy Ring vs local storage options",
            category=QueryCategory.HOME_LIFESTYLE,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["feature"],
        ),
        TestQuery(
            query="recipe apps without tracking privacy focused meal planning",
            category=QueryCategory.HOME_LIFESTYLE,
            expected_goal="compare",
            expected_use_cases=["comparison"],
            expected_complexity="moderate",
            expected_constraints=["feature"],
        ),
        TestQuery(
            query="gardening apps plant identification privacy concerns",
            category=QueryCategory.HOME_LIFESTYLE,
            expected_goal="compare",
            expected_use_cases=["comparison", "verification"],
            expected_complexity="beginner",
            expected_constraints=[],
        ),
        TestQuery(
            query="DIY home improvement tutorials without YouTube alternatives",
            category=QueryCategory.HOME_LIFESTYLE,
            expected_goal="compare",
            expected_use_cases=["comparison", "learning"],
            expected_complexity="moderate",
            expected_constraints=["provider"],
        ),
    ],
}


class ComprehensiveQueryTestSuite:
    """Test suite for diverse queries"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = []

    def test_intent_extraction(self, test_query: TestQuery) -> Dict[str, Any]:
        """Test intent extraction for a specific query"""
        print(f"\nTesting: {test_query.query[:60]}...")

        payload = {
            "product": "search",
            "input": {"text": test_query.query},
            "context": {"sessionId": f"test-{int(time.time())}", "userLocale": "en-US"},
        }

        start_time = time.time()
        response = requests.post(f"{self.base_url}/extract-intent", json=payload)
        elapsed = (time.time() - start_time) * 1000

        if response.status_code != 200:
            return {
                "query": test_query.query,
                "status": "FAILED",
                "error": f"HTTP {response.status_code}",
                "latency_ms": elapsed,
            }

        data = response.json()
        intent = data.get("intent", {})
        declared = intent.get("declared", {})
        inferred = intent.get("inferred", {})

        # Validate results
        validations = {
            "goal_match": declared.get("goal") == test_query.expected_goal,
            "complexity_match": inferred.get("complexity") == test_query.expected_complexity,
            "has_intent_id": bool(intent.get("intentId")),
            "has_constraints": len(declared.get("constraints", [])) > 0,
        }

        return {
            "query": test_query.query,
            "category": test_query.category.value,
            "status": "PASSED" if all(validations.values()) else "PARTIAL",
            "validations": validations,
            "extracted_goal": declared.get("goal"),
            "extracted_complexity": inferred.get("complexity"),
            "confidence": data.get("extractionMetrics", {}).get("confidence"),
            "latency_ms": elapsed,
        }

    def test_url_ranking(self, test_query: TestQuery, urls: List[str]) -> Dict[str, Any]:
        """Test URL ranking for a query"""
        print(f"  Ranking {len(urls)} URLs...")

        payload = {
            "query": test_query.query,
            "urls": urls,
            "options": {
                "exclude_big_tech": True,
                "weights": {"relevance": 0.40, "privacy": 0.30, "quality": 0.20, "ethics": 0.10},
            },
        }

        start_time = time.time()
        response = requests.post(f"{self.base_url}/rank-urls", json=payload)
        elapsed = (time.time() - start_time) * 1000

        if response.status_code != 200:
            return {"status": "FAILED", "error": f"HTTP {response.status_code}", "latency_ms": elapsed}

        data = response.json()
        ranked_urls = data.get("ranked_urls", [])

        return {
            "status": "PASSED" if len(ranked_urls) > 0 else "FAILED",
            "urls_ranked": len(ranked_urls),
            "top_score": ranked_urls[0].get("final_score") if ranked_urls else 0,
            "cache_hit_rate": data.get("cache_hit_rate", 0),
            "latency_ms": elapsed,
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("=" * 70)
        print("COMPREHENSIVE QUERY TEST SUITE")
        print("=" * 70)
        print(f"Testing {sum(len(v) for v in DIVERSE_TEST_QUERIES.values())} queries")
        print(f"Categories: {len(DIVERSE_TEST_QUERIES)}")
        print("=" * 70)

        all_results = []
        category_stats = {}

        # Test URLs for ranking
        test_urls = [
            "https://github.com",
            "https://protonmail.com",
            "https://duckduckgo.com",
            "https://signal.org",
            "https://privacytools.io",
        ]

        for category, queries in DIVERSE_TEST_QUERIES.items():
            print(f"\n{'='*70}")
            print(f"CATEGORY: {category.value.upper()}")
            print(f"{'='*70}")

            category_results = []

            for test_query in queries:
                # Test intent extraction
                result = self.test_intent_extraction(test_query)
                category_results.append(result)
                all_results.append(result)

                # Test URL ranking
                if result["status"] == "PASSED":
                    rank_result = self.test_url_ranking(test_query, test_urls)
                    print(f"  URL Ranking: {rank_result['status']} ({rank_result['latency_ms']:.0f}ms)")

            # Category statistics
            passed = sum(1 for r in category_results if r["status"] == "PASSED")
            partial = sum(1 for r in category_results if r["status"] == "PARTIAL")
            failed = sum(1 for r in category_results if r["status"] == "FAILED")
            avg_latency = statistics.mean([r["latency_ms"] for r in category_results])

            category_stats[category.value] = {
                "total": len(category_results),
                "passed": passed,
                "partial": partial,
                "failed": failed,
                "avg_latency_ms": avg_latency,
            }

            print(
                f"\n  Category Summary: {passed}/{len(category_results)} passed, " f"avg latency: {avg_latency:.0f}ms"
            )

        # Overall summary
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)

        total = len(all_results)
        passed = sum(1 for r in all_results if r["status"] == "PASSED")
        partial = sum(1 for r in all_results if r["status"] == "PARTIAL")
        failed = sum(1 for r in all_results if r["status"] == "FAILED")
        avg_latency = statistics.mean([r["latency_ms"] for r in all_results])

        print(f"\nTotal Queries Tested: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Partial: {partial} ({partial/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Average Latency: {avg_latency:.0f}ms")

        print("\nCategory Breakdown:")
        for cat, stats in category_stats.items():
            print(f"  {cat:20s}: {stats['passed']}/{stats['total']} passed, " f"{stats['avg_latency_ms']:.0f}ms avg")

        print("=" * 70)

        return {
            "total": total,
            "passed": passed,
            "partial": partial,
            "failed": failed,
            "avg_latency_ms": avg_latency,
            "category_stats": category_stats,
            "detailed_results": all_results,
        }


def main():
    """Run comprehensive query tests"""

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("❌ Server not responding correctly")
            return
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        print("Please start the server first with: docker-compose up")
        return

    print("SUCCESS: Server is running")

    # Run tests
    suite = ComprehensiveQueryTestSuite()
    results = suite.run_all_tests()

    # Save results
    with open("comprehensive_query_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to comprehensive_query_test_results.json")


if __name__ == "__main__":
    main()
