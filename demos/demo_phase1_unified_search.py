#!/usr/bin/env python3
"""
Intent Engine - Phase 1 Integration Demo

Demonstrates the new unified search capabilities with:
- Query Router (intent-based backend selection)
- Result Aggregator (deduplication and merging)
- Federated Search (parallel execution across backends)

Usage:
    python demos/demo_phase1_unified_search.py

Requirements:
    - SearXNG running on http://localhost:8080
    - Go Crawler running on http://localhost:8081 (optional)
    - Intent Engine API running on http://localhost:8000
"""

import asyncio
import json
import time
from typing import Optional

# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str):
    """Print header text"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}--- {text} ---{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


async def test_query_router():
    """Test the Query Router module directly"""
    print_header("TEST 1: Query Router")

    from core.schema import (
        DeclaredIntent,
        EthicalDimension,
        EthicalSignal,
        InferredIntent,
        IntentGoal,
        UniversalIntent,
        UseCase,
    )
    from searxng.query_router import SearchBackend, get_query_router

    router = get_query_router()

    # Test Case 1: Troubleshooting Query
    print_section("Test 1.1: Troubleshooting Query")
    intent_troubleshoot = UniversalIntent(
        intentId="test-1",
        context={"product": "search"},
        declared=DeclaredIntent(
            query="python not working on windows",
            goal=IntentGoal.TROUBLESHOOTING,
        ),
        inferred=InferredIntent(useCases=[UseCase.TROUBLESHOOTING]),
    )

    route = router.route(intent_troubleshoot)
    print_info(f"Query: 'python not working on windows'")
    print_info(f"Goal: {IntentGoal.TROUBLESHOOTING.value}")
    print_success(f"Routed to: {[b.value for b in route.backends]}")
    print_info(f"Parallel: {route.parallel}")
    print_info(f"Fallback: {[b.value for b in route.fallback_chain]}")

    # Test Case 2: Comparison Query
    print_section("Test 1.2: Comparison Query")
    intent_comparison = UniversalIntent(
        intentId="test-2",
        context={"product": "search"},
        declared=DeclaredIntent(
            query="python vs java comparison",
            goal=IntentGoal.COMPARISON,
        ),
        inferred=InferredIntent(useCases=[UseCase.COMPARISON]),
    )

    route = router.route(intent_comparison)
    print_info(f"Query: 'python vs java comparison'")
    print_info(f"Goal: {IntentGoal.COMPARISON.value}")
    print_success(f"Routed to: {[b.value for b in route.backends]}")
    print_info(f"Weights: {route.weights}")
    print_info(f"Parallel: {route.parallel}")

    # Test Case 3: Privacy-Focused Query
    print_section("Test 1.3: Privacy-Focused Query")
    intent_privacy = UniversalIntent(
        intentId="test-3",
        context={"product": "search"},
        declared=DeclaredIntent(
            query="best privacy-focused email provider",
            goal=IntentGoal.FIND_INFORMATION,
        ),
        inferred=InferredIntent(
            ethicalSignals=[
                EthicalSignal(
                    dimension=EthicalDimension.PRIVACY,
                    preference="privacy-first",
                )
            ]
        ),
    )

    route = router.route(intent_privacy)
    print_info(f"Query: 'best privacy-focused email provider'")
    print_info(f"Ethical Signal: privacy-first")
    print_success(f"Routed to: {[b.value for b in route.backends]}")
    print_info(f"Parallel: {route.parallel}")

    print_header("Query Router Tests Complete")


async def test_result_aggregator():
    """Test the Result Aggregator module"""
    print_header("TEST 2: Result Aggregator")

    from searxng.query_router import SearchBackend, SearchResult
    from searxng.result_aggregator import get_result_aggregator

    aggregator = get_result_aggregator()

    # Create sample results (simulating duplicates from different backends)
    sample_results = [
        SearchResult(
            source=SearchBackend.SEARXNG,
            url="https://example.com/python-tutorial?utm_source=google",
            title="Python Tutorial",
            content="Learn Python programming from scratch",
            score=0.9,
            engine="google",
        ),
        SearchResult(
            source=SearchBackend.GO_CRAWLER,
            url="https://example.com/python-tutorial?utm_campaign=test",
            title="Python Tutorial - Complete Guide",
            content="Complete Python programming tutorial for beginners",
            score=0.85,
            engine="go-crawler",
        ),
        SearchResult(
            source=SearchBackend.SEARXNG,
            url="https://different-site.com/java-guide",
            title="Java Guide",
            content="Learn Java programming",
            score=0.7,
            engine="duckduckgo",
        ),
    ]

    print_section("Input Results")
    print_info(f"Total results: {len(sample_results)}")
    for i, result in enumerate(sample_results, 1):
        print(f"  {i}. {result.url[:50]}... (score: {result.score})")

    # Aggregate results
    aggregated = aggregator.aggregate(sample_results)

    print_section("Aggregated Results")
    print_success(f"Unique results: {len(aggregated)}")
    for i, result in enumerate(aggregated, 1):
        print(f"\n  {Colors.BOLD}Result {i}:{Colors.ENDC}")
        print(f"    URL: {result.url[:60]}...")
        print(f"    Title: {result.title}")
        print(f"    Sources: {result.sources}")
        print(f"    Best Score: {result.best_score:.3f}")
        print(f"    Result Count: {result.result_count}")

    # Test URL normalization
    print_section("URL Normalization")
    test_urls = [
        "https://example.com/page?utm_source=google&utm_campaign=test",
        "https://example.com/page?ref=twitter",
        "https://example.com/page?gclid=abc123",
    ]

    for url in test_urls:
        normalized = aggregator._normalize_url(url)
        print_info(f"Original:  {url}")
        print_success(f"Normalized: {normalized}")
        print()

    print_header("Result Aggregator Tests Complete")


async def test_unified_search_service():
    """Test the enhanced Unified Search Service"""
    print_header("TEST 3: Unified Search Service")

    try:
        from models import UnifiedSearchRequest
        from searxng.unified_search import get_unified_search_service

        service = get_unified_search_service()

        print_section("Unified Search Test")

        # Create search request
        request = UnifiedSearchRequest(
            query="best python tutorials for beginners",
            extract_intent=True,
            rank_results=True,
            max_results=5,
            categories=["general", "tech"],
        )

        print_info(f"Query: '{request.query}'")
        print_info(f"Extract Intent: {request.extract_intent}")
        print_info(f"Rank Results: {request.rank_results}")
        print_info(f"Max Results: {request.max_results}")

        print_section("Executing Search...")
        start_time = time.time()
        response = await service.search(request)
        processing_time = (time.time() - start_time) * 1000

        print_success(f"Search completed in {processing_time:.2f}ms")

        # Display results
        print_section(f"Results ({len(response.results)} total)")

        for i, result in enumerate(response.results[:3], 1):
            print(f"\n  {Colors.BOLD}Result {i}:{Colors.ENDC}")
            print(f"    Title: {result.title[:80]}...")
            print(f"    URL: {result.url[:60]}...")
            print(f"    Score: {result.ranked_score:.3f}")
            print(f"    Engine: {result.engine}")
            if result.match_reasons:
                print(f"    Match Reasons: {result.match_reasons}")

        # Display metrics
        if hasattr(response, "metrics") and response.metrics:
            print_section("Search Metrics")
            for key, value in response.metrics.items():
                print_info(f"{key}: {value}")

    except ImportError as e:
        print_warning(f"Unified Search Service not available: {e}")
        print_info("This is expected if dependencies are not installed")
    except Exception as e:
        print_error(f"Search failed: {e}")
        import traceback

        print(traceback.format_exc())

    print_header("Unified Search Service Tests Complete")


async def test_web_intent_extractor():
    """Test the Web Intent Extractor"""
    print_header("TEST 4: Web Intent Extractor")

    from extraction.web_extractor import get_web_intent_extractor

    extractor = get_web_intent_extractor()

    # Test with sample content
    sample_content = """
    <html>
    <head><title>Python Tutorial for Beginners</title></head>
    <body>
        <h1>Learn Python Programming - Complete Tutorial</h1>
        <p>This is a comprehensive guide for beginners who want to learn Python 
        programming from scratch. We'll cover the basics step by step.</p>
        
        <h2>Chapter 1: Getting Started</h2>
        <p>Python is a powerful, easy-to-learn programming language. It's perfect 
        for beginners and experts alike.</p>
        
        <h2>Chapter 2: Variables and Data Types</h2>
        <p>In this tutorial, you'll learn about variables, strings, numbers, and 
        other fundamental concepts.</p>
    </body>
    </html>
    """

    print_section("Extracting Intent from Sample Content")

    intent = extractor.extract_from_content(
        url="https://example.com/python-tutorial",
        content=sample_content,
    )

    print_info(f"URL: {intent.url}")
    print_success(f"Primary Goal: {intent.primary_goal.value}")
    print_info(f"Use Cases: {[uc.value for uc in intent.use_cases]}")
    print_info(f"Result Type: {intent.result_type.value}")
    print_info(f"Complexity: {intent.complexity.value}")
    print_success(f"Skill Level: {intent.skill_level}")
    print_info(f"Topics: {intent.topics[:3]}")
    print_info(f"Confidence: {intent.confidence:.2f}")

    print_header("Web Intent Extractor Tests Complete")


async def run_all_tests():
    """Run all Phase 1 integration tests"""
    print(
        f"\n{Colors.HEADER}{Colors.BOLD}"
        "╔══════════════════════════════════════════════════════════╗"
    )
    print(
        "║                                                          ║"
    )
    print(
        "║     Intent Engine - Phase 1 Integration Demo             ║"
    )
    print(
        "║     Unified Search with Query Router & Aggregator        ║"
    )
    print(
        "║                                                          ║"
    )
    print(
        f"╚══════════════════════════════════════════════════════════╗{Colors.ENDC}\n"
    )

    start_time = time.time()

    # Run all tests
    await test_query_router()
    await test_result_aggregator()
    await test_unified_search_service()
    await test_web_intent_extractor()

    total_time = time.time() - start_time

    print_header("All Tests Complete")
    print_success(f"Total execution time: {total_time:.2f}s")
    print_info(
        "Next steps: Check the Grafana dashboard at http://localhost:3000"
    )
    print_info("View traces at http://localhost:16686")


def main():
    """Main entry point"""
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Demo interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.FAIL}Demo failed: {e}{Colors.ENDC}")
        import traceback

        print(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
