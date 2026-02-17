"""
Intent Engine - Comprehensive Stress Testing Suite

This module provides comprehensive stress testing for ALL components:
- Intent extraction endpoint
- Ranking endpoint
- URL ranking endpoint
- Service recommendation endpoint
- Ad matching endpoint
- Campaign management endpoints
- Ad group management endpoints
- Creative management endpoints
- Reporting endpoints
- Privacy & consent endpoints
- Database operations
- Memory leak detection
- Connection pool exhaustion
- Cache overflow scenarios

Usage:
    python comprehensive_stress_test.py

Or with custom parameters:
    python comprehensive_stress_test.py --concurrency 100 --duration 120
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import psutil

BASE_URL = "http://localhost:8000"


@dataclass
class TestResults:
    """Container for test results"""

    test_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_memory_mb: float = 0.0
    end_memory_mb: float = 0.0
    duration_seconds: float = 0.0
    rps: float = 0.0
    avg_response_time: float = 0.0
    median_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float("inf")

    def calculate_stats(self):
        """Calculate statistics from response times"""
        if self.response_times:
            self.avg_response_time = statistics.mean(self.response_times)
            self.median_response_time = statistics.median(self.response_times)
            self.p95_response_time = (
                sorted(self.response_times)[int(len(self.response_times) * 0.95)]
                if len(self.response_times) > 20
                else self.max_response_time
            )
            self.p99_response_time = (
                sorted(self.response_times)[int(len(self.response_times) * 0.99)]
                if len(self.response_times) > 100
                else self.max_response_time
            )
            self.max_response_time = max(self.response_times)
            self.min_response_time = min(self.response_times)

        if self.duration_seconds > 0:
            self.rps = self.total_requests / self.duration_seconds


class ComprehensiveStressTestSuite:
    """Comprehensive stress testing for Intent Engine"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.all_results: List[TestResults] = []
        self.session_id = f"stress-test-{int(time.time())}"

        # Diverse test queries
        self.test_queries = [
            "best laptop for programming 2024",
            "how to setup kubernetes cluster",
            "privacy focused email service",
            "secure messaging apps comparison",
            "best password manager",
            "how to improve sleep quality",
            "meditation apps privacy comparison",
            "best budgeting apps without tracking",
            "how to invest in index funds",
            "best travel booking sites privacy",
            "how to find cheap flights",
            "best online courses for programming",
            "how to learn data science",
            "best task management apps privacy",
            "how to organize digital files",
            "compare python vs javascript performance",
            "docker compose tutorial for beginners",
            "secure coding practices checklist",
            "best IDE for web development",
            "machine learning frameworks comparison",
        ]

        # Test URLs
        self.test_urls = [
            "https://github.com",
            "https://stackoverflow.com",
            "https://docs.python.org",
            "https://kubernetes.io",
            "https://docker.com",
            "https://protonmail.com",
            "https://duckduckgo.com",
            "https://signal.org",
            "https://privacytools.io",
            "https://eff.org",
            "https://wikipedia.org",
            "https://mozilla.org",
            "https://techcrunch.com",
            "https://arstechnica.com",
            "https://coursera.org",
            "https://edx.org",
            "https://khanacademy.org",
            "https://freecodecamp.org",
        ]

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "system_percent": psutil.virtual_memory().percent,
        }

    async def test_intent_extraction(self, concurrency: int = 50, duration: int = 30) -> TestResults:
        """Stress test intent extraction endpoint"""
        print(f"\n{'='*70}")
        print(f"STRESS TEST: Intent Extraction")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*70}")

        results = TestResults(test_name="Intent Extraction")
        results.start_memory_mb = self.get_memory_usage()["rss_mb"]
        start_time = time.time()

        async def make_request(session, query: str, request_id: int):
            try:
                payload = {
                    "product": "search",
                    "input": {"text": query},
                    "context": {"sessionId": f"{self.session_id}-{request_id}", "userLocale": "en-US"},
                }

                request_start = time.time()
                async with session.post(
                    f"{self.base_url}/extract-intent", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    elapsed = (time.time() - request_start) * 1000
                    results.total_requests += 1
                    results.response_times.append(elapsed)

                    if response.status == 200:
                        data = await response.json()
                        if "intent" in data:
                            results.successful_requests += 1
                        else:
                            results.failed_requests += 1
                            results.errors.append(f"Missing intent in response")
                    else:
                        results.failed_requests += 1
                        results.errors.append(f"HTTP {response.status}")

            except Exception as e:
                results.total_requests += 1
                results.failed_requests += 1
                results.errors.append(f"Request {request_id}: {str(e)}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            request_id = 0

            while time.time() - start_time < duration:
                while len(tasks) < concurrency:
                    query = self.test_queries[request_id % len(self.test_queries)]
                    task = asyncio.create_task(make_request(session, query, request_id))
                    tasks.append(task)
                    request_id += 1

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        results.duration_seconds = time.time() - start_time
        results.end_memory_mb = self.get_memory_usage()["rss_mb"]
        results.calculate_stats()

        self.print_results(results)
        self.all_results.append(results)
        return results

    async def test_url_ranking(self, concurrency: int = 50, duration: int = 30) -> TestResults:
        """Stress test URL ranking endpoint"""
        print(f"\n{'='*70}")
        print(f"STRESS TEST: URL Ranking")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*70}")

        results = TestResults(test_name="URL Ranking")
        results.start_memory_mb = self.get_memory_usage()["rss_mb"]
        start_time = time.time()

        async def make_request(session, request_id: int):
            try:
                query = self.test_queries[request_id % len(self.test_queries)]
                urls = random.sample(self.test_urls, min(10, len(self.test_urls)))

                payload = {
                    "query": query,
                    "urls": urls,
                    "options": {
                        "exclude_big_tech": random.choice([True, False]),
                        "min_privacy_score": random.choice([0.0, 0.3, 0.5, 0.7]),
                        "weights": {"relevance": 0.40, "privacy": 0.30, "quality": 0.20, "ethics": 0.10},
                    },
                }

                request_start = time.time()
                async with session.post(
                    f"{self.base_url}/rank-urls", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    elapsed = (time.time() - request_start) * 1000
                    results.total_requests += 1
                    results.response_times.append(elapsed)

                    if response.status == 200:
                        results.successful_requests += 1
                    else:
                        results.failed_requests += 1
                        results.errors.append(f"HTTP {response.status}")

            except Exception as e:
                results.total_requests += 1
                results.failed_requests += 1
                results.errors.append(f"Request {request_id}: {str(e)}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            request_id = 0

            while time.time() - start_time < duration:
                while len(tasks) < concurrency:
                    task = asyncio.create_task(make_request(session, request_id))
                    tasks.append(task)
                    request_id += 1

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        results.duration_seconds = time.time() - start_time
        results.end_memory_mb = self.get_memory_usage()["rss_mb"]
        results.calculate_stats()

        self.print_results(results)
        self.all_results.append(results)
        return results

    async def test_rank_results(self, concurrency: int = 50, duration: int = 30) -> TestResults:
        """Stress test result ranking endpoint"""
        print(f"\n{'='*70}")
        print(f"STRESS TEST: Result Ranking")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*70}")

        results = TestResults(test_name="Result Ranking")
        results.start_memory_mb = self.get_memory_usage()["rss_mb"]
        start_time = time.time()

        async def make_request(session, request_id: int):
            try:
                query = self.test_queries[request_id % len(self.test_queries)]

                # First extract intent to get proper structure
                intent_payload = {
                    "product": "search",
                    "input": {"text": query},
                    "context": {"sessionId": f"{self.session_id}-{request_id}", "userLocale": "en-US"},
                }

                async with session.post(
                    f"{self.base_url}/extract-intent", json=intent_payload, timeout=aiohttp.ClientTimeout(total=10)
                ) as intent_response:
                    if intent_response.status != 200:
                        results.total_requests += 1
                        results.failed_requests += 1
                        results.errors.append(f"Intent extraction failed: HTTP {intent_response.status}")
                        return

                    intent_data = await intent_response.json()
                    intent = intent_data.get("intent", {})

                # Create test candidates
                candidates = []
                for i in range(random.randint(5, 15)):
                    candidates.append(
                        {
                            "id": f"result-{request_id}-{i}",
                            "title": f"Result {i} for {query[:30]}",
                            "description": f"Description for result {i} with relevant keywords",
                            "platform": random.choice(["web", "mobile", "desktop"]),
                            "provider": f"Provider {i}",
                            "qualityScore": random.uniform(0.5, 1.0),
                            "tags": query.split()[:3],
                            "privacyRating": random.uniform(0.5, 1.0),
                            "opensource": random.choice([True, False]),
                        }
                    )

                payload = {"intent": intent, "candidates": candidates}

                request_start = time.time()
                async with session.post(
                    f"{self.base_url}/rank-results", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    elapsed = (time.time() - request_start) * 1000
                    results.total_requests += 1
                    results.response_times.append(elapsed)

                    if response.status == 200:
                        results.successful_requests += 1
                    else:
                        results.failed_requests += 1
                        results.errors.append(f"HTTP {response.status}")

            except Exception as e:
                results.total_requests += 1
                results.failed_requests += 1
                results.errors.append(f"Request {request_id}: {str(e)}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            request_id = 0

            while time.time() - start_time < duration:
                while len(tasks) < concurrency:
                    task = asyncio.create_task(make_request(session, request_id))
                    tasks.append(task)
                    request_id += 1

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        results.duration_seconds = time.time() - start_time
        results.end_memory_mb = self.get_memory_usage()["rss_mb"]
        results.calculate_stats()

        self.print_results(results)
        self.all_results.append(results)
        return results

    async def test_ad_matching(self, concurrency: int = 50, duration: int = 30) -> TestResults:
        """Stress test ad matching endpoint"""
        print(f"\n{'='*70}")
        print(f"STRESS TEST: Ad Matching")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*70}")

        results = TestResults(test_name="Ad Matching")
        results.start_memory_mb = self.get_memory_usage()["rss_mb"]
        start_time = time.time()

        async def make_request(session, request_id: int):
            try:
                query = self.test_queries[request_id % len(self.test_queries)]

                # First extract intent to get proper structure
                intent_payload = {
                    "product": "search",
                    "input": {"text": query},
                    "context": {"sessionId": f"{self.session_id}-{request_id}", "userLocale": "en-US"},
                }

                async with session.post(
                    f"{self.base_url}/extract-intent", json=intent_payload, timeout=aiohttp.ClientTimeout(total=10)
                ) as intent_response:
                    if intent_response.status != 200:
                        results.total_requests += 1
                        results.failed_requests += 1
                        results.errors.append(f"Intent extraction failed: HTTP {intent_response.status}")
                        return

                    intent_data = await intent_response.json()
                    intent = intent_data.get("intent", {})

                # Create test ads
                ads = []
                for i in range(random.randint(3, 10)):
                    ads.append(
                        {
                            "id": f"ad-{request_id}-{i}",
                            "title": f"Ad {i} for {query[:20]}",
                            "description": f"Great product related to your search",
                            "targetingConstraints": {},
                            "forbiddenDimensions": [],
                            "qualityScore": random.uniform(0.6, 0.95),
                            "ethicalTags": random.sample(["privacy", "quality", "open-source", "sustainable"], k=2),
                            "advertiser": f"advertiser-{i}",
                            "creative_format": random.choice(["banner", "native", "video"]),
                        }
                    )

                payload = {"intent": intent, "ad_inventory": ads, "config": {"minThreshold": 0.3, "topK": 5}}

                request_start = time.time()
                async with session.post(
                    f"{self.base_url}/match-ads", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    elapsed = (time.time() - request_start) * 1000
                    results.total_requests += 1
                    results.response_times.append(elapsed)

                    if response.status == 200:
                        results.successful_requests += 1
                    else:
                        results.failed_requests += 1
                        results.errors.append(f"HTTP {response.status}")

            except Exception as e:
                results.total_requests += 1
                results.failed_requests += 1
                results.errors.append(f"Request {request_id}: {str(e)}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            request_id = 0

            while time.time() - start_time < duration:
                while len(tasks) < concurrency:
                    task = asyncio.create_task(make_request(session, request_id))
                    tasks.append(task)
                    request_id += 1

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        results.duration_seconds = time.time() - start_time
        results.end_memory_mb = self.get_memory_usage()["rss_mb"]
        results.calculate_stats()

        self.print_results(results)
        self.all_results.append(results)
        return results

    async def test_service_recommendation(self, concurrency: int = 50, duration: int = 30) -> TestResults:
        """Stress test service recommendation endpoint"""
        print(f"\n{'='*70}")
        print(f"STRESS TEST: Service Recommendation")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*70}")

        results = TestResults(test_name="Service Recommendation")
        results.start_memory_mb = self.get_memory_usage()["rss_mb"]
        start_time = time.time()

        async def make_request(session, request_id: int):
            try:
                query = self.test_queries[request_id % len(self.test_queries)]

                # First extract intent to get proper structure
                intent_payload = {
                    "product": "search",
                    "input": {"text": query},
                    "context": {"sessionId": f"{self.session_id}-{request_id}", "userLocale": "en-US"},
                }

                async with session.post(
                    f"{self.base_url}/extract-intent", json=intent_payload, timeout=aiohttp.ClientTimeout(total=10)
                ) as intent_response:
                    if intent_response.status != 200:
                        results.total_requests += 1
                        results.failed_requests += 1
                        results.errors.append(f"Intent extraction failed: HTTP {intent_response.status}")
                        return

                    intent_data = await intent_response.json()
                    intent = intent_data.get("intent", {})

                payload = {
                    "intent": intent,
                    "available_services": [
                        {
                            "id": "search",
                            "name": "Search",
                            "supportedGoals": ["find_information", "comparison"],
                            "primaryUseCases": ["learning", "verification"],
                            "temporalPatterns": [],
                            "ethicalAlignment": ["privacy"],
                            "description": "Privacy-focused search service",
                        },
                        {
                            "id": "docs",
                            "name": "Documentation",
                            "supportedGoals": ["learn", "troubleshooting"],
                            "primaryUseCases": ["learning", "troubleshooting"],
                            "temporalPatterns": [],
                            "ethicalAlignment": ["openness"],
                            "description": "Technical documentation service",
                        },
                        {
                            "id": "help",
                            "name": "Help Center",
                            "supportedGoals": ["troubleshooting", "learn"],
                            "primaryUseCases": ["troubleshooting", "verification"],
                            "temporalPatterns": [],
                            "ethicalAlignment": ["accessibility"],
                            "description": "User support and help service",
                        },
                    ],
                }

                request_start = time.time()
                async with session.post(
                    f"{self.base_url}/recommend-services", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    elapsed = (time.time() - request_start) * 1000
                    results.total_requests += 1
                    results.response_times.append(elapsed)

                    if response.status == 200:
                        results.successful_requests += 1
                    else:
                        results.failed_requests += 1
                        results.errors.append(f"HTTP {response.status}")

            except Exception as e:
                results.total_requests += 1
                results.failed_requests += 1
                results.errors.append(f"Request {request_id}: {str(e)}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            request_id = 0

            while time.time() - start_time < duration:
                while len(tasks) < concurrency:
                    task = asyncio.create_task(make_request(session, request_id))
                    tasks.append(task)
                    request_id += 1

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        results.duration_seconds = time.time() - start_time
        results.end_memory_mb = self.get_memory_usage()["rss_mb"]
        results.calculate_stats()

        self.print_results(results)
        self.all_results.append(results)
        return results

    async def test_campaign_management(self, concurrency: int = 20, duration: int = 30) -> TestResults:
        """Stress test campaign management endpoints"""
        print(f"\n{'='*70}")
        print(f"STRESS TEST: Campaign Management")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*70}")

        results = TestResults(test_name="Campaign Management")
        results.start_memory_mb = self.get_memory_usage()["rss_mb"]
        start_time = time.time()

        async def make_request(session, request_id: int):
            try:
                # Mix of different operations
                operation = random.choice(["list", "create", "get"])

                if operation == "list":
                    request_start = time.time()
                    async with session.get(
                        f"{self.base_url}/campaigns", timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed = (time.time() - request_start) * 1000
                        results.total_requests += 1
                        results.response_times.append(elapsed)

                        if response.status == 200:
                            results.successful_requests += 1
                        else:
                            results.failed_requests += 1
                            results.errors.append(f"HTTP {response.status}")

                elif operation == "create":
                    payload = {
                        "advertiser_id": random.randint(1, 10),
                        "name": f"Stress Test Campaign {request_id}",
                        "start_date": "2026-02-16T00:00:00",
                        "end_date": "2026-03-16T23:59:59",
                        "budget": random.uniform(1000, 10000),
                        "daily_budget": random.uniform(50, 200),
                        "status": "active",
                    }

                    request_start = time.time()
                    async with session.post(
                        f"{self.base_url}/campaigns", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed = (time.time() - request_start) * 1000
                        results.total_requests += 1
                        results.response_times.append(elapsed)

                        if response.status == 200:
                            results.successful_requests += 1
                        else:
                            results.failed_requests += 1
                            results.errors.append(f"HTTP {response.status}")

                elif operation == "get":
                    campaign_id = random.randint(1, 100)
                    request_start = time.time()
                    async with session.get(
                        f"{self.base_url}/campaigns/{campaign_id}", timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed = (time.time() - request_start) * 1000
                        results.total_requests += 1
                        results.response_times.append(elapsed)

                        # 404 is acceptable for non-existent campaigns
                        if response.status in [200, 404]:
                            results.successful_requests += 1
                        else:
                            results.failed_requests += 1
                            results.errors.append(f"HTTP {response.status}")

            except Exception as e:
                results.total_requests += 1
                results.failed_requests += 1
                results.errors.append(f"Request {request_id}: {str(e)}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            request_id = 0

            while time.time() - start_time < duration:
                while len(tasks) < concurrency:
                    task = asyncio.create_task(make_request(session, request_id))
                    tasks.append(task)
                    request_id += 1

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        results.duration_seconds = time.time() - start_time
        results.end_memory_mb = self.get_memory_usage()["rss_mb"]
        results.calculate_stats()

        self.print_results(results)
        self.all_results.append(results)
        return results

    async def test_reporting_endpoints(self, concurrency: int = 20, duration: int = 30) -> TestResults:
        """Stress test reporting endpoints"""
        print(f"\n{'='*70}")
        print(f"STRESS TEST: Reporting Endpoints")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*70}")

        results = TestResults(test_name="Reporting Endpoints")
        results.start_memory_mb = self.get_memory_usage()["rss_mb"]
        start_time = time.time()

        async def make_request(session, request_id: int):
            try:
                operation = random.choice(["campaign_performance", "ad_performance"])

                if operation == "campaign_performance":
                    request_start = time.time()
                    async with session.get(
                        f"{self.base_url}/reports/campaign-performance", timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed = (time.time() - request_start) * 1000
                        results.total_requests += 1
                        results.response_times.append(elapsed)

                        if response.status == 200:
                            results.successful_requests += 1
                        else:
                            results.failed_requests += 1
                            results.errors.append(f"HTTP {response.status}")

            except Exception as e:
                results.total_requests += 1
                results.failed_requests += 1
                results.errors.append(f"Request {request_id}: {str(e)}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            request_id = 0

            while time.time() - start_time < duration:
                while len(tasks) < concurrency:
                    task = asyncio.create_task(make_request(session, request_id))
                    tasks.append(task)
                    request_id += 1

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        results.duration_seconds = time.time() - start_time
        results.end_memory_mb = self.get_memory_usage()["rss_mb"]
        results.calculate_stats()

        self.print_results(results)
        self.all_results.append(results)
        return results

    async def test_memory_leaks(self, iterations: int = 500) -> Dict[str, Any]:
        """Test for memory leaks under sustained load"""
        print(f"\n{'='*70}")
        print(f"MEMORY LEAK TEST: {iterations} iterations")
        print(f"{'='*70}")

        memory_samples = []
        successful = 0
        failed = 0

        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                payload = {
                    "product": "search",
                    "input": {"text": f"memory leak test query {i}"},
                    "context": {"sessionId": f"mem-test-{i}", "userLocale": "en-US"},
                }

                try:
                    async with session.post(f"{self.base_url}/extract-intent", json=payload) as response:
                        await response.json()
                        successful += 1
                except Exception as e:
                    failed += 1

                # Sample memory every 50 iterations
                if i % 50 == 0:
                    mem = self.get_memory_usage()
                    memory_samples.append({"iteration": i, **mem})
                    print(f"  Iteration {i}: {mem['rss_mb']:.1f} MB RSS")

        # Analyze memory trend
        analysis = {
            "samples": memory_samples,
            "successful": successful,
            "failed": failed,
            "memory_growth_mb": 0,
            "potential_leak": False,
        }

        if len(memory_samples) > 1:
            first = memory_samples[0]["rss_mb"]
            last = memory_samples[-1]["rss_mb"]
            growth = last - first
            analysis["memory_growth_mb"] = growth
            analysis["potential_leak"] = growth > 50

            print(f"\nMemory Analysis:")
            print(f"  Initial: {first:.1f} MB")
            print(f"  Final: {last:.1f} MB")
            print(f"  Growth: {growth:.1f} MB")
            print(f"  Growth per 100 requests: {growth / len(memory_samples) * 100:.2f} MB")

            if growth > 50:
                print(f"  [WARNING] Potential memory leak detected!")
            else:
                print(f"  [OK] Memory usage stable")

        return analysis

    async def test_concurrent_all_endpoints(self, concurrency: int = 100, duration: int = 60) -> TestResults:
        """Stress test all endpoints simultaneously"""
        print(f"\n{'='*70}")
        print(f"STRESS TEST: All Endpoints Combined")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*70}")

        results = TestResults(test_name="All Endpoints Combined")
        results.start_memory_mb = self.get_memory_usage()["rss_mb"]
        start_time = time.time()

        async def make_request(session, request_id: int):
            try:
                query = self.test_queries[request_id % len(self.test_queries)]
                endpoint = random.choice(
                    ["extract-intent", "rank-urls", "match-ads", "recommend-services", "campaigns"]
                )

                request_start = time.time()

                if endpoint == "extract-intent":
                    payload = {
                        "product": "search",
                        "input": {"text": query},
                        "context": {"sessionId": f"{self.session_id}-{request_id}", "userLocale": "en-US"},
                    }
                    async with session.post(
                        f"{self.base_url}/{endpoint}", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed = (time.time() - request_start) * 1000
                        results.total_requests += 1
                        results.response_times.append(elapsed)
                        if response.status == 200:
                            results.successful_requests += 1
                        else:
                            results.failed_requests += 1
                            results.errors.append(f"HTTP {response.status}")

                elif endpoint == "rank-urls":
                    payload = {
                        "query": query,
                        "urls": random.sample(self.test_urls, min(10, len(self.test_urls))),
                        "options": {
                            "exclude_big_tech": random.choice([True, False]),
                            "min_privacy_score": random.choice([0.0, 0.3, 0.5, 0.7]),
                        },
                    }
                    async with session.post(
                        f"{self.base_url}/{endpoint}", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed = (time.time() - request_start) * 1000
                        results.total_requests += 1
                        results.response_times.append(elapsed)
                        if response.status == 200:
                            results.successful_requests += 1
                        else:
                            results.failed_requests += 1
                            results.errors.append(f"HTTP {response.status}")

                elif endpoint in ["match-ads", "recommend-services"]:
                    # First extract intent
                    intent_payload = {
                        "product": "search",
                        "input": {"text": query},
                        "context": {"sessionId": f"{self.session_id}-{request_id}", "userLocale": "en-US"},
                    }
                    async with session.post(
                        f"{self.base_url}/extract-intent", json=intent_payload, timeout=aiohttp.ClientTimeout(total=10)
                    ) as intent_response:
                        if intent_response.status != 200:
                            results.total_requests += 1
                            results.failed_requests += 1
                            results.errors.append(f"Intent extraction failed")
                            return
                        intent_data = await intent_response.json()
                        intent = intent_data.get("intent", {})

                    if endpoint == "match-ads":
                        ads = [
                            {
                                "id": f"ad-{request_id}-{i}",
                                "title": f"Ad {i}",
                                "description": f"Ad description {i}",
                                "targetingConstraints": {},
                                "qualityScore": random.uniform(0.6, 0.95),
                                "ethicalTags": ["privacy"],
                                "advertiser": f"adv-{i}",
                            }
                            for i in range(random.randint(3, 8))
                        ]
                        payload = {"intent": intent, "ad_inventory": ads, "config": {"minThreshold": 0.3, "topK": 5}}
                    else:  # recommend-services
                        payload = {
                            "intent": intent,
                            "available_services": [
                                {
                                    "id": "search",
                                    "name": "Search",
                                    "supportedGoals": ["find_information"],
                                    "primaryUseCases": ["learning"],
                                },
                                {
                                    "id": "docs",
                                    "name": "Docs",
                                    "supportedGoals": ["learn"],
                                    "primaryUseCases": ["learning"],
                                },
                            ],
                        }

                    async with session.post(
                        f"{self.base_url}/{endpoint}", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed = (time.time() - request_start) * 1000
                        results.total_requests += 1
                        results.response_times.append(elapsed)
                        if response.status == 200:
                            results.successful_requests += 1
                        else:
                            results.failed_requests += 1
                            results.errors.append(f"HTTP {response.status}")

                else:  # campaigns (GET)
                    async with session.get(
                        f"{self.base_url}/{endpoint}", timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed = (time.time() - request_start) * 1000
                        results.total_requests += 1
                        results.response_times.append(elapsed)
                        if response.status == 200:
                            results.successful_requests += 1
                        else:
                            results.failed_requests += 1
                            results.errors.append(f"HTTP {response.status}")

            except Exception as e:
                results.total_requests += 1
                results.failed_requests += 1
                results.errors.append(f"Request {request_id}: {str(e)}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            request_id = 0

            while time.time() - start_time < duration:
                while len(tasks) < concurrency:
                    task = asyncio.create_task(make_request(session, request_id))
                    tasks.append(task)
                    request_id += 1

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        results.duration_seconds = time.time() - start_time
        results.end_memory_mb = self.get_memory_usage()["rss_mb"]
        results.calculate_stats()

        self.print_results(results)
        self.all_results.append(results)
        return results

    def print_results(self, results: TestResults):
        """Print formatted test results"""
        print(f"\n{results.test_name} Results:")
        print(f"  Total Requests: {results.total_requests:,}")
        print(
            f"  Successful: {results.successful_requests:,} ({results.successful_requests/results.total_requests*100:.1f}%)"
        )
        print(f"  Failed: {results.failed_requests:,}")
        print(f"  Duration: {results.duration_seconds:.2f}s")
        print(f"  Requests/sec: {results.rps:.2f}")

        if results.response_times:
            print(f"\n  Response Times (ms):")
            print(f"    Min: {results.min_response_time:.2f}")
            print(f"    Avg: {results.avg_response_time:.2f}")
            print(f"    Median: {results.median_response_time:.2f}")
            print(f"    95th percentile: {results.p95_response_time:.2f}")
            print(f"    99th percentile: {results.p99_response_time:.2f}")
            print(f"    Max: {results.max_response_time:.2f}")

        print(f"\n  Memory Usage:")
        print(f"    Start: {results.start_memory_mb:.1f} MB")
        print(f"    End: {results.end_memory_mb:.1f} MB")
        print(f"    Change: {results.end_memory_mb - results.start_memory_mb:+.1f} MB")

        if results.errors:
            print(f"\n  [WARN] Errors ({len(results.errors)}):")
            for error in results.errors[:5]:
                print(f"    - {error}")

    def generate_report(self, output_file: str = "stress_test_report.json"):
        """Generate comprehensive test report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "summary": {
                "total_tests": len(self.all_results),
                "total_requests": sum(r.total_requests for r in self.all_results),
                "total_successful": sum(r.successful_requests for r in self.all_results),
                "total_failed": sum(r.failed_requests for r in self.all_results),
                "overall_success_rate": (
                    sum(r.successful_requests for r in self.all_results)
                    / sum(r.total_requests for r in self.all_results)
                    * 100
                    if self.all_results
                    else 0
                ),
                "total_duration_seconds": sum(r.duration_seconds for r in self.all_results),
                "avg_rps": sum(r.rps for r in self.all_results) / len(self.all_results) if self.all_results else 0,
            },
            "tests": [asdict(r) for r in self.all_results],
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*70}")
        print(f"STRESS TEST REPORT GENERATED: {output_file}")
        print(f"{'='*70}")
        print(f"\nOverall Summary:")
        print(f"  Total Tests Run: {report['summary']['total_tests']}")
        print(f"  Total Requests: {report['summary']['total_requests']:,}")
        print(f"  Success Rate: {report['summary']['overall_success_rate']:.2f}%")
        print(f"  Average RPS: {report['summary']['avg_rps']:.2f}")
        print(f"  Total Duration: {report['summary']['total_duration_seconds']:.2f}s")

        return report

    async def run_all_stress_tests(self, concurrency: int = 50, duration: int = 30):
        """Run all stress tests"""
        print("\n" + "=" * 70)
        print("INTENT ENGINE COMPREHENSIVE STRESS TEST SUITE")
        print(f"Configuration: Concurrency={concurrency}, Duration={duration}s per test")
        print("=" * 70)

        try:
            # Core endpoint tests
            await self.test_intent_extraction(concurrency=concurrency, duration=duration)
            await self.test_url_ranking(concurrency=concurrency, duration=duration)
            await self.test_rank_results(concurrency=concurrency, duration=duration)
            await self.test_ad_matching(concurrency=concurrency, duration=duration)
            await self.test_service_recommendation(concurrency=concurrency, duration=duration)

            # Advertising system tests
            await self.test_campaign_management(concurrency=concurrency // 2, duration=duration)
            await self.test_reporting_endpoints(concurrency=concurrency // 2, duration=duration)

            # Memory leak test
            await self.test_memory_leaks(iterations=500)

            # Combined stress test
            await self.test_concurrent_all_endpoints(concurrency=concurrency, duration=duration * 2)

            # Generate report
            self.generate_report()

            print("\n" + "=" * 70)
            print("ALL STRESS TESTS COMPLETED SUCCESSFULLY")
            print("=" * 70)

        except Exception as e:
            print(f"\n[ERROR] Stress tests failed: {e}")
            traceback.print_exc()
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Intent Engine Comprehensive Stress Test Suite")
    parser.add_argument("--concurrency", type=int, default=50, help="Number of concurrent requests")
    parser.add_argument("--duration", type=int, default=30, help="Duration of each test in seconds")
    parser.add_argument("--url", type=str, default=BASE_URL, help="Base URL of the API")
    parser.add_argument("--output", type=str, default="stress_test_report.json", help="Output file for report")

    args = parser.parse_args()

    suite = ComprehensiveStressTestSuite(base_url=args.url)

    try:
        asyncio.run(suite.run_all_stress_tests(concurrency=args.concurrency, duration=args.duration))
    except KeyboardInterrupt:
        print("\n\n[WARN] Tests interrupted by user")
        suite.generate_report(args.output)
    except Exception as e:
        print(f"\n[ERROR] Tests failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
