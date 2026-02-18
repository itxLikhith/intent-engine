"""
Search Engine Stress Test Suite

Comprehensive stress testing for the unified search endpoint (/search)
which combines SearXNG privacy search with Intent Engine ranking.

Tests include:
- Basic search throughput
- Concurrent search requests
- Intent extraction + search + ranking pipeline
- Privacy filter performance
- Large result set handling
- Cache effectiveness
"""

import asyncio
import statistics
import time
from datetime import datetime
from typing import Dict, List

import aiohttp

BASE_URL = "http://localhost:8000"


class SearchEngineStressTest:
    """Stress testing suite for search engine capabilities"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = {}

        # Test queries covering different search intents
        self.test_queries = [
            # Informational queries
            "how to set up encrypted email",
            "best privacy-focused browser for Linux",
            "what is differential privacy",
            "how does end-to-end encryption work",
            "open source alternatives to Google Drive",
            # Comparison queries
            "Signal vs Telegram privacy comparison",
            "ProtonMail vs Tutanota features",
            "best password manager 2024",
            "Firefox vs Brave privacy browser",
            "DuckDuckGo vs Startpage search engine",
            # Transactional queries
            "download Signal for Android",
            "install Tor Browser Ubuntu",
            "setup VPN on router",
            "configure DNS over HTTPS",
            "enable two-factor authentication",
            # Privacy-focused queries
            "privacy tools for journalists",
            "secure messaging apps for business",
            "how to avoid online tracking",
            "best VPN for privacy 2024",
            "anonymous browsing techniques",
            # Technical queries
            "docker container security best practices",
            "Kubernetes encryption at rest",
            "implement OAuth2 authentication",
            "API rate limiting strategies",
            "microservices architecture patterns",
        ]

    async def test_basic_search_throughput(self, iterations: int = 100):
        """Test basic search throughput without intent extraction"""
        print(f"\n{'='*60}")
        print(f"Basic Search Throughput Test: {iterations} requests")
        print(f"{'='*60}")

        results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": [],
        }

        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                query = self.test_queries[i % len(self.test_queries)]

                payload = {"query": query, "extract_intent": False, "rank_results": False, "num_results": 10}

                start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/search", json=payload, timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        elapsed = (time.time() - start) * 1000
                        results["total_requests"] += 1
                        results["response_times"].append(elapsed)

                        if response.status == 200:
                            results["successful_requests"] += 1
                            result_data = await response.json()
                            if i == 0:
                                print(f"  Sample: {len(result_data.get('results', []))} results returned")
                        else:
                            results["failed_requests"] += 1
                            results["errors"].append(f"Request {i}: HTTP {response.status}")

                except Exception as e:
                    results["total_requests"] += 1
                    results["failed_requests"] += 1
                    results["errors"].append(f"Request {i}: {str(e)}")

        self._print_test_results("Basic Search Throughput", results)
        return results

    async def test_intent_extraction_only(self, iterations: int = 50):
        """Test intent extraction endpoint performance"""
        print(f"\n{'='*60}")
        print(f"Intent Extraction Performance Test: {iterations} requests")
        print(f"{'='*60}")

        results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": [],
        }

        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                query = self.test_queries[i % len(self.test_queries)]

                payload = {
                    "product": "search",
                    "input": {"text": query},
                    "context": {"sessionId": f"stress-test-{i}", "userLocale": "en-US"},
                }

                start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/extract-intent", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed = (time.time() - start) * 1000
                        results["total_requests"] += 1
                        results["response_times"].append(elapsed)

                        if response.status == 200:
                            results["successful_requests"] += 1
                            if i == 0:
                                result_data = await response.json()
                                intent = result_data.get("intent", {})
                                declared = intent.get("declared", {})
                                inferred = intent.get("inferred", {})
                                goal = declared.get("goal", "unknown")
                                complexity = inferred.get("complexity", "unknown")
                                print(f"  Sample: goal={goal}, complexity={complexity}")
                        else:
                            results["failed_requests"] += 1
                            results["errors"].append(f"Request {i}: HTTP {response.status}")

                except Exception as e:
                    results["total_requests"] += 1
                    results["failed_requests"] += 1
                    results["errors"].append(f"Request {i}: {str(e)}")

        self._print_test_results("Intent Extraction", results)
        return results

    async def test_search_with_separate_intent(self, iterations: int = 30):
        """Test search with pre-extracted intent (2-step process)"""
        print(f"\n{'='*60}")
        print(f"Search + Intent (2-Step) Test: {iterations} requests")
        print(f"{'='*60}")

        results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "search_times": [],
            "intent_times": [],
            "errors": [],
        }

        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                query = self.test_queries[i % len(self.test_queries)]

                # Step 1: Extract intent
                intent_payload = {
                    "product": "search",
                    "input": {"text": query},
                    "context": {"sessionId": f"two-step-{i}", "userLocale": "en-US"},
                }

                intent_start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/extract-intent", json=intent_payload, timeout=aiohttp.ClientTimeout(total=30)
                    ) as intent_response:
                        intent_elapsed = (time.time() - intent_start) * 1000
                        results["intent_times"].append(intent_elapsed)

                        if intent_response.status != 200:
                            results["failed_requests"] += 1
                            results["errors"].append(f"Request {i} intent: HTTP {intent_response.status}")
                            continue

                        intent_data = await intent_response.json()
                        intent_data.get("intent", {})

                        # Step 2: Basic search (without intent extraction)
                        search_payload = {
                            "query": query,
                            "extract_intent": False,
                            "rank_results": False,
                            "num_results": 10,
                        }

                        search_start = time.time()
                        async with session.post(
                            f"{self.base_url}/search", json=search_payload, timeout=aiohttp.ClientTimeout(total=60)
                        ) as search_response:
                            search_elapsed = (time.time() - search_start) * 1000
                            total_elapsed = (time.time() - intent_start) * 1000

                            results["total_requests"] += 1
                            results["response_times"].append(total_elapsed)
                            results["search_times"].append(search_elapsed)

                            if search_response.status == 200:
                                results["successful_requests"] += 1
                                if i == 0:
                                    search_data = await search_response.json()
                                    print(f"  Sample: {len(search_data.get('results', []))} results")
                                    print(f"  Intent time: {intent_elapsed:.2f}ms, Search time: {search_elapsed:.2f}ms")
                            else:
                                results["failed_requests"] += 1
                                results["errors"].append(f"Request {i} search: HTTP {search_response.status}")

                except Exception as e:
                    results["total_requests"] += 1
                    results["failed_requests"] += 1
                    results["errors"].append(f"Request {i}: {str(e)}")

        self._print_test_results("Search + Intent (2-Step)", results)
        return results

    async def test_concurrent_search(self, concurrency: int = 30, duration: int = 30):
        """Test search under high concurrent load"""
        print(f"\n{'='*60}")
        print("Concurrent Search Stress Test")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*60}")

        results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": [],
            "start_time": time.time(),
            "end_time": None,
        }

        async def make_request(session, query: str, request_id: int):
            payload = {"query": query, "extract_intent": True, "rank_results": True, "num_results": 10}

            start = time.time()
            try:
                async with session.post(
                    f"{self.base_url}/search", json=payload, timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    elapsed = (time.time() - start) * 1000
                    results["total_requests"] += 1
                    results["response_times"].append(elapsed)

                    if response.status == 200:
                        results["successful_requests"] += 1
                    else:
                        results["failed_requests"] += 1
                        results["errors"].append(f"Request {request_id}: HTTP {response.status}")

            except Exception as e:
                results["total_requests"] += 1
                results["failed_requests"] += 1
                results["errors"].append(f"Request {request_id}: {str(e)}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            request_id = 0
            start_time = time.time()

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

        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        results["rps"] = results["total_requests"] / results["duration"]

        self._print_test_results("Concurrent Search Stress", results)
        return results

    async def test_privacy_filter_performance(self, iterations: int = 50):
        """Test performance with privacy filters enabled"""
        print(f"\n{'='*60}")
        print(f"Privacy Filter Performance Test: {iterations} requests")
        print(f"{'='*60}")

        test_configs = [
            {"name": "No Filters", "exclude_big_tech": False, "min_privacy_score": None},
            {"name": "Exclude Big Tech", "exclude_big_tech": True, "min_privacy_score": None},
            {"name": "Min Privacy 0.7", "exclude_big_tech": False, "min_privacy_score": 0.7},
            {"name": "Both Filters", "exclude_big_tech": True, "min_privacy_score": 0.7},
        ]

        all_results = {}

        async with aiohttp.ClientSession() as session:
            for config in test_configs:
                print(f"\n  Testing: {config['name']}...")

                results = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "response_times": [],
                    "result_counts": [],
                }

                for i in range(iterations):
                    query = self.test_queries[i % len(self.test_queries)]

                    payload = {
                        "query": query,
                        "extract_intent": True,
                        "rank_results": True,
                        "num_results": 20,
                        "exclude_big_tech": config["exclude_big_tech"],
                    }

                    if config["min_privacy_score"]:
                        payload["min_privacy_score"] = config["min_privacy_score"]

                    start = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}/search", json=payload, timeout=aiohttp.ClientTimeout(total=60)
                        ) as response:
                            elapsed = (time.time() - start) * 1000
                            results["total_requests"] += 1
                            results["response_times"].append(elapsed)

                            if response.status == 200:
                                results["successful_requests"] += 1
                                result_data = await response.json()
                                results["result_counts"].append(len(result_data.get("results", [])))

                    except Exception:
                        results["total_requests"] += 1
                        results["failed_requests"] += 1

                all_results[config["name"]] = results

                print(f"    Avg response time: {statistics.mean(results['response_times']):.2f}ms")
                print(f"    Avg results returned: {statistics.mean(results['result_counts']):.1f}")
                print(f"    Success rate: {results['successful_requests']/results['total_requests']*100:.1f}%")

        return all_results

    async def test_large_result_handling(self, num_results_list: List[int] = [10, 50, 100]):
        """Test handling of different result set sizes"""
        print(f"\n{'='*60}")
        print("Large Result Set Handling Test")
        print(f"{'='*60}")

        results_by_size = {}

        async with aiohttp.ClientSession() as session:
            for num_results in num_results_list:
                print(f"\n  Testing with {num_results} results...")

                results = {"total_requests": 0, "successful_requests": 0, "response_times": [], "actual_results": []}

                for i in range(20):
                    query = self.test_queries[i % len(self.test_queries)]

                    payload = {
                        "query": query,
                        "extract_intent": False,
                        "rank_results": False,
                        "num_results": num_results,
                    }

                    start = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}/search", json=payload, timeout=aiohttp.ClientTimeout(total=60)
                        ) as response:
                            elapsed = (time.time() - start) * 1000
                            results["total_requests"] += 1
                            results["response_times"].append(elapsed)

                            if response.status == 200:
                                results["successful_requests"] += 1
                                result_data = await response.json()
                                results["actual_results"].append(len(result_data.get("results", [])))

                    except Exception:
                        results["total_requests"] += 1

                results_by_size[num_results] = results

                avg_time = statistics.mean(results["response_times"]) if results["response_times"] else 0
                avg_returned = statistics.mean(results["actual_results"]) if results["actual_results"] else 0

                print(f"    Avg response time: {avg_time:.2f}ms")
                print(f"    Avg results returned: {avg_returned:.1f}")

        return results_by_size

    async def test_search_result_quality(self, iterations: int = 20):
        """Test search result quality and completeness"""
        print(f"\n{'='*60}")
        print(f"Search Result Quality Test: {iterations} queries")
        print(f"{'='*60}")

        quality_metrics = {
            "total_results_returned": [],
            "results_with_content": [],
            "results_with_thumbnails": [],
            "unique_engines": set(),
            "processing_times": [],
        }

        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                query = self.test_queries[i % len(self.test_queries)]

                payload = {"query": query, "extract_intent": True, "rank_results": True, "num_results": 20}

                try:
                    async with session.post(
                        f"{self.base_url}/search", json=payload, timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 200:
                            result_data = await response.json()
                            results = result_data.get("results", [])

                            quality_metrics["total_results_returned"].append(len(results))

                            # Count results with content
                            with_content = sum(1 for r in results if r.get("content"))
                            quality_metrics["results_with_content"].append(with_content)

                            # Count results with thumbnails
                            with_thumbnails = sum(1 for r in results if r.get("thumbnail"))
                            quality_metrics["results_with_thumbnails"].append(with_thumbnails)

                            # Track engines used
                            engines = result_data.get("engines_used", [])
                            quality_metrics["unique_engines"].update(engines)

                            # Track processing time
                            processing_time = result_data.get("processing_time_ms", 0)
                            quality_metrics["processing_times"].append(processing_time)

                except Exception as e:
                    print(f"  Error on query {i}: {e}")

        # Print quality metrics
        print("\n  Quality Metrics:")
        if quality_metrics["total_results_returned"]:
            print(f"    Avg results per query: {statistics.mean(quality_metrics['total_results_returned']):.1f}")
        if quality_metrics["results_with_content"]:
            print(f"    Avg results with content: {statistics.mean(quality_metrics['results_with_content']):.1f}")
        if quality_metrics["results_with_thumbnails"]:
            print(f"    Avg results with thumbnails: {statistics.mean(quality_metrics['results_with_thumbnails']):.1f}")
        print(f"    Unique engines used: {len(quality_metrics['unique_engines'])}")
        print(f"    Engines: {', '.join(sorted(quality_metrics['unique_engines']))}")
        if quality_metrics["processing_times"]:
            print(f"    Avg processing time: {statistics.mean(quality_metrics['processing_times']):.2f}ms")

        return quality_metrics

    def _print_test_results(self, test_name: str, results: Dict):
        """Print formatted test results"""
        print(f"\n{test_name} Results:")
        print(f"  Total requests: {results['total_requests']}")
        print(f"  Successful: {results['successful_requests']}")
        print(f"  Failed: {results['failed_requests']}")

        if results["total_requests"] > 0:
            success_rate = results["successful_requests"] / results["total_requests"] * 100
            print(f"  Success rate: {success_rate:.1f}%")

        if "duration" in results and results["duration"]:
            print(f"  Duration: {results['duration']:.2f}s")

        if "rps" in results:
            print(f"  Throughput: {results['rps']:.2f} requests/second")

        if results["response_times"]:
            print("\n  Response Times:")
            print(f"    Average: {statistics.mean(results['response_times']):.2f}ms")
            print(f"    Median: {statistics.median(results['response_times']):.2f}ms")
            print(f"    Min: {min(results['response_times']):.2f}ms")
            print(f"    Max: {max(results['response_times']):.2f}ms")

            sorted_times = sorted(results["response_times"])
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            print(f"    95th percentile: {sorted_times[p95_idx]:.2f}ms")
            print(f"    99th percentile: {sorted_times[min(p99_idx, len(sorted_times)-1)]:.2f}ms")

        if results.get("errors"):
            print(f"\n  [WARN] Errors ({len(results['errors'])}):")
            for error in results["errors"][:5]:
                print(f"    - {error}")

    async def run_all_search_tests(self):
        """Run all search engine stress tests"""
        print("\n" + "=" * 60)
        print("SEARCH ENGINE STRESS TEST SUITE")
        print(f"Target: {self.base_url}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        try:
            # Test 1: Basic search throughput
            await self.test_basic_search_throughput(iterations=50)

            # Test 2: Intent extraction performance
            await self.test_intent_extraction_only(iterations=50)

            # Test 3: Combined search + intent (2-step process)
            await self.test_search_with_separate_intent(iterations=30)

            # Test 4: Concurrent search load
            await self.test_concurrent_search(concurrency=20, duration=30)

            # Test 5: Privacy filters
            await self.test_privacy_filter_performance(iterations=30)

            # Test 6: Large result sets
            await self.test_large_result_handling(num_results_list=[10, 30, 50])

            # Test 7: Result quality
            await self.test_search_result_quality(iterations=15)

            print("\n" + "=" * 60)
            print("ALL SEARCH ENGINE TESTS COMPLETED")
            print("=" * 60)

        except Exception as e:
            print(f"\n[ERROR] Tests failed: {e}")
            raise


def main():
    """Main entry point"""
    print("Starting Search Engine Stress Tests...")

    tester = SearchEngineStressTest()

    try:
        asyncio.run(tester.run_all_search_tests())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Tests failed: {e}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
