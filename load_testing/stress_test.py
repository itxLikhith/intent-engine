"""
Intent Engine - Stress Testing Suite

This module provides stress testing to find the breaking point of the system.
Tests include:
- Memory leak detection
- Connection pool exhaustion
- Cache overflow scenarios
- Concurrent request handling
"""

import asyncio
import aiohttp
import time
import statistics
import psutil
import os
from typing import List, Dict, Any
import json
import concurrent.futures
from datetime import datetime


BASE_URL = "http://localhost:8000"

# FIX: Add semaphore for concurrent request limiting
MAX_CONCURRENT_REQUESTS = 50
_request_semaphore = None


def get_request_semaphore():
    """Get or create semaphore for limiting concurrent requests"""
    global _request_semaphore
    if _request_semaphore is None:
        _request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _request_semaphore


class StressTestSuite:
    """Comprehensive stress testing for Intent Engine"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = []
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": psutil.virtual_memory().percent
        }
    
    async def test_concurrent_intent_extraction(self, concurrency: int = 100, duration: int = 60):
        """Test intent extraction under high concurrent load"""
        print(f"\n{'='*60}")
        print(f"Stress Test: Concurrent Intent Extraction")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*60}")
        
        queries = [
            "best laptop for programming",
            "how to learn machine learning",
            "privacy focused email service",
            "secure messaging apps comparison",
            "best password manager 2024"
        ]
        
        results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "start_memory": self.get_memory_usage(),
            "end_memory": None,
            "errors": []
        }
        
        start_time = time.time()
        
        async def make_request(session, query: str, request_id: int):
            # FIX: Use semaphore to limit concurrent requests
            async with get_request_semaphore():
                try:
                    payload = {
                        "product": "search",
                        "input": {"text": query},
                        "context": {"sessionId": f"stress-{request_id}", "userLocale": "en-US"}
                    }

                    request_start = time.time()
                    async with session.post(
                        f"{self.base_url}/extract-intent",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed = (time.time() - request_start) * 1000

                        results["total_requests"] += 1
                        results["response_times"].append(elapsed)

                        if response.status == 200:
                            results["successful_requests"] += 1
                        else:
                            results["failed_requests"] += 1
                            results["errors"].append(f"Request {request_id}: HTTP {response.status}")

                except asyncio.TimeoutError as e:
                    results["total_requests"] += 1
                    results["failed_requests"] += 1
                    results["errors"].append(f"Request {request_id}: Timeout - {str(e)}")
                except aiohttp.ClientError as e:
                    results["total_requests"] += 1
                    results["failed_requests"] += 1
                    results["errors"].append(f"Request {request_id}: ClientError - {str(e)}")
                except Exception as e:
                    results["total_requests"] += 1
                    results["failed_requests"] += 1
                    results["errors"].append(f"Request {request_id}: {type(e).__name__} - {str(e)}")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            request_id = 0

            while time.time() - start_time < duration:
                # Maintain concurrency level
                while len(tasks) < concurrency:
                    query = queries[request_id % len(queries)]
                    task = asyncio.create_task(make_request(session, query, request_id))
                    tasks.append(task)
                    request_id += 1

                # Wait for at least one task to complete
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=5.0  # FIX: Add timeout to prevent indefinite waits
                )
                
                # Cancel done tasks to free resources
                for task in done:
                    try:
                        task.result()  # Consume any exceptions
                    except Exception:
                        pass  # Already logged in make_request
                
                tasks = list(pending)

            # Wait for remaining tasks with timeout
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    # FIX: Cancel remaining tasks gracefully
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    # Wait for cancellation to complete
                    await asyncio.sleep(0.1)
        
        results["end_memory"] = self.get_memory_usage()
        results["duration"] = time.time() - start_time
        results["rps"] = results["total_requests"] / results["duration"]

        # Calculate statistics
        if results["response_times"]:
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["median_response_time"] = statistics.median(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
            # FIX: Prevent off-by-one error in percentile calculation
            sorted_times = sorted(results["response_times"])
            p95_index = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
            results["p95_response_time"] = sorted_times[p95_index]
        
        self.print_stress_results("Intent Extraction", results)
        return results

    async def test_memory_leaks(self, iterations: int = 1000):
        """Test for memory leaks under sustained load"""
        print(f"\n{'='*60}")
        print(f"Memory Leak Test: {iterations} iterations")
        print(f"{'='*60}")

        memory_samples = []
        session = None
        
        try:
            session = aiohttp.ClientSession()
            
            for i in range(iterations):
                # Make a request
                payload = {
                    "product": "search",
                    "input": {"text": f"test query {i}"},
                    "context": {"sessionId": f"mem-test-{i}", "userLocale": "en-US"}
                }

                try:
                    async with session.post(
                        f"{self.base_url}/extract-intent",
                        json=payload
                    ) as response:
                        await response.json()
                except Exception as e:
                    print(f"Error on iteration {i}: {e}")

                # Sample memory every 100 iterations
                if i % 100 == 0:
                    mem = self.get_memory_usage()
                    memory_samples.append({"iteration": i, **mem})
                    print(f"  Iteration {i}: {mem['rss_mb']:.1f} MB RSS")
        finally:
            # FIX: Idempotent session cleanup with state check
            if session and not session.closed:
                try:
                    await session.close()
                except Exception as e:
                    print(f"Warning: Session cleanup error (safe to ignore): {e}")

        # Analyze memory trend
        if len(memory_samples) > 1:
            first = memory_samples[0]["rss_mb"]
            last = memory_samples[-1]["rss_mb"]
            growth = last - first
            
            print(f"\nMemory Analysis:")
            print(f"  Initial: {first:.1f} MB")
            print(f"  Final: {last:.1f} MB")
            print(f"  Growth: {growth:.1f} MB")
            print(f"  Growth per 100 requests: {growth / len(memory_samples):.2f} MB")
            
            if growth > 50:  # More than 50MB growth
                print(f"  [WARNING] Potential memory leak detected!")
            else:
                print(f"  [OK] Memory usage stable")
        
        return memory_samples
    
    async def test_cache_overflow(self, unique_queries: int = 5000):
        """Test cache behavior with many unique queries"""
        print(f"\n{'='*60}")
        print(f"Cache Overflow Test: {unique_queries} unique queries")
        print(f"{'='*60}")

        results = {
            "first_pass": {"total": 0, "avg_time": 0},
            "second_pass": {"total": 0, "avg_time": 0},
            "repeated_queries": {"total": 0, "avg_time": 0}
        }

        session = None
        try:
            session = aiohttp.ClientSession()
            
            # First pass - populate cache with unique queries
            print("  First pass (cache population)...")
            times = []

            for i in range(unique_queries):
                payload = {
                    "product": "search",
                    "input": {"text": f"unique query number {i} with different keywords"},
                    "context": {"sessionId": f"cache-test-{i}", "userLocale": "en-US"}
                }

                start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/extract-intent",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        await response.json()
                        times.append((time.time() - start) * 1000)
                except Exception as e:
                    print(f"  Warning: Query {i} failed: {e}")
                    continue

            results["first_pass"]["total"] = len(times)
            results["first_pass"]["avg_time"] = statistics.mean(times) if times else 0

            # Second pass - exact same queries (should hit cache)
            print("  Second pass (cache retrieval - same queries)...")
            times = []

            for i in range(unique_queries):
                payload = {
                    "product": "search",
                    "input": {"text": f"unique query number {i} with different keywords"},
                    "context": {"sessionId": f"cache-test-{i}", "userLocale": "en-US"}
                }

                start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/extract-intent",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        await response.json()
                        times.append((time.time() - start) * 1000)
                except Exception:
                    continue

            results["second_pass"]["total"] = len(times)
            results["second_pass"]["avg_time"] = statistics.mean(times) if times else 0

            # Third pass - repeated common queries (better cache test)
            print("  Third pass (repeated common queries - high cache hit)...")
            common_queries = [
                "best laptop for programming",
                "how to learn machine learning",
                "privacy focused email service",
                "secure messaging apps comparison",
                "best password manager 2024",
                "how to set up encrypted email",
                "open source alternatives to google",
                "best linux distribution for beginners",
                "how to use docker containers",
                "python vs javascript for backend"
            ]
            times = []
            query_count = min(unique_queries, 1000)  # Limit to avoid too long test

            for i in range(query_count):
                query = common_queries[i % len(common_queries)]
                payload = {
                    "product": "search",
                    "input": {"text": query},
                    "context": {"sessionId": f"cache-test-repeat-{i}", "userLocale": "en-US"}
                }

                start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/extract-intent",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        await response.json()
                        times.append((time.time() - start) * 1000)
                except Exception:
                    continue

            results["repeated_queries"]["total"] = len(times)
            results["repeated_queries"]["avg_time"] = statistics.mean(times) if times else 0
            
        finally:
            # FIX: Idempotent session cleanup
            if session and not session.closed:
                try:
                    await session.close()
                except Exception as e:
                    print(f"Warning: Session cleanup error (safe to ignore): {e}")

        # Compare results
        speedup_second = results["first_pass"]["avg_time"] / results["second_pass"]["avg_time"] if results["second_pass"]["avg_time"] > 0 else 0
        speedup_repeated = results["first_pass"]["avg_time"] / results["repeated_queries"]["avg_time"] if results["repeated_queries"]["avg_time"] > 0 else 0
        
        print(f"\nCache Performance:")
        print(f"  First pass avg (cold cache): {results['first_pass']['avg_time']:.2f}ms")
        print(f"  Second pass avg (same queries): {results['second_pass']['avg_time']:.2f}ms")
        print(f"  Speedup (same queries): {speedup_second:.2f}x")
        print(f"  Repeated common queries avg: {results['repeated_queries']['avg_time']:.2f}ms")
        print(f"  Speedup (repeated queries): {speedup_repeated:.2f}x")

        if speedup_second > 1.5 or speedup_repeated > 2:
            print(f"  [OK] Cache working effectively")
        else:
            print(f"  [WARN] Cache may be full or not working optimally")
            print(f"  [INFO] Note: /extract-intent uses rule-based extraction, not embeddings")
            print(f"  [INFO] Cache is used in /rank-results and /match-ads endpoints")

        return results

    async def test_ranking_with_embeddings(self, concurrency: int = 30, duration: int = 20):
        """Test ranking endpoint which uses embedding cache"""
        print(f"\n{'='*60}")
        print(f"Stress Test: Ranking with Embeddings (uses cache)")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*60}")

        # First, get a valid intent from the extract-intent endpoint
        async with aiohttp.ClientSession() as session:
            # Extract intent first
            extract_payload = {
                "product": "search",
                "input": {"text": "best laptop for programming"},
                "context": {"sessionId": "ranking-test-session", "userLocale": "en-US"}
            }
            
            try:
                async with session.post(
                    f"{self.base_url}/extract-intent",
                    json=extract_payload
                ) as response:
                    if response.status != 200:
                        print(f"  [WARN] Failed to get valid intent: HTTP {response.status}")
                        print(f"  [INFO] Skipping ranking test")
                        return {}
                    
                    intent_response = await response.json()
                    valid_intent = intent_response.get('intent', {})
            except Exception as e:
                print(f"  [ERROR] Failed to extract intent: {e}")
                print(f"  [INFO] Skipping ranking test")
                return {}

        # Sample candidates for ranking
        sample_candidates = [
            {"id": f"result_{i}", "title": f"Laptop Review {i}", "description": f"Best programming laptop review number {i} for developers", "platform": "Web", "qualityScore": 0.8}
            for i in range(10)
        ]

        results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "start_memory": self.get_memory_usage(),
            "end_memory": None,
            "errors": []
        }

        start_time = time.time()

        async def make_request(session, request_id: int):
            try:
                payload = {
                    "intent": valid_intent,
                    "candidates": sample_candidates,
                    "options": {"numResults": 5}
                }

                request_start = time.time()
                async with session.post(
                    f"{self.base_url}/rank-results",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    elapsed = (time.time() - request_start) * 1000

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

            while time.time() - start_time < duration:
                while len(tasks) < concurrency:
                    task = asyncio.create_task(make_request(session, request_id))
                    tasks.append(task)
                    request_id += 1

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        results["end_memory"] = self.get_memory_usage()
        results["duration"] = time.time() - start_time
        results["rps"] = results["total_requests"] / results["duration"]

        if results["response_times"]:
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["median_response_time"] = statistics.median(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
            # FIX: Prevent off-by-one error in percentile calculation
            sorted_times = sorted(results["response_times"])
            p95_index = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
            results["p95_response_time"] = sorted_times[p95_index]

        self.print_stress_results("Ranking with Embeddings", results)
        return results
    
    def print_stress_results(self, test_name: str, results: Dict):
        """Print formatted stress test results"""
        print(f"\n{test_name} Results:")
        print(f"  Total requests: {results['total_requests']}")
        print(f"  Successful: {results['successful_requests']}")
        print(f"  Failed: {results['failed_requests']}")
        print(f"  Success rate: {(results['successful_requests']/results['total_requests']*100):.1f}%")
        print(f"  Duration: {results['duration']:.2f}s")
        print(f"  RPS: {results['rps']:.2f}")
        
        if results['response_times']:
            print(f"\n  Response Times:")
            print(f"    Average: {results['avg_response_time']:.2f}ms")
            print(f"    Median: {results['median_response_time']:.2f}ms")
            print(f"    95th percentile: {results['p95_response_time']:.2f}ms")
            print(f"    Max: {results['max_response_time']:.2f}ms")
        
        print(f"\n  Memory Usage:")
        print(f"    Start: {results['start_memory']['rss_mb']:.1f} MB")
        print(f"    End: {results['end_memory']['rss_mb']:.1f} MB")
        print(f"    Growth: {results['end_memory']['rss_mb'] - results['start_memory']['rss_mb']:.1f} MB")
        
        if results['errors']:
            print(f"\n  [WARN] Errors ({len(results['errors'])}):")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"    - {error}")
    
    async def run_all_stress_tests(self):
        """Run all stress tests"""
        print("\n" + "="*60)
        print("INTENT ENGINE STRESS TEST SUITE")
        print("="*60)

        try:
            # Test 1: Concurrent load
            await self.test_concurrent_intent_extraction(concurrency=50, duration=30)

            # Test 2: Memory leak
            await self.test_memory_leaks(iterations=500)

            # Test 3: Cache overflow (note: extract-intent uses rule-based, not embeddings)
            await self.test_cache_overflow(unique_queries=1000)

            # Test 4: Ranking with embeddings (uses cache)
            await self.test_ranking_with_embeddings(concurrency=30, duration=20)

            print("\n" + "="*60)
            print("ALL STRESS TESTS COMPLETED")
            print("="*60)

        except Exception as e:
            print(f"\n[ERROR] Stress tests failed: {e}")
            raise


def main():
    """Main entry point"""
    import sys
    
    # Check if locust is available
    try:
        import locust
        print("[OK] Locust is installed")
    except ImportError:
        print("[WARN] Locust not installed. Install with: pip install locust")
        print("Continuing with basic stress tests...")
    
    # Run stress tests
    suite = StressTestSuite()
    
    try:
        asyncio.run(suite.run_all_stress_tests())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
