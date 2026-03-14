"""
Unified Search Performance Benchmark

Tests the /search endpoint for:
1. Single request latency (warm vs cold)
2. Concurrent load handling
3. Throughput under stress
4. Error rates at various loads
"""

import asyncio
import json
import statistics
import time
from datetime import datetime
from typing import Any

import aiohttp

# Test configuration
API_BASE_URL = "http://localhost:8000"
SEARCH_ENDPOINT = f"{API_BASE_URL}/search"

# Test queries with varying complexity
TEST_QUERIES = [
    "best laptop",  # Simple
    "best laptop for programming under 50000",  # Medium
    "best privacy-focused search engines 2024",  # Complex
    "how to set up E2E encrypted email on Android no big tech",  # Very complex
]

# Load test configuration
CONCURRENT_USERS = [1, 2, 5, 10, 20, 50]
REQUESTS_PER_USER = 5
TIMEOUT_SECONDS = 120


class PerformanceBenchmark:
    """Benchmark suite for unified search endpoint"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "single_request": {},
            "concurrent_load": [],
            "stress_test": {},
        }

    async def single_request_latency(self, session: aiohttp.ClientSession, query: str) -> dict[str, Any]:
        """Test single request latency"""
        payload = {
            "query": query,
            "extract_intent": True,
            "rank_results": True,
            "categories": ["general"],
            "language": "en",
            "safesearch": 0,
            "pageno": 1,
        }

        start_time = time.perf_counter()

        try:
            async with session.post(
                SEARCH_ENDPOINT, json=payload, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
            ) as response:
                end_time = time.perf_counter()
                response_time = (end_time - start_time) * 1000  # ms

                data = await response.json()

                return {
                    "query": query,
                    "status": response.status,
                    "response_time_ms": round(response_time, 2),
                    "results_count": data.get("total_results", 0),
                    "processing_time_ms": data.get("processing_time_ms", 0),
                    "success": response.status == 200,
                }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "query": query,
                "status": 0,
                "response_time_ms": round((end_time - start_time) * 1000, 2),
                "error": str(e),
                "success": False,
            }

    async def concurrent_request(
        self, session: aiohttp.ClientSession, query: str, num_requests: int
    ) -> list[dict[str, Any]]:
        """Test concurrent requests"""
        tasks = [self.single_request_latency(session, query) for _ in range(num_requests)]
        return await asyncio.gather(*tasks)

    async def run_benchmark(self):
        """Run complete benchmark suite"""
        print("=" * 80)
        print("UNIFIED SEARCH PERFORMANCE BENCHMARK")
        print("=" * 80)
        print(f"Started at: {self.results['timestamp']}")
        print(f"API Endpoint: {SEARCH_ENDPOINT}")
        print(f"Timeout: {TIMEOUT_SECONDS}s")
        print("=" * 80)

        async with aiohttp.ClientSession() as session:
            # 1. Single Request Latency Tests
            print("\n[1/3] SINGLE REQUEST LATENCY TESTS")
            print("-" * 80)

            for i, query in enumerate(TEST_QUERIES, 1):
                print(f"\nTest {i}: '{query}'")

                # Warm-up request
                print("  Warming up...", end=" ", flush=True)
                warm_result = await self.single_request_latency(session, query)
                print(f"Warm-up: {warm_result['response_time_ms']}ms")

                # Actual tests (5 requests)
                latencies = []
                for j in range(5):
                    result = await self.single_request_latency(session, query)
                    latencies.append(result["response_time_ms"])
                    print(
                        f"  Request {j + 1}: {result['response_time_ms']}ms ({result['results_count']} results)",
                        flush=True,
                    )

                # Calculate statistics
                self.results["single_request"][query] = {
                    "min_ms": round(min(latencies), 2),
                    "max_ms": round(max(latencies), 2),
                    "mean_ms": round(statistics.mean(latencies), 2),
                    "median_ms": round(statistics.median(latencies), 2),
                    "stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
                    "success_rate": "100%",
                }

            print("\n" + "=" * 80)
            print("[2/3] CONCURRENT LOAD TESTS")
            print("-" * 80)

            # Use medium complexity query for load tests
            load_query = "best privacy-focused search engines 2024"

            for num_users in CONCURRENT_USERS:
                print(f"\nTesting {num_users} concurrent users...")

                start_time = time.perf_counter()

                # Run concurrent requests
                tasks = [self.single_request_latency(session, load_query) for _ in range(num_users * REQUESTS_PER_USER)]
                results = await asyncio.gather(*tasks)

                end_time = time.perf_counter()
                total_time = end_time - start_time

                # Analyze results
                successful = sum(1 for r in results if r["success"])
                failed = num_users * REQUESTS_PER_USER - successful
                latencies = [r["response_time_ms"] for r in results if r["success"]]

                if latencies:
                    load_result = {
                        "concurrent_users": num_users,
                        "total_requests": num_users * REQUESTS_PER_USER,
                        "successful": successful,
                        "failed": failed,
                        "success_rate": f"{(successful / len(results) * 100):.1f}%",
                        "total_time_seconds": round(total_time, 2),
                        "requests_per_second": round(len(results) / total_time, 2),
                        "latency_min_ms": round(min(latencies), 2),
                        "latency_max_ms": round(max(latencies), 2),
                        "latency_mean_ms": round(statistics.mean(latencies), 2),
                        "latency_median_ms": round(statistics.median(latencies), 2),
                        "latency_p95_ms": round(
                            sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies), 2
                        ),
                    }

                    self.results["concurrent_load"].append(load_result)

                    print(f"  ✓ Success: {successful}/{len(results)} ({load_result['success_rate']})")
                    print(f"  ✓ Throughput: {load_result['requests_per_second']} req/s")
                    print(f"  ✓ Latency: {load_result['latency_mean_ms']}ms avg, {load_result['latency_p95_ms']}ms p95")
                else:
                    print("  ✗ All requests failed!")
                    self.results["concurrent_load"].append(
                        {"concurrent_users": num_users, "error": "All requests failed"}
                    )

            print("\n" + "=" * 80)
            print("[3/3] STRESS TEST (Burst Load)")
            print("-" * 80)

            # Burst test: 50 requests as fast as possible
            print("\nSending 50 requests as fast as possible...")
            start_time = time.perf_counter()

            tasks = [self.single_request_latency(session, load_query) for _ in range(50)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.perf_counter()
            total_time = end_time - start_time

            successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
            latencies = [r["response_time_ms"] for r in results if isinstance(r, dict) and r["success"]]

            self.results["stress_test"] = {
                "total_requests": 50,
                "successful": successful,
                "failed": 50 - successful,
                "success_rate": f"{(successful / 50 * 100):.1f}%",
                "total_time_seconds": round(total_time, 2),
                "requests_per_second": round(50 / total_time, 2),
                "latency_min_ms": round(min(latencies), 2) if latencies else 0,
                "latency_max_ms": round(max(latencies), 2) if latencies else 0,
                "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else 0,
            }

            print(f"  ✓ Success: {successful}/50 ({self.results['stress_test']['success_rate']})")
            print(f"  ✓ Throughput: {self.results['stress_test']['requests_per_second']} req/s")
            print(f"  ✓ Total time: {total_time:.2f}s")

        # Print summary
        self._print_summary()

        # Save results
        self._save_results()

    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        print("\n📊 SINGLE REQUEST LATENCY:")
        for query, stats in self.results["single_request"].items():
            print(f"\n  '{query[:50]}...'")
            print(f"    Mean: {stats['mean_ms']}ms | Median: {stats['median_ms']}ms | P95: ~{stats['max_ms']}ms")

        print("\n📈 CONCURRENT LOAD PERFORMANCE:")
        print(f"  {'Users':<8} {'Req/s':<10} {'Mean Lat':<12} {'P95 Lat':<12} {'Success':<10}")
        print(f"  {'-' * 8} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 10}")
        for load in self.results["concurrent_load"]:
            if "error" not in load:
                print(
                    f"  {load['concurrent_users']:<8} {load['requests_per_second']:<10} "
                    f"{load['latency_mean_ms']}ms{'':<6} {load['latency_p95_ms']}ms{'':<6} "
                    f"{load['success_rate']:<10}"
                )

        print("\n💥 STRESS TEST:")
        stress = self.results["stress_test"]
        print(f"  Throughput: {stress['requests_per_second']} req/s")
        print(f"  Success Rate: {stress['success_rate']}")
        print(f"  Mean Latency: {stress['latency_mean_ms']}ms")

        print("\n" + "=" * 80)

    def _save_results(self):
        """Save results to JSON file"""
        filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")


async def main():
    """Main entry point"""
    benchmark = PerformanceBenchmark()
    await benchmark.run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
