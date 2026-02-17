"""
Intent Engine - Database Connection Pool Stress Test

This module tests database connection pool behavior under load:
- Connection pool exhaustion scenarios
- Concurrent database operations
- Connection cleanup and release
- Transaction handling under stress
"""

import asyncio
import statistics
import time
from datetime import datetime
from typing import Dict

import aiohttp

BASE_URL = "http://localhost:8000"


class DatabaseStressTest:
    """Database connection pool stress testing"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = {}

    async def test_campaign_crud_operations(self, iterations: int = 100):
        """Test create, read, update, delete operations on campaigns"""
        print(f"\n{'='*60}")
        print("Database Stress Test: Campaign CRUD Operations")
        print(f"Iterations: {iterations}")
        print(f"{'='*60}")

        results = {
            "creates": {"success": 0, "failed": 0, "times": []},
            "reads": {"success": 0, "failed": 0, "times": []},
            "updates": {"success": 0, "failed": 0, "times": []},
            "deletes": {"success": 0, "failed": 0, "times": []},
        }

        created_ids = []

        async with aiohttp.ClientSession() as session:
            # CREATE operations
            print("  Running CREATE operations...")
            for i in range(iterations):
                payload = {
                    "name": f"Stress Test Campaign {i}-{int(time.time())}",
                    "advertiser_id": 1,  # Assuming advertiser 1 exists
                    "budget": 1000.0,
                    "daily_budget": 100.0,
                    "start_date": datetime.utcnow().isoformat(),
                    "status": "active",
                }

                start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/campaigns", json=payload, timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        elapsed = (time.time() - start) * 1000
                        if response.status == 200:
                            data = await response.json()
                            created_ids.append(data.get("id"))
                            results["creates"]["success"] += 1
                        else:
                            results["creates"]["failed"] += 1
                        results["creates"]["times"].append(elapsed)
                except Exception:
                    results["creates"]["failed"] += 1
                    results["creates"]["times"].append((time.time() - start) * 1000)

            # READ operations
            print("  Running READ operations...")
            for i in range(iterations):
                start = time.time()
                try:
                    async with session.get(
                        f"{self.base_url}/campaigns", timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        elapsed = (time.time() - start) * 1000
                        if response.status == 200:
                            results["reads"]["success"] += 1
                        else:
                            results["reads"]["failed"] += 1
                        results["reads"]["times"].append(elapsed)
                except Exception:
                    results["reads"]["failed"] += 1
                    results["reads"]["times"].append((time.time() - start) * 1000)

            # UPDATE operations (on first 10 created)
            print("  Running UPDATE operations...")
            for i, campaign_id in enumerate(created_ids[:10]):
                payload = {"name": f"Updated Campaign {i}", "budget": 1500.0, "status": "paused"}

                start = time.time()
                try:
                    async with session.put(
                        f"{self.base_url}/campaigns/{campaign_id}",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        elapsed = (time.time() - start) * 1000
                        if response.status == 200:
                            results["updates"]["success"] += 1
                        else:
                            results["updates"]["failed"] += 1
                        results["updates"]["times"].append(elapsed)
                except Exception:
                    results["updates"]["failed"] += 1
                    results["updates"]["times"].append((time.time() - start) * 1000)

            # DELETE operations (cleanup)
            print("  Running DELETE operations...")
            for campaign_id in created_ids[:10]:
                start = time.time()
                try:
                    async with session.delete(
                        f"{self.base_url}/campaigns/{campaign_id}", timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        elapsed = (time.time() - start) * 1000
                        if response.status == 200:
                            results["deletes"]["success"] += 1
                        else:
                            results["deletes"]["failed"] += 1
                        results["deletes"]["times"].append(elapsed)
                except Exception:
                    results["deletes"]["failed"] += 1
                    results["deletes"]["times"].append((time.time() - start) * 1000)

        # Print results
        self._print_crud_results("Campaign CRUD", results)
        return results

    async def test_concurrent_database_access(self, concurrency: int = 50, duration: int = 30):
        """Test concurrent database read/write operations"""
        print(f"\n{'='*60}")
        print("Database Stress Test: Concurrent Database Access")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
        print(f"{'='*60}")

        results = {"total_requests": 0, "successful": 0, "failed": 0, "response_times": [], "errors": []}

        start_time = time.time()

        async def make_request(session, request_id: int):
            try:
                # Mix of read operations only (ads endpoint may have validation issues)
                # Focus on testing database connection pool, not API validation
                async with session.get(
                    f"{self.base_url}/campaigns", timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    elapsed = (time.time() - start_time) * 1000

                    results["total_requests"] += 1
                    results["response_times"].append(elapsed)

                    if response.status == 200:
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"Request {request_id}: HTTP {response.status}")

            except Exception as e:
                results["total_requests"] += 1
                results["failed"] += 1
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

        # Calculate statistics
        results["duration"] = time.time() - start_time
        results["rps"] = results["total_requests"] / results["duration"]

        if results["response_times"]:
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["median_response_time"] = statistics.median(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
            results["p95_response_time"] = sorted(results["response_times"])[int(len(results["response_times"]) * 0.95)]

        self._print_concurrent_results("Concurrent Database Access", results)
        return results

    async def test_connection_cleanup(self, iterations: int = 200):
        """Test that database connections are properly cleaned up"""
        print(f"\n{'='*60}")
        print("Database Stress Test: Connection Cleanup")
        print(f"Iterations: {iterations}")
        print(f"{'='*60}")

        results = {"total": 0, "success": 0, "failed": 0, "times": []}

        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                start = time.time()
                try:
                    # Rapid fire read operations
                    async with session.get(
                        f"{self.base_url}/campaigns", timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        elapsed = (time.time() - start) * 1000
                        results["total"] += 1
                        results["times"].append(elapsed)

                        if response.status == 200:
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                except Exception:
                    results["total"] += 1
                    results["failed"] += 1
                    results["times"].append((time.time() - start) * 1000)

        # Analyze connection cleanup
        print("\nConnection Cleanup Results:")
        print(f"  Total requests: {results['total']}")
        print(f"  Successful: {results['success']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Success rate: {(results['success']/results['total']*100):.1f}%")

        if results["times"]:
            print("\n  Response Times:")
            print(f"    Average: {statistics.mean(results['times']):.2f}ms")
            print(f"    Median: {statistics.median(results['times']):.2f}ms")
            print(f"    Max: {max(results['times']):.2f}ms")

            # Check for connection exhaustion pattern
            # (increasing response times indicate connection pool issues)
            first_quarter_avg = statistics.mean(results["times"][: len(results["times"]) // 4])
            last_quarter_avg = statistics.mean(results["times"][-len(results["times"]) // 4 :])

            if last_quarter_avg > first_quarter_avg * 2:
                print(f"\n  [WARN] Response times increased {last_quarter_avg/first_quarter_avg:.1f}x")
                print("  [WARN] Possible connection pool exhaustion detected!")
            else:
                print("\n  [OK] Connection pool stable (no exhaustion detected)")

        return results

    async def test_transaction_rollback(self):
        """Test transaction rollback behavior"""
        print(f"\n{'='*60}")
        print("Database Stress Test: Transaction Rollback")
        print(f"{'='*60}")

        # Note: This test requires API endpoints that support transactions
        # For now, we'll test error handling with invalid data
        results = {"invalid_requests": 0, "proper_errors": 0, "server_errors": 0}

        async with aiohttp.ClientSession() as session:
            # Try to create campaigns with invalid data
            invalid_payloads = [
                {"name": "", "advertiser_id": 99999},  # Invalid advertiser
                {"name": "Test", "budget": -100},  # Negative budget
                {"name": "Test" * 1000},  # Very long name
            ]

            for payload in invalid_payloads:
                for i in range(10):
                    try:
                        async with session.post(
                            f"{self.base_url}/campaigns", json=payload, timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            results["invalid_requests"] += 1
                            if response.status in [400, 422]:
                                results["proper_errors"] += 1
                            elif response.status == 500:
                                results["server_errors"] += 1
                    except Exception:
                        results["invalid_requests"] += 1

        print("\nTransaction Rollback Results:")
        print(f"  Invalid requests: {results['invalid_requests']}")
        print(f"  Proper validation errors (400/422): {results['proper_errors']}")
        print(f"  Server errors (500): {results['server_errors']}")

        if results["server_errors"] > 0:
            print("\n  [WARN] Server errors detected during invalid input!")
        else:
            print("\n  [OK] Proper error handling (no server errors)")

        return results

    def _print_crud_results(self, test_name: str, results: Dict):
        """Print CRUD test results"""
        print(f"\n{test_name} Results:")
        for operation, data in results.items():
            success = data.get("success", 0)
            failed = data.get("failed", 0)
            total = success + failed
            times = data.get("times", [])

            print(f"\n  {operation.upper()}:")
            print(f"    Total: {total}")
            print(f"    Success: {success}")
            print(f"    Failed: {failed}")
            if total > 0:
                print(f"    Success rate: {(success/total*100):.1f}%")
            if times:
                print(f"    Avg time: {statistics.mean(times):.2f}ms")
                print(f"    Max time: {max(times):.2f}ms")

    def _print_concurrent_results(self, test_name: str, results: Dict):
        """Print concurrent test results"""
        print(f"\n{test_name} Results:")
        print(f"  Total requests: {results['total_requests']}")
        print(f"  Successful: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Success rate: {(results['successful']/results['total_requests']*100):.1f}%")
        print(f"  Duration: {results['duration']:.2f}s")
        print(f"  RPS: {results['rps']:.2f}")

        if results.get("response_times"):
            print("\n  Response Times:")
            print(f"    Average: {results['avg_response_time']:.2f}ms")
            print(f"    Median: {results['median_response_time']:.2f}ms")
            print(f"    95th percentile: {results['p95_response_time']:.2f}ms")
            print(f"    Max: {results['max_response_time']:.2f}ms")

        if results.get("errors") and len(results["errors"]) > 0:
            print(f"\n  [WARN] Errors ({len(results['errors'])}):")
            for error in results["errors"][:5]:
                print(f"    - {error}")

    async def run_all_database_tests(self):
        """Run all database stress tests"""
        print("\n" + "=" * 60)
        print("DATABASE CONNECTION POOL STRESS TEST SUITE")
        print("=" * 60)

        try:
            # Test 1: CRUD operations
            await self.test_campaign_crud_operations(iterations=50)

            # Test 2: Concurrent access
            await self.test_concurrent_database_access(concurrency=30, duration=20)

            # Test 3: Connection cleanup
            await self.test_connection_cleanup(iterations=200)

            # Test 4: Transaction rollback
            await self.test_transaction_rollback()

            print("\n" + "=" * 60)
            print("ALL DATABASE TESTS COMPLETED")
            print("=" * 60)

        except Exception as e:
            print(f"\n[ERROR] Database tests failed: {e}")
            raise


def main():
    """Main entry point"""
    test = DatabaseStressTest()

    try:
        asyncio.run(test.run_all_database_tests())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Tests failed: {e}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
