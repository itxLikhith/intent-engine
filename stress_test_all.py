"""
Intent Engine - Comprehensive Stress Test Suite

This script performs comprehensive stress testing on:
1. PostgreSQL database connections and pooling
2. Redis caching performance
3. CORS configuration
4. API endpoints under concurrent load
5. System resource monitoring

Usage:
    python stress_test_all.py [--workers N] [--requests N] [--duration S]
"""

import asyncio
import aiohttp
import time
import statistics
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import threading
import subprocess
import psutil


# Test Configuration
@dataclass
class TestConfig:
    base_url: str = "http://localhost:8000"
    num_workers: int = 50
    num_requests: int = 500
    test_duration: int = 60  # seconds
    timeout: int = 30
    cors_test_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://example.com",
        "http://unauthorized.com"
    ])


@dataclass
class TestResult:
    test_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    requests_per_second: float = 0.0
    errors: List[str] = field(default_factory=list)
    status_codes: Dict[int, int] = field(default_factory=dict)
    success_rate: float = 0.0


class StressTester:
    """Comprehensive stress testing suite for Intent Engine"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.stop_event = asyncio.Event()
        self.db_connections_active = 0
        self.db_connections_max = 0
        self._lock = threading.Lock()

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=self.config.num_workers,
            limit_per_host=self.config.num_workers,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> tuple:
        """Make a single HTTP request and return (status, latency, error)"""
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.perf_counter()
        
        try:
            if method == "GET":
                async with self.session.get(url, headers=headers) as response:
                    latency = (time.perf_counter() - start_time) * 1000
                    return response.status, latency, None
            elif method == "POST":
                async with self.session.post(
                    url, 
                    json=data, 
                    headers={**(headers or {}), "Content-Type": "application/json"}
                ) as response:
                    latency = (time.perf_counter() - start_time) * 1000
                    return response.status, latency, None
            elif method == "OPTIONS":
                async with self.session.options(
                    url,
                    headers=headers
                ) as response:
                    latency = (time.perf_counter() - start_time) * 1000
                    return response.status, latency, None
        except asyncio.TimeoutError:
            latency = (time.perf_counter() - start_time) * 1000
            return 0, latency, "Timeout"
        except aiohttp.ClientError as e:
            latency = (time.perf_counter() - start_time) * 1000
            return 0, latency, str(e)
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return 0, latency, str(e)
        
        # Fallback for unknown methods
        return 0, (time.perf_counter() - start_time) * 1000, f"Unknown method: {method}"

    async def run_load_test(
        self,
        test_name: str,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        num_requests: Optional[int] = None
    ) -> TestResult:
        """Run a load test with specified parameters"""
        result = TestResult(test_name=test_name)
        latencies: List[float] = []
        num_requests = num_requests or self.config.num_requests
        
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        print(f"Endpoint: {method} {endpoint}")
        print(f"Requests: {num_requests}")
        print(f"Workers: {self.config.num_workers}")
        
        async def worker(worker_id: int):
            for i in range(num_requests // self.config.num_workers + 1):
                if i * self.config.num_workers + worker_id >= num_requests:
                    break
                status, latency, error = await self.make_request(
                    method, endpoint, data, headers
                )
                
                with self._lock:
                    result.total_requests += 1
                    if status >= 200 and status < 400:
                        result.successful_requests += 1
                    else:
                        result.failed_requests += 1
                        if error:
                            result.errors.append(f"Request {i}: {error}")
                    
                    result.status_codes[status] = result.status_codes.get(status, 0) + 1
                    latencies.append(latency)
        
        start_time = time.time()
        workers = [asyncio.create_task(worker(i)) for i in range(self.config.num_workers)]
        await asyncio.gather(*workers)
        duration = time.time() - start_time
        
        # Calculate statistics
        if latencies:
            result.avg_latency_ms = statistics.mean(latencies)
            result.min_latency_ms = min(latencies)
            result.max_latency_ms = max(latencies)
            sorted_latencies = sorted(latencies)
            result.p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2]
            result.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            result.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        result.requests_per_second = result.total_requests / duration
        result.success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
        
        self._print_result(result, duration)
        self.results.append(result)
        return result

    def _print_result(self, result: TestResult, duration: float):
        """Print test results in a formatted way"""
        print(f"\n{'-'*60}")
        print(f"Results: {result.test_name}")
        print(f"{'-'*60}")
        print(f"Duration: {duration:.2f}s")
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful: {result.successful_requests} ({result.success_rate:.1f}%)")
        print(f"Failed: {result.failed_requests}")
        print(f"Throughput: {result.requests_per_second:.2f} req/s")
        print(f"\nLatency (ms):")
        print(f"  Min: {result.min_latency_ms:.2f}")
        print(f"  Avg: {result.avg_latency_ms:.2f}")
        print(f"  P50: {result.p50_latency_ms:.2f}")
        print(f"  P95: {result.p95_latency_ms:.2f}")
        print(f"  P99: {result.p99_latency_ms:.2f}")
        print(f"  Max: {result.max_latency_ms:.2f}")
        print(f"\nStatus Codes: {result.status_codes}")
        if result.errors:
            print(f"\nErrors (first 5): {result.errors[:5]}")

    # ==================== POSTGRESQL TESTS ====================

    async def test_postgresql_connection_pool(self):
        """Test PostgreSQL connection pool under load"""
        print("\n\n" + "="*80)
        print("POSTGRESQL CONNECTION POOL STRESS TEST")
        print("="*80)
        
        # Test 1: Rapid sequential database operations
        await self.run_load_test(
            test_name="PostgreSQL - Sequential DB Operations",
            method="GET",
            endpoint="/campaigns",
            num_requests=200
        )
        
        # Test 2: Concurrent database writes
        test_data = {
            "name": f"Stress Test Campaign {int(time.time())}",
            "advertiser_id": 1,
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-12-31T23:59:59",
            "budget": 10000.0,
            "daily_budget": 100.0,
            "status": "active"
        }
        
        await self.run_load_test(
            test_name="PostgreSQL - Concurrent Writes",
            method="POST",
            endpoint="/campaigns",
            data=test_data,
            num_requests=100
        )
        
        # Test 3: Mixed read/write operations
        await self._test_mixed_db_operations()

    async def _test_mixed_db_operations(self):
        """Test mixed read/write database operations"""
        result = TestResult(test_name="PostgreSQL - Mixed Read/Write Operations")
        latencies: List[float] = []
        num_ops = 100
        
        print(f"\n{'='*60}")
        print(f"Running: {result.test_name}")
        print(f"{'='*60}")
        
        async def mixed_worker(worker_id: int):
            for i in range(num_ops // self.config.num_workers + 1):
                if i * self.config.num_workers + worker_id >= num_ops:
                    break
                
                # Alternate between read and write
                if i % 2 == 0:
                    status, latency, error = await self.make_request("GET", "/campaigns")
                else:
                    test_data = {
                        "name": f"Mixed Ops Campaign {worker_id}_{i}",
                        "advertiser_id": 1,
                        "budget": 1000.0,
                        "status": "active"
                    }
                    status, latency, error = await self.make_request(
                        "POST", "/campaigns", test_data
                    )
                
                with self._lock:
                    result.total_requests += 1
                    if status >= 200 and status < 400:
                        result.successful_requests += 1
                    else:
                        result.failed_requests += 1
                    result.status_codes[status] = result.status_codes.get(status, 0) + 1
                    latencies.append(latency)
        
        start_time = time.time()
        workers = [asyncio.create_task(mixed_worker(i)) for i in range(self.config.num_workers)]
        await asyncio.gather(*workers)
        duration = time.time() - start_time
        
        if latencies:
            result.avg_latency_ms = statistics.mean(latencies)
            result.min_latency_ms = min(latencies)
            result.max_latency_ms = max(latencies)
            sorted_latencies = sorted(latencies)
            result.p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2]
            result.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            result.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        result.requests_per_second = result.total_requests / duration
        result.success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
        
        self._print_result(result, duration)
        self.results.append(result)

    # ==================== REDIS TESTS ====================

    async def test_redis_caching(self):
        """Test Redis caching performance"""
        print("\n\n" + "="*80)
        print("REDIS CACHING STRESS TEST")
        print("="*80)
        
        # Test 1: Repeated identical queries (should hit cache)
        await self._test_query_caching()
        
        # Test 2: Cache invalidation under load
        await self._test_cache_invalidation()

    async def _test_query_caching(self):
        """Test query caching behavior"""
        result = TestResult(test_name="Redis - Query Caching Performance")
        latencies_first: List[float] = []
        latencies_cached: List[float] = []
        
        print(f"\n{'='*60}")
        print(f"Running: {result.test_name}")
        print(f"{'='*60}")
        
        query = "best laptop for programming"
        test_data = {
            "product": "search",
            "input": {"text": query}
        }
        
        # First request (cache miss)
        for i in range(10):
            status, latency, error = await self.make_request(
                "POST", "/extract-intent", test_data
            )
            latencies_first.append(latency)
        
        # Subsequent requests (should be cached)
        for i in range(50):
            status, latency, error = await self.make_request(
                "POST", "/extract-intent", test_data
            )
            latencies_cached.append(latency)
            
            with self._lock:
                result.total_requests += 1
                if status >= 200 and status < 400:
                    result.successful_requests += 1
                result.status_codes[status] = result.status_codes.get(status, 0) + 1
        
        # Calculate statistics
        all_latencies = latencies_first + latencies_cached
        if all_latencies:
            result.avg_latency_ms = statistics.mean(all_latencies)
            result.min_latency_ms = min(all_latencies)
            result.max_latency_ms = max(all_latencies)
        
        result.success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
        
        print(f"\nFirst Request Avg (cache miss): {statistics.mean(latencies_first):.2f}ms")
        print(f"Cached Request Avg (cache hit): {statistics.mean(latencies_cached):.2f}ms")
        if latencies_first and latencies_cached:
            speedup = statistics.mean(latencies_first) / statistics.mean(latencies_cached) if statistics.mean(latencies_cached) > 0 else 0
            print(f"Cache Speedup: {speedup:.2f}x")
        
        self._print_result(result, 0)
        self.results.append(result)

    async def _test_cache_invalidation(self):
        """Test cache behavior with varying queries"""
        result = TestResult(test_name="Redis - Cache Invalidation Under Load")
        latencies: List[float] = []
        
        print(f"\n{'='*60}")
        print(f"Running: {result.test_name}")
        print(f"{'='*60}")
        
        queries = [
            "best laptop for programming",
            "best laptop for gaming",
            "best laptop for video editing",
            "best laptop for students",
            "best laptop for business"
        ]
        
        for query in queries:
            test_data = {
                "product": "search",
                "input": {"text": query}
            }
            for i in range(20):
                status, latency, error = await self.make_request(
                    "POST", "/extract-intent", test_data
                )
                latencies.append(latency)
                
                with self._lock:
                    result.total_requests += 1
                    if status >= 200 and status < 400:
                        result.successful_requests += 1
                    result.status_codes[status] = result.status_codes.get(status, 0) + 1
        
        if latencies:
            result.avg_latency_ms = statistics.mean(latencies)
            result.min_latency_ms = min(latencies)
            result.max_latency_ms = max(latencies)
        
        result.success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
        self._print_result(result, 0)
        self.results.append(result)

    # ==================== CORS TESTS ====================

    async def test_cors_configuration(self):
        """Test CORS configuration"""
        print("\n\n" + "="*80)
        print("CORS CONFIGURATION STRESS TEST")
        print("="*80)
        
        await self._test_cors_origins()
        await self._test_cors_headers()
        await self._test_cors_preflight()

    async def _test_cors_origins(self):
        """Test CORS with different origins"""
        result = TestResult(test_name="CORS - Origin Validation")
        
        print(f"\n{'='*60}")
        print(f"Running: {result.test_name}")
        print(f"{'='*60}")
        
        for origin in self.config.cors_test_origins:
            headers = {"Origin": origin}
            status, latency, error = await self.make_request(
                "GET", "/", headers=headers
            )
            
            result.total_requests += 1
            if status >= 200 and status < 400:
                result.successful_requests += 1
            else:
                result.failed_requests += 1
            
            result.status_codes[status] = result.status_codes.get(status, 0) + 1
            print(f"Origin: {origin}")
            print(f"  Status: {status}, Latency: {latency:.2f}ms")
        
        result.success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
        self._print_result(result, 0)
        self.results.append(result)

    async def _test_cors_headers(self):
        """Test CORS headers handling"""
        result = TestResult(test_name="CORS - Custom Headers")
        
        print(f"\n{'='*60}")
        print(f"Running: {result.test_name}")
        print(f"{'='*60}")
        
        test_headers = [
            {"Authorization": "Bearer token123"},
            {"X-Requested-With": "XMLHttpRequest"},
            {"Content-Type": "application/json"},
            {"X-Custom-Header": "test-value"},
        ]
        
        for headers in test_headers:
            headers["Origin"] = "http://localhost:3000"
            status, latency, error = await self.make_request(
                "GET", "/", headers=headers
            )
            
            result.total_requests += 1
            if status >= 200 and status < 400:
                result.successful_requests += 1
            
            print(f"Headers: {headers}")
            print(f"  Status: {status}, Latency: {latency:.2f}ms")
        
        result.success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
        self._print_result(result, 0)
        self.results.append(result)

    async def _test_cors_preflight(self):
        """Test CORS preflight OPTIONS requests"""
        result = TestResult(test_name="CORS - Preflight (OPTIONS)")
        latencies: List[float] = []
        
        print(f"\n{'='*60}")
        print(f"Running: {result.test_name}")
        print(f"{'='*60}")
        
        endpoints = [
            "/extract-intent",
            "/campaigns",
            "/match-ads",
            "/rank-results"
        ]
        
        for endpoint in endpoints:
            for i in range(10):
                status, latency, error = await self.make_request(
                    "OPTIONS", endpoint,
                    headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type,Authorization"
                    }
                )
                latencies.append(latency)
                
                with self._lock:
                    result.total_requests += 1
                    if status >= 200 and status < 400:
                        result.successful_requests += 1
                    result.status_codes[status] = result.status_codes.get(status, 0) + 1
        
        if latencies:
            result.avg_latency_ms = statistics.mean(latencies)
        
        result.success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
        self._print_result(result, 0)
        self.results.append(result)

    # ==================== API ENDPOINT TESTS ====================

    async def test_api_endpoints(self):
        """Test all API endpoints under load"""
        print("\n\n" + "="*80)
        print("API ENDPOINTS STRESS TEST")
        print("="*80)
        
        # Test 1: Intent extraction
        await self.run_load_test(
            test_name="API - Intent Extraction",
            method="POST",
            endpoint="/extract-intent",
            data={"product": "search", "input": {"text": "best laptop for programming"}},
            num_requests=100
        )
        
        # Test 2: Health check
        await self.run_load_test(
            test_name="API - Health Check",
            method="GET",
            endpoint="/",
            num_requests=200
        )
        
        # Test 3: Campaign listing
        await self.run_load_test(
            test_name="API - Campaign List",
            method="GET",
            endpoint="/campaigns",
            num_requests=100
        )
        
        # Test 4: Status endpoint
        await self.run_load_test(
            test_name="API - Status",
            method="GET",
            endpoint="/status",
            num_requests=100
        )
        
        # Test 5: Sustained load test
        await self._test_sustained_load()

    async def _test_sustained_load(self):
        """Test sustained load over time"""
        result = TestResult(test_name="API - Sustained Load (30s)")
        latencies: List[float] = []
        duration = 30  # seconds
        
        print(f"\n{'='*60}")
        print(f"Running: {result.test_name}")
        print(f"{'='*60}")
        print(f"Duration: {duration}s with {self.config.num_workers} workers")
        
        start_time = time.time()
        
        async def sustained_worker(worker_id: int):
            while time.time() - start_time < duration:
                status, latency, error = await self.make_request(
                    "GET", "/status"
                )
                
                with self._lock:
                    result.total_requests += 1
                    if status >= 200 and status < 400:
                        result.successful_requests += 1
                    else:
                        result.failed_requests += 1
                    result.status_codes[status] = result.status_codes.get(status, 0) + 1
                    latencies.append(latency)
                
                await asyncio.sleep(0.1)  # Small delay between requests
        
        workers = [asyncio.create_task(sustained_worker(i)) for i in range(self.config.num_workers)]
        await asyncio.gather(*workers)
        actual_duration = time.time() - start_time
        
        if latencies:
            result.avg_latency_ms = statistics.mean(latencies)
            result.min_latency_ms = min(latencies)
            result.max_latency_ms = max(latencies)
            sorted_latencies = sorted(latencies)
            result.p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2]
            result.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            result.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        result.requests_per_second = result.total_requests / actual_duration
        result.success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
        
        self._print_result(result, actual_duration)
        self.results.append(result)

    # ==================== SYSTEM MONITORING ====================

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024 ** 3),
            "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        }

    async def monitor_system_resources(self, duration: int):
        """Monitor system resources during tests"""
        print("\n\n" + "="*80)
        print("SYSTEM RESOURCE MONITORING")
        print("="*80)
        
        metrics_samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            metrics = self.get_system_metrics()
            metrics_samples.append(metrics)
            print(f"CPU: {metrics['cpu_percent']:.1f}% | "
                  f"Memory: {metrics['memory_percent']:.1f}% "
                  f"({metrics['memory_used_gb']:.2f}/{metrics['memory_total_gb']:.2f} GB)")
            await asyncio.sleep(2)
        
        if metrics_samples:
            avg_cpu = statistics.mean([m['cpu_percent'] for m in metrics_samples])
            avg_memory = statistics.mean([m['memory_percent'] for m in metrics_samples])
            print(f"\nAverage CPU: {avg_cpu:.1f}%")
            print(f"Average Memory: {avg_memory:.1f}%")
        
        return metrics_samples

    # ==================== REPORT GENERATION ====================

    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("\n" + "="*80)
        report.append("COMPREHENSIVE STRESS TEST REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Base URL: {self.config.base_url}")
        report.append(f"Workers: {self.config.num_workers}")
        report.append("")
        
        # Summary
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.failed_requests for r in self.results)
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        
        report.append("SUMMARY")
        report.append("-"*60)
        report.append(f"Total Tests: {len(self.results)}")
        report.append(f"Total Requests: {total_requests}")
        report.append(f"Successful: {total_successful}")
        report.append(f"Failed: {total_failed}")
        report.append(f"Overall Success Rate: {overall_success_rate:.1f}%")
        report.append("")
        
        # Detailed Results
        report.append("DETAILED RESULTS")
        report.append("-"*60)
        
        for result in self.results:
            report.append(f"\n{result.test_name}:")
            report.append(f"  Requests: {result.total_requests} | "
                         f"Success: {result.success_rate:.1f}% | "
                         f"Throughput: {result.requests_per_second:.1f} req/s")
            report.append(f"  Latency (ms) - Avg: {result.avg_latency_ms:.2f} | "
                         f"P95: {result.p95_latency_ms:.2f} | "
                         f"P99: {result.p99_latency_ms:.2f}")
            if result.errors:
                report.append(f"  Errors: {len(result.errors)}")
        
        # Recommendations
        report.append("\n" + "="*80)
        report.append("RECOMMENDATIONS")
        report.append("="*80)
        
        if overall_success_rate < 99:
            report.append("[WARNING] High error rate detected - investigate failing requests")
        
        high_latency_tests = [r for r in self.results if r.p99_latency_ms > 1000]
        if high_latency_tests:
            report.append("[WARNING] High P99 latency detected in:")
            for r in high_latency_tests:
                report.append(f"  - {r.test_name}: {r.p99_latency_ms:.2f}ms")
        
        low_throughput_tests = [r for r in self.results if r.requests_per_second < 10]
        if low_throughput_tests:
            report.append("[WARNING] Low throughput detected in:")
            for r in low_throughput_tests:
                report.append(f"  - {r.test_name}: {r.requests_per_second:.1f} req/s")
        
        if overall_success_rate >= 99 and not high_latency_tests:
            report.append("[OK] System performing well under load")
        
        report.append("")
        return "\n".join(report)

    # ==================== MAIN RUNNER ====================

    async def run_all_tests(self):
        """Run all stress tests"""
        print("\n" + "="*80)
        print("INTENT ENGINE - COMPREHENSIVE STRESS TEST SUITE")
        print("="*80)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Configuration:")
        print(f"  Base URL: {self.config.base_url}")
        print(f"  Workers: {self.config.num_workers}")
        print(f"  Requests per test: {self.config.num_requests}")
        
        # Run all test suites
        await self.test_postgresql_connection_pool()
        await self.test_redis_caching()
        await self.test_cors_configuration()
        await self.test_api_endpoints()
        
        # Monitor system resources
        await self.monitor_system_resources(duration=10)
        
        # Generate report
        report = self.generate_report()
        print(report)
        
        # Save report to file
        report_file = f"stress_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_file}")
        
        return self.results


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Intent Engine Stress Test Suite')
    parser.add_argument('--workers', type=int, default=50, help='Number of concurrent workers')
    parser.add_argument('--requests', type=int, default=500, help='Number of requests per test')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--url', type=str, default='http://localhost:8000', help='Base URL')
    
    args = parser.parse_args()
    
    config = TestConfig(
        base_url=args.url,
        num_workers=args.workers,
        num_requests=args.requests,
        test_duration=args.duration
    )
    
    async with StressTester(config) as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
