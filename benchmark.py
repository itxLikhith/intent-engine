"""
Intent Engine - Performance Benchmark Suite

This script benchmarks the optimized Intent Engine and compares performance
before and after optimizations.
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any
import requests
import json

BASE_URL = "http://localhost:8000"


class PerformanceBenchmark:
    """Benchmark suite for Intent Engine API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
    
    def benchmark_intent_extraction(self, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark intent extraction endpoint"""
        print(f"\nBenchmarking Intent Extraction ({iterations} iterations)...")
        
        test_queries = [
            "How to set up encrypted email?",
            "Best privacy-focused browser for Linux",
            "Compare secure messaging apps",
            "Setup VPN on Ubuntu server",
            "Best open source password manager",
        ]
        
        times = []
        for i in range(iterations):
            query = test_queries[i % len(test_queries)]
            
            payload = {
                "product": "search",
                "input": {"text": query},
                "context": {"sessionId": f"bench-{i}", "userLocale": "en-US"}
            }
            
            start = time.time()
            response = requests.post(f"{self.base_url}/extract-intent", json=payload)
            elapsed = (time.time() - start) * 1000
            
            if response.status_code == 200:
                times.append(elapsed)
            else:
                print(f"  Failed: {response.status_code}")
        
        return self._calculate_stats(times)
    
    def benchmark_ranking(self, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark ranking endpoint"""
        print(f"\nBenchmarking Result Ranking ({iterations} iterations)...")
        
        # Create test candidates
        candidates = []
        for i in range(10):
            candidates.append({
                "id": f"result-{i}",
                "title": f"Privacy Tool {i}",
                "description": f"Secure privacy solution number {i} with encryption",
                "platform": "web",
                "provider": f"Provider-{i}",
                "qualityScore": 0.5 + (i * 0.05),
                "tags": ["privacy", "security", "encryption"]
            })
        
        # First extract an intent
        intent_response = requests.post(f"{self.base_url}/extract-intent", json={
            "product": "search",
            "input": {"text": "privacy focused tools"},
            "context": {"sessionId": "bench-ranking", "userLocale": "en-US"}
        })
        
        intent = intent_response.json()["intent"]
        
        times = []
        for i in range(iterations):
            payload = {
                "intent": intent,
                "candidates": candidates,
                "options": {}
            }
            
            start = time.time()
            response = requests.post(f"{self.base_url}/rank-results", json=payload)
            elapsed = (time.time() - start) * 1000
            
            if response.status_code == 200:
                times.append(elapsed)
            else:
                print(f"  Failed: {response.status_code}")
        
        return self._calculate_stats(times)
    
    def benchmark_url_ranking(self, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark URL ranking endpoint"""
        print(f"\nBenchmarking URL Ranking ({iterations} iterations)...")
        
        urls = [
            "https://protonmail.com",
            "https://duckduckgo.com",
            "https://firefox.com",
            "https://signal.org",
            "https://tutanota.com",
            "https://privacytools.io",
            "https://eff.org",
            "https://wikipedia.org",
            "https://github.com",
            "https://gitlab.com",
        ]
        
        times = []
        for i in range(iterations):
            payload = {
                "query": "privacy email service",
                "urls": urls,
                "options": {"exclude_big_tech": True}
            }
            
            start = time.time()
            response = requests.post(f"{self.base_url}/rank-urls", json=payload)
            elapsed = (time.time() - start) * 1000
            
            if response.status_code == 200:
                times.append(elapsed)
                result = response.json()
                if i == 0:
                    print(f"  Cache hit rate: {result.get('cache_hit_rate', 0)*100:.1f}%")
            else:
                print(f"  Failed: {response.status_code}")
        
        return self._calculate_stats(times)
    
    def benchmark_ad_matching(self, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark ad matching endpoint"""
        print(f"\nBenchmarking Ad Matching ({iterations} iterations)...")
        
        # First extract an intent
        intent_response = requests.post(f"{self.base_url}/extract-intent", json={
            "product": "search",
            "input": {"text": "privacy focused email service"},
            "context": {"sessionId": "bench-ads", "userLocale": "en-US"}
        })
        
        intent = intent_response.json()["intent"]
        
        # Create test ads
        ads = []
        for i in range(5):
            ads.append({
                "id": f"ad-{i}",
                "title": f"Privacy Solution {i}",
                "description": f"Secure and private service {i}",
                "targetingConstraints": {},
                "forbiddenDimensions": [],
                "qualityScore": 0.7 + (i * 0.05),
                "ethicalTags": ["privacy", "encryption"],
                "advertiser": f"advertiser-{i}"
            })
        
        times = []
        for i in range(iterations):
            payload = {
                "intent": intent,
                "ad_inventory": ads,
                "config": {"minThreshold": 0.3, "topK": 3}
            }
            
            start = time.time()
            response = requests.post(f"{self.base_url}/match-ads", json=payload)
            elapsed = (time.time() - start) * 1000
            
            if response.status_code == 200:
                times.append(elapsed)
            else:
                print(f"  Failed: {response.status_code}")
        
        return self._calculate_stats(times)
    
    def _calculate_stats(self, times: List[float]) -> Dict[str, Any]:
        """Calculate statistics from timing data"""
        if not times:
            return {"error": "No successful requests"}
        
        return {
            "count": len(times),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "mean_ms": round(statistics.mean(times), 2),
            "median_ms": round(statistics.median(times), 2),
            "stdev_ms": round(statistics.stdev(times), 2) if len(times) > 1 else 0,
            "p95_ms": round(self._percentile(times, 95), 2),
            "p99_ms": round(self._percentile(times, 99), 2),
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks"""
        print("=" * 60)
        print("Intent Engine Performance Benchmark")
        print("=" * 60)
        
        # Check if server is running
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code != 200:
                print("Server is not responding correctly!")
                return {}
        except Exception as e:
            print(f"Cannot connect to server: {e}")
            print("Please start the server first with: docker-compose up")
            return {}
        
        print("\nServer is running. Starting benchmarks...")
        
        # Run benchmarks
        self.results["intent_extraction"] = self.benchmark_intent_extraction()
        self.results["ranking"] = self.benchmark_ranking()
        self.results["url_ranking"] = self.benchmark_url_ranking()
        self.results["ad_matching"] = self.benchmark_ad_matching()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        for endpoint, stats in self.results.items():
            print(f"\n{endpoint.upper().replace('_', ' ')}:")
            if "error" in stats:
                print(f"  Error: {stats['error']}")
            else:
                print(f"  Count: {stats['count']}")
                print(f"  Mean: {stats['mean_ms']:.2f} ms")
                print(f"  Median: {stats['median_ms']:.2f} ms")
                print(f"  Min/Max: {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms")
                print(f"  P95: {stats['p95_ms']:.2f} ms")
                print(f"  P99: {stats['p99_ms']:.2f} ms")


def main():
    """Main entry point"""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Save results to file
    if results:
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
