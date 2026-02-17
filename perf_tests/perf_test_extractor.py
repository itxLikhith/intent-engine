"""
Performance test for the intent extractor module
"""
import time
from extraction.extractor import extract_intent


def test_performance():
    """Test the performance of the intent extractor"""
    queries = [
        "How to set up E2E encrypted email on Android, no big tech",
        "Compare ProtonMail and Tutanota for privacy",
        "Fix email sync issues on iPhone",
        "Best open source calendar app for Linux",
        "Setup secure messaging app without Google services"
    ]
    
    # Warm up the model
    for _ in range(3):
        extract_intent("How to set up email?")
    
    # Measure performance
    times = []
    for query in queries * 5:  # Repeat 5 times for each query
        start_time = time.time()
        intent = extract_intent(query)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        print(f"Query: '{query[:30]}...' | Time: {elapsed_ms:.2f}ms | Goal: {intent.declared.goal}")
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    print(f"\nPerformance Results:")
    print(f"Average time: {avg_time:.2f}ms")
    print(f"Max time: {max_time:.2f}ms")
    print(f"Min time: {min_time:.2f}ms")
    print(f"95th percentile: {sorted(times)[int(0.95 * len(times))]:.2f}ms")
    
    # Check if we meet the <50ms requirement
    if max_time < 50:
        print("\n✅ Performance requirement met: All queries under 50ms")
    else:
        print(f"\n❌ Performance issue: Max time {max_time:.2f}ms exceeds 50ms limit")
    
    if avg_time < 50:
        print(f"✅ Average performance good: {avg_time:.2f}ms < 50ms")
    else:
        print(f"❌ Average performance issue: {avg_time:.2f}ms >= 50ms")


if __name__ == "__main__":
    test_performance()