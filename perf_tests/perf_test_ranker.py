"""
Performance test for the intent ranker module
"""
import time
from ranking.ranker import (
    rank_results,
    RankingRequest,
    SearchResult
)
from core.schema import (
    UniversalIntent,
    DeclaredIntent,
    InferredIntent,
    TemporalIntent,
    Constraint,
    ConstraintType,
    UseCase,
    EthicalSignal,
    EthicalDimension,
    TemporalHorizon,
    Recency,
    Frequency,
    SkillLevel
)


def create_test_intent():
    """Create a test intent for performance testing"""
    return UniversalIntent(
        intentId="perf-test-123",
        context={
            "product": "search",
            "timestamp": "2026-01-23T12:00:00Z",
            "sessionId": "perf-session",
            "userLocale": "en-US"
        },
        declared=DeclaredIntent(
            query="How to set up E2E encrypted email on Android, no big tech solutions",
            constraints=[
                Constraint(type=ConstraintType.INCLUSION, dimension="platform", value="Android", hardFilter=True),
                Constraint(type=ConstraintType.EXCLUSION, dimension="provider", value=["Google", "Microsoft"], hardFilter=True),
                Constraint(type=ConstraintType.INCLUSION, dimension="feature", value="end-to-end_encryption", hardFilter=True)
            ],
            negativePreferences=["no big tech"],
            skillLevel=SkillLevel.INTERMEDIATE
        ),
        inferred=InferredIntent(
            useCases=[UseCase.LEARNING, UseCase.TROUBLESHOOTING],
            temporalIntent=TemporalIntent(
                horizon=TemporalHorizon.TODAY,
                recency=Recency.RECENT,
                frequency=Frequency.ONEOFF
            ),
            resultType=None,
            ethicalSignals=[
                EthicalSignal(dimension=EthicalDimension.PRIVACY, preference="privacy-first"),
                EthicalSignal(dimension=EthicalDimension.OPENNESS, preference="open-source_preferred")
            ]
        )
    )


def create_test_candidates(count=10):
    """Create test candidates for performance testing"""
    candidates = []
    for i in range(count):
        candidates.append(SearchResult(
            id=str(i+1),
            title=f"Email Setup Guide {i+1}",
            description="Guide to setting up encrypted email on Android devices",
            platform="Android",
            provider="ProtonMail" if i % 2 == 0 else "Tutanota",  # Alternate providers
            license="open-source",
            tags=["Android", "Email", "Encryption", "Setup", "Guide", "Privacy"],
            qualityScore=0.8,
            privacyRating=0.9,
            opensource=True,
            complexity="intermediate",
            recency="2026-01-20T10:00:00Z"
        ))
    return candidates


def test_performance():
    """Test the performance of the intent ranker"""
    intent = create_test_intent()
    candidates = create_test_candidates(10)  # 10 candidates as specified
    
    # Warm up the model
    request = RankingRequest(intent=intent, candidates=candidates[:1])
    rank_results(request)
    
    # Measure performance
    times = []
    for _ in range(10):  # Run 10 iterations
        start_time = time.time()
        request = RankingRequest(intent=intent, candidates=candidates)
        response = rank_results(request)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        print(f"Iteration - Time: {elapsed_ms:.2f}ms, Results: {len(response.rankedResults)}")
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    print(f"\nPerformance Results:")
    print(f"Average time: {avg_time:.2f}ms")
    print(f"Max time: {max_time:.2f}ms")
    print(f"Min time: {min_time:.2f}ms")
    print(f"95th percentile: {sorted(times)[int(0.95 * len(times))]:.2f}ms")
    
    # Check if we meet the <100ms requirement
    if max_time < 100:
        print(f"\n✅ Performance requirement met: Max time {max_time:.2f}ms < 100ms")
    else:
        print(f"\n❌ Performance issue: Max time {max_time:.2f}ms exceeds 100ms limit")
    
    if avg_time < 100:
        print(f"✅ Average performance good: {avg_time:.2f}ms < 100ms")
    else:
        print(f"❌ Average performance issue: {avg_time:.2f}ms >= 100ms")


if __name__ == "__main__":
    test_performance()