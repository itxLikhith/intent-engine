"""
Performance test for the service recommender module
"""
import time
from services.recommender import (
    recommend_services,
    ServiceRecommendationRequest,
    ServiceMetadata
)
from core.schema import (
    UniversalIntent,
    DeclaredIntent,
    InferredIntent,
    TemporalIntent,
    IntentGoal,
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
            "product": "workspace",
            "timestamp": "2026-01-23T12:00:00Z",
            "sessionId": "perf-session",
            "userLocale": "en-US"
        },
        declared=DeclaredIntent(
            query="How to collaborate on a research document with my team?",
            goal=IntentGoal.COLLABORATE,
            skillLevel=SkillLevel.INTERMEDIATE
        ),
        inferred=InferredIntent(
            useCases=[UseCase.PROFESSIONAL_DEVELOPMENT, UseCase.LEARNING],
            temporalIntent=TemporalIntent(
                horizon=TemporalHorizon.WEEK,
                recency=Recency.EVERGREEN,
                frequency=Frequency.RECURRING
            ),
            ethicalSignals=[
                EthicalSignal(dimension=EthicalDimension.OPENNESS, preference="open_format")
            ]
        )
    )


def create_test_services(count=10):
    """Create test services for performance testing"""
    services = []
    for i in range(count):
        services.append(ServiceMetadata(
            id=f"service_{i+1}",
            name=f"Service {i+1}",
            supportedGoals=["CREATE", "COLLABORATE", "EDIT"] if i % 2 == 0 else ["FIND_INFORMATION", "LEARN"],
            primaryUseCases=["writing", "research", "drafting"] if i % 3 == 0 else ["searching", "discovery", "learning"],
            temporalPatterns=["long_session", "recurring_edit"] if i % 4 == 0 else ["quick_lookup", "one_time_task"],
            ethicalAlignment=["open_format", "local_first"] if i % 5 == 0 else ["encrypted", "privacy_first"],
            description=f"Description for service {i+1}"
        ))
    return services


def test_performance():
    """Test the performance of the service recommender"""
    intent = create_test_intent()
    services = create_test_services(10)  # 10 services as specified
    
    # Warm up the model
    request = ServiceRecommendationRequest(intent=intent, availableServices=services[:1])
    recommend_services(request)
    
    # Measure performance
    times = []
    for _ in range(10):  # Run 10 iterations
        start_time = time.time()
        request = ServiceRecommendationRequest(intent=intent, availableServices=services)
        response = recommend_services(request)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        print(f"Iteration - Time: {elapsed_ms:.2f}ms, Recommendations: {len(response.recommendations)}")
    
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
        print(f"\n✅ Performance requirement met: Max time {max_time:.2f}ms < 50ms")
    else:
        print(f"\n❌ Performance issue: Max time {max_time:.2f}ms exceeds 50ms limit")
    
    if avg_time < 50:
        print(f"✅ Average performance good: {avg_time:.2f}ms < 50ms")
    else:
        print(f"❌ Average performance issue: {avg_time:.2f}ms >= 50ms")


if __name__ == "__main__":
    test_performance()