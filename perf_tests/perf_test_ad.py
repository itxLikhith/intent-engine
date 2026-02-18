"""
Performance test for the ad matcher module
"""

import time

from ads.matcher import AdMatchingRequest, AdMetadata, match_ads
from core.schema import (
    Constraint,
    ConstraintType,
    DeclaredIntent,
    EthicalDimension,
    EthicalSignal,
    InferredIntent,
    IntentGoal,
    SkillLevel,
    UniversalIntent,
    UseCase,
)


def create_test_intent():
    """Create a test intent for performance testing"""
    return UniversalIntent(
        intentId="perf-test-123",
        context={
            "product": "search",
            "timestamp": "2026-01-23T12:00:00Z",
            "sessionId": "perf-session",
            "userLocale": "en-US",
        },
        declared=DeclaredIntent(
            query="How to setup E2E encrypted email on Android, no big tech solutions",
            goal=IntentGoal.LEARN,
            constraints=[
                Constraint(type=ConstraintType.INCLUSION, dimension="platform", value="Android", hardFilter=True),
                Constraint(
                    type=ConstraintType.EXCLUSION, dimension="provider", value=["Google", "Microsoft"], hardFilter=True
                ),
                Constraint(type=ConstraintType.INCLUSION, dimension="license", value="open_source", hardFilter=True),
            ],
            negativePreferences=["no big tech"],
            skillLevel=SkillLevel.INTERMEDIATE,
        ),
        inferred=InferredIntent(
            useCases=[UseCase.LEARNING, UseCase.TROUBLESHOOTING],
            ethicalSignals=[EthicalSignal(dimension=EthicalDimension.PRIVACY, preference="privacy-first")],
        ),
    )


def create_test_ads(count=20):
    """Create test ads for performance testing"""
    ads = []
    for i in range(count):
        ads.append(
            AdMetadata(
                id=f"ad_{i + 1}",
                title=f"Email Service {i + 1}",
                description=(
                    "Secure, open-source email for Android devices" if i % 2 == 0 else "General service description"
                ),
                targetingConstraints=(
                    {
                        "platform": ["Android"] if i % 3 != 2 else ["iOS"],
                        "license": ["open_source"] if i % 4 != 3 else ["proprietary"],
                        "provider": ["ProtonMail", "Tutanota"] if i % 5 != 4 else ["Google", "Microsoft"],
                    }
                    if i % 6 != 5
                    else {"platform": ["Android"]}
                ),
                forbiddenDimensions=[] if i % 7 != 0 else ["age", "location"],  # Some ads have forbidden dims
                qualityScore=0.7 + (i % 4) * 0.1,
                ethicalTags=["privacy", "open_source"] if i % 2 == 0 else ["basic_service"],
                advertiser=f"Advertiser_{i + 1}",
            )
        )
    return ads


def test_performance():
    """Test the performance of the ad matcher"""
    intent = create_test_intent()
    ads = create_test_ads(20)  # 20 ads as specified in requirements

    # Warm up the model
    request = AdMatchingRequest(intent=intent, adInventory=ads[:1], config={"topK": 5, "minThreshold": 0.4})
    match_ads(request)

    # Measure performance
    times = []
    for _ in range(10):  # Run 10 iterations
        start_time = time.time()
        request = AdMatchingRequest(intent=intent, adInventory=ads, config={"topK": 5, "minThreshold": 0.4})
        response = match_ads(request)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        print(f"Iteration - Time: {elapsed_ms:.2f}ms, Matched Ads: {len(response.matchedAds)}")

    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    print("\nPerformance Results:")
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
