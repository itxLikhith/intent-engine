"""
Demo script to show the intent ranker functionality
"""

from core.schema import (
    Constraint,
    ConstraintType,
    DeclaredIntent,
    EthicalDimension,
    EthicalSignal,
    Frequency,
    InferredIntent,
    Recency,
    SkillLevel,
    TemporalHorizon,
    TemporalIntent,
    UniversalIntent,
    UseCase,
)
from ranking.ranker import RankingRequest, SearchResult, rank_results


def create_sample_intent():
    """Create a sample intent for demonstration"""
    return UniversalIntent(
        intentId="demo-intent-123",
        context={
            "product": "search",
            "timestamp": "2026-01-23T12:00:00Z",
            "sessionId": "demo-session",
            "userLocale": "en-US",
        },
        declared=DeclaredIntent(
            query="How to set up E2E encrypted email on Android, no big tech solutions",
            constraints=[
                Constraint(type=ConstraintType.INCLUSION, dimension="platform", value="Android", hardFilter=True),
                Constraint(
                    type=ConstraintType.EXCLUSION, dimension="provider", value=["Google", "Microsoft"], hardFilter=True
                ),
                Constraint(
                    type=ConstraintType.INCLUSION, dimension="feature", value="end-to-end_encryption", hardFilter=True
                ),
            ],
            negativePreferences=["no big tech"],
            skillLevel=SkillLevel.INTERMEDIATE,
        ),
        inferred=InferredIntent(
            useCases=[UseCase.LEARNING, UseCase.TROUBLESHOOTING],
            temporalIntent=TemporalIntent(
                horizon=TemporalHorizon.TODAY, recency=Recency.RECENT, frequency=Frequency.ONEOFF
            ),
            resultType=None,
            ethicalSignals=[
                EthicalSignal(dimension=EthicalDimension.PRIVACY, preference="privacy-first"),
                EthicalSignal(dimension=EthicalDimension.OPENNESS, preference="open-source_preferred"),
            ],
        ),
    )


def demo_ranking():
    """Demonstrate the ranking functionality"""
    print("=== Intent Engine Phase 2: Constraint Satisfaction & Ranking Demo ===\n")

    # Create sample intent
    intent = create_sample_intent()
    print(f"Input Intent Query: {intent.declared.query}\n")

    # Create sample search results
    candidates = [
        SearchResult(
            id="1",
            title="ProtonMail Android Setup Guide",
            description=(
                "Complete guide to setting up ProtonMail with end-to-end encryption "
                "on Android devices. Perfect for intermediate users."
            ),
            platform="Android",
            provider="ProtonMail",
            license="open-source",
            tags=["Android", "Email", "Encryption", "Setup", "Guide", "Privacy", "Open Source"],
            qualityScore=0.9,
            privacyRating=0.9,
            opensource=True,
            complexity="intermediate",
            recency="2026-01-20T10:00:00Z",
        ),
        SearchResult(
            id="2",
            title="Tutanota Android Configuration",
            description=(
                "Step-by-step tutorial for configuring Tutanota email client on Android with encryption features."
            ),
            platform="Android",
            provider="Tutanota",
            license="open-source",
            tags=["Android", "Email", "Encryption", "Configuration", "Tutorial"],
            qualityScore=0.85,
            privacyRating=0.95,
            opensource=True,
            complexity="intermediate",
            recency="2026-01-18T15:30:00Z",
        ),
        SearchResult(
            id="3",
            title="Gmail Setup on Android",
            description="Guide to setting up Gmail on Android devices.",
            platform="Android",
            provider="Google",  # This will be filtered out due to exclusion constraint
            license="proprietary",
            tags=["Android", "Email", "Gmail", "Setup"],
            qualityScore=0.7,
            privacyRating=0.3,
            opensource=False,
            complexity="beginner",
            recency="2026-01-15T09:00:00Z",
        ),
        SearchResult(
            id="4",
            title="Email Protocols Overview",
            description="General overview of email protocols and concepts. Not specific to Android.",
            platform="Web",  # This will be filtered out due to inclusion constraint
            provider="EducationalSite",
            license="CC BY-SA",
            tags=["Email", "Protocols", "Learning"],
            qualityScore=0.6,
            privacyRating=0.5,
            opensource=True,
            complexity="intermediate",
            recency="2026-01-10T14:00:00Z",
        ),
    ]

    print("Candidate Results:")
    for candidate in candidates:
        print(
            f"  - ID: {candidate.id}, Title: {candidate.title}, "
            f"Platform: {candidate.platform}, Provider: {candidate.provider}"
        )
    print()

    # Create ranking request
    request = RankingRequest(intent=intent, candidates=candidates)

    # Perform ranking
    response = rank_results(request)

    print("Ranked Results (after constraint filtering):")
    for i, ranked_result in enumerate(response.rankedResults, 1):
        print(f"  {i}. Title: {ranked_result.result.title}")
        print(f"     Score: {ranked_result.alignmentScore:.3f}")
        print(f"     Reasons: {', '.join(ranked_result.matchReasons)}")
        print(f"     Platform: {ranked_result.result.platform}, Provider: {ranked_result.result.provider}")
        print()

    print(
        f"Summary: {len(candidates)} candidates were processed, "
        f"{len(response.rankedResults)} passed constraints and were ranked."
    )


if __name__ == "__main__":
    demo_ranking()
