"""
Demo script to show the service recommender functionality
"""

from core.schema import (
    DeclaredIntent,
    EthicalDimension,
    EthicalSignal,
    Frequency,
    InferredIntent,
    IntentGoal,
    Recency,
    SkillLevel,
    TemporalHorizon,
    TemporalIntent,
    UniversalIntent,
    UseCase,
)
from services.recommender import ServiceMetadata, ServiceRecommendationRequest, recommend_services


def create_demo_intent():
    """Create a demo intent for demonstration"""
    return UniversalIntent(
        intentId="demo-intent-123",
        context={
            "product": "workspace",
            "timestamp": "2026-01-23T12:00:00Z",
            "sessionId": "demo-session",
            "userLocale": "en-US",
        },
        declared=DeclaredIntent(
            query="How to collaborate on a research document with my team?",
            goal=IntentGoal.COLLABORATE,
            skillLevel=SkillLevel.INTERMEDIATE,
        ),
        inferred=InferredIntent(
            useCases=[UseCase.PROFESSIONAL_DEVELOPMENT, UseCase.LEARNING],
            temporalIntent=TemporalIntent(
                horizon=TemporalHorizon.WEEK, recency=Recency.EVERGREEN, frequency=Frequency.RECURRING
            ),
            ethicalSignals=[
                EthicalSignal(dimension=EthicalDimension.OPENNESS, preference="open_format"),
                EthicalSignal(dimension=EthicalDimension.PRIVACY, preference="privacy_first"),
            ],
        ),
    )


def demo_service_recommendation():
    """Demonstrate the service recommendation functionality"""
    print("=== Intent Engine Phase 3: Service Recommendation Demo ===\n")

    # Create demo intent
    intent = create_demo_intent()
    print(f"Input Intent Query: {intent.declared.query}\n")
    print(f"Declared Goal: {intent.declared.goal.value}")
    print(f"Use Cases: {[uc.value for uc in intent.inferred.useCases]}")
    print(f"Ethical Signals: {[(es.dimension.value, es.preference) for es in intent.inferred.ethicalSignals]}\n")

    # Define available services
    services = [
        ServiceMetadata(
            id="docs",
            name="Documents",
            supportedGoals=["CREATE", "COLLABORATE", "EDIT", "DRAFT_DOCUMENT"],
            primaryUseCases=["writing", "research", "drafting", "collaboration", "teamwork"],
            temporalPatterns=["long_session", "recurring_edit", "extended_work"],
            ethicalAlignment=["open_format", "local_first", "privacy_first"],
            description="Collaborative document editor with real-time editing",
        ),
        ServiceMetadata(
            id="mail",
            name="Email",
            supportedGoals=["COMMUNICATE", "ORGANIZE"],
            primaryUseCases=["communication", "organization", "sharing"],
            temporalPatterns=["short_session", "frequent_access", "quick_message"],
            ethicalAlignment=["encrypted", "privacy_first", "no_tracking"],
            description="Secure email service with end-to-end encryption",
        ),
        ServiceMetadata(
            id="calendar",
            name="Calendar",
            supportedGoals=["SCHEDULE", "ORGANIZE"],
            primaryUseCases=["scheduling", "planning", "coordination"],
            temporalPatterns=["quick_check", "regular_updates", "event_planning"],
            ethicalAlignment=["open_source", "no_ads", "privacy_first"],
            description="Privacy-focused calendar with scheduling tools",
        ),
        ServiceMetadata(
            id="search",
            name="Search",
            supportedGoals=["FIND_INFORMATION", "LEARN", "RESEARCH"],
            primaryUseCases=["searching", "discovery", "research", "information_gathering"],
            temporalPatterns=["quick_lookup", "one_time_task", "information_retrieval"],
            ethicalAlignment=["no_ads", "privacy_first", "ethical_ranking"],
            description="Private search engine without tracking",
        ),
        ServiceMetadata(
            id="notes",
            name="Notes",
            supportedGoals=["CREATE", "REFLECT", "ORGANIZE"],
            primaryUseCases=["note_taking", "brainstorming", "personal_organization"],
            temporalPatterns=["quick_capture", "frequent_access", "personal_use"],
            ethicalAlignment=["local_first", "encrypted", "privacy_first"],
            description="Fast note-taking app with sync capabilities",
        ),
    ]

    print("Available Services:")
    for service in services:
        print(f"  - {service.name} ({service.id}): Supports {service.supportedGoals[:3]}...")
    print()

    # Create recommendation request
    request = ServiceRecommendationRequest(intent=intent, availableServices=services)

    # Perform recommendation
    response = recommend_services(request)

    print("Recommended Services (ranked by relevance):")
    for i, recommendation in enumerate(response.recommendations, 1):
        print(f"  {i}. {recommendation.service.name} ({recommendation.service.id})")
        print(f"     Score: {recommendation.serviceScore:.3f}")
        print(f"     Reasons: {', '.join(recommendation.matchReasons)}")
        print()

    print(f"Summary: {len(services)} services were evaluated, {len(response.recommendations)} were recommended.")


if __name__ == "__main__":
    demo_service_recommendation()
