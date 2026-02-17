"""
Demo script to show the ad matcher functionality
"""
from ads.matcher import (
    match_ads,
    AdMatchingRequest,
    AdMetadata
)
from core.schema import (
    UniversalIntent,
    DeclaredIntent,
    InferredIntent,
    Constraint,
    ConstraintType,
    UseCase,
    EthicalSignal,
    EthicalDimension,
    IntentGoal,
    SkillLevel
)


def create_demo_intent():
    """Create a demo intent for demonstration"""
    return UniversalIntent(
        intentId="demo-intent-123",
        context={
            "product": "search",
            "timestamp": "2026-01-23T12:00:00Z",
            "sessionId": "demo-session",
            "userLocale": "en-US"
        },
        declared=DeclaredIntent(
            query="How to setup E2E encrypted email on Android, no big tech solutions",
            goal=IntentGoal.LEARN,
            constraints=[
                Constraint(type=ConstraintType.INCLUSION, dimension="platform", value="Android", hardFilter=True),
                Constraint(type=ConstraintType.EXCLUSION, dimension="provider", value=["Google", "Microsoft"], hardFilter=True),
                Constraint(type=ConstraintType.INCLUSION, dimension="license", value="open_source", hardFilter=True)
            ],
            negativePreferences=["no big tech"],
            skillLevel=SkillLevel.INTERMEDIATE
        ),
        inferred=InferredIntent(
            useCases=[UseCase.LEARNING, UseCase.TROUBLESHOOTING],
            ethicalSignals=[
                EthicalSignal(dimension=EthicalDimension.PRIVACY, preference="privacy-first"),
                EthicalSignal(dimension=EthicalDimension.OPENNESS, preference="open-source_preferred")
            ]
        )
    )


def demo_ad_matching():
    """Demonstrate the ad matching functionality"""
    print("=== Intent Engine Phase 4: Privacy-First Ad Matching Demo ===\n")
    
    # Create demo intent
    intent = create_demo_intent()
    print(f"User Query: {intent.declared.query}\n")
    print(f"Declared Constraints:")
    for constraint in intent.declared.constraints:
        print(f"  - {constraint.type.value} {constraint.dimension}: {constraint.value} (hard filter: {constraint.hardFilter})")
    print(f"\nEthical Signals: {[(es.dimension.value, es.preference) for es in intent.inferred.ethicalSignals]}\n")
    
    # Define ad inventory
    ads = [
        AdMetadata(
            id="ad_1",
            title="ProtonMail Android App",
            description="Secure, open-source email with end-to-end encryption for Android devices",
            targetingConstraints={
                "platform": ["Android"],
                "provider": ["ProtonMail"],
                "license": ["open_source"],
                "feature": ["encryption"]
            },
            forbiddenDimensions=[],  # No forbidden dimensions
            qualityScore=0.92,
            ethicalTags=["privacy", "open_source", "no_tracking"],
            advertiser="Proton Technologies"
        ),
        AdMetadata(
            id="ad_2",
            title="Tutanota Email Client",
            description="Privacy-focused encrypted email for Android with local-first approach",
            targetingConstraints={
                "platform": ["Android", "iOS"],
                "license": ["open_source"],
                "feature": ["encryption", "privacy"]
            },
            forbiddenDimensions=[],  # No forbidden dimensions
            qualityScore=0.88,
            ethicalTags=["privacy", "open_source", "no_ads"],
            advertiser="Tutao GmbH"
        ),
        AdMetadata(
            id="ad_3",
            title="Gmail Setup Guide",
            description="Learn to configure Gmail on your Android device",
            targetingConstraints={
                "platform": ["Android"],
                "provider": ["Google"]  # Would violate exclusion constraint
            },
            forbiddenDimensions=[],  # No forbidden dimensions
            qualityScore=0.85,
            ethicalTags=["convenience"],
            advertiser="Google Ads"
        ),
        AdMetadata(
            id="ad_4",
            title="Secure Messaging App",
            description="Encrypted communication platform for teams",
            targetingConstraints={
                "platform": ["Android"],
                "license": ["open_source"]
            },
            forbiddenDimensions=["age", "location"],  # FORBIDDEN: Contains forbidden dimensions
            qualityScore=0.90,
            ethicalTags=["privacy", "security"],
            advertiser="SecureCom Inc."
        ),
        AdMetadata(
            id="ad_5",
            title="Open Source Calendar",
            description="Privacy-respecting calendar app for Android",
            targetingConstraints={
                "platform": ["Android"],
                "license": ["open_source"]
            },
            forbiddenDimensions=[],  # No forbidden dimensions
            qualityScore=0.75,
            ethicalTags=["privacy", "open_source"],
            advertiser="OpenSource Devs"
        )
    ]
    
    print("Available Ad Inventory:")
    for ad in ads:
        print(f"  - {ad.title} (ID: {ad.id})")
        print(f"    Targeting: {ad.targetingConstraints}")
        print(f"    Forbidden: {ad.forbiddenDimensions}")
        print(f"    Ethical Tags: {ad.ethicalTags}")
    print()
    
    # Create ad matching request
    request = AdMatchingRequest(
        intent=intent,
        adInventory=ads,
        config={"topK": 5, "minThreshold": 0.4}
    )
    
    # Perform ad matching
    response = match_ads(request)
    
    print("Matched Ads (Top 5):")
    for i, matched_ad in enumerate(response.matchedAds, 1):
        print(f"  {i}. {matched_ad.ad.title}")
        print(f"     ID: {matched_ad.ad.id}")
        print(f"     Relevance Score: {matched_ad.adRelevanceScore:.3f}")
        print(f"     Reasons: {', '.join(matched_ad.matchReasons)}")
        print(f"     Advertiser: {matched_ad.ad.advertiser}")
        print()
    
    print("Metrics:")
    print(f"  - Total ads evaluated: {response.metrics['totalAdsEvaluated']}")
    print(f"  - Ads passed fairness: {response.metrics['adsPassedFairness']}")
    print(f"  - Ads filtered out: {response.metrics['adsFiltered']}")
    print(f"  - Ads shown to user: {len(response.matchedAds)}")


if __name__ == "__main__":
    demo_ad_matching()