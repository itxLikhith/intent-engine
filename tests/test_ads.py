"""
Unit tests for the ad_matcher module
"""

import copy
import unittest

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


class TestAdMatcher(unittest.TestCase):
    def setUp(self):
        """Set up test data - use deepcopy to prevent fixture mutations"""
        # Create a sample intent for testing
        self.sample_intent_base = UniversalIntent(
            intentId="test-intent-123",
            context={
                "product": "search",
                "timestamp": "2026-01-23T12:00:00Z",
                "sessionId": "test-session",
                "userLocale": "en-US",
            },
            declared=DeclaredIntent(
                query="How to setup E2E encrypted email on Android, no big tech solutions",
                goal=IntentGoal.LEARN,
                constraints=[
                    Constraint(type=ConstraintType.INCLUSION, dimension="platform", value="Android", hardFilter=True),
                    Constraint(
                        type=ConstraintType.EXCLUSION,
                        dimension="provider",
                        value=["Google", "Microsoft"],
                        hardFilter=True,
                    ),
                    Constraint(
                        type=ConstraintType.INCLUSION, dimension="license", value="open_source", hardFilter=True
                    ),
                ],
                negativePreferences=["no big tech"],
                skillLevel=SkillLevel.INTERMEDIATE,
            ),
            inferred=InferredIntent(
                useCases=[UseCase.LEARNING, UseCase.TROUBLESHOOTING],
                ethicalSignals=[
                    EthicalSignal(dimension=EthicalDimension.PRIVACY, preference="privacy-first"),
                    EthicalSignal(dimension=EthicalDimension.OPENNESS, preference="open-source_preferred"),
                ],
            ),
        )
        # Deep copy for each test to prevent mutation issues
        self.sample_intent = copy.deepcopy(self.sample_intent_base)

    def test_forbidden_dimension_rejection(self):
        """Test that ads with forbidden dimensions are rejected"""
        # Create an ad with forbidden dimension
        bad_ad = AdMetadata(
            id="bad_ad_1",
            title="Email Service",
            description="Secure email service",
            targetingConstraints={"platform": ["Android"]},
            forbiddenDimensions=["age", "location"],  # Forbidden dimensions
            qualityScore=0.9,
            ethicalTags=["privacy", "open_source"],
        )

        # Create a good ad without forbidden dimensions
        good_ad = AdMetadata(
            id="good_ad_1",
            title="Open Source Email",
            description="Privacy-focused email for Android",
            targetingConstraints={"platform": ["Android"], "license": ["open_source"]},
            forbiddenDimensions=[],  # No forbidden dimensions
            qualityScore=0.85,
            ethicalTags=["privacy", "open_source"],
        )

        request = AdMatchingRequest(
            intent=self.sample_intent, adInventory=[bad_ad, good_ad], config={"topK": 5, "minThreshold": 0.4}
        )

        response = match_ads(request)

        # Only the good ad should be returned
        self.assertEqual(len(response.matchedAds), 1)
        self.assertEqual(response.matchedAds[0].ad.id, "good_ad_1")
        self.assertEqual(response.metrics["adsPassedFairness"], 1)

    def test_constraint_based_ad_filtering(self):
        """Test that ads are filtered based on user constraints"""
        # Create an ad that satisfies all constraints
        compliant_ad = AdMetadata(
            id="compliant_ad_1",
            title="ProtonMail Android",
            description="Secure email for Android devices",
            targetingConstraints={"platform": ["Android"], "provider": ["ProtonMail"], "license": ["open_source"]},
            forbiddenDimensions=[],
            qualityScore=0.9,
            ethicalTags=["privacy", "open_source"],
        )

        # Create an ad that violates platform constraint
        non_compliant_ad_platform = AdMetadata(
            id="non_compliant_ad_1",
            title="iOS Email App",
            description="Email app for iOS",
            targetingConstraints={
                "platform": ["iOS"],  # Doesn't target Android
                "provider": ["SomeProvider"],
                "license": ["open_source"],
            },
            forbiddenDimensions=[],
            qualityScore=0.7,
            ethicalTags=["privacy"],
        )

        # Create an ad that violates provider exclusion
        non_compliant_ad_provider = AdMetadata(
            id="non_compliant_ad_2",
            title="Gmail Service",
            description="Google email service",
            targetingConstraints={
                "platform": ["Android"],
                "provider": ["Google"],  # Violates exclusion constraint
                "license": ["proprietary"],
            },
            forbiddenDimensions=[],
            qualityScore=0.6,
            ethicalTags=["ad_supported"],
        )

        request = AdMatchingRequest(
            intent=self.sample_intent,
            adInventory=[compliant_ad, non_compliant_ad_platform, non_compliant_ad_provider],
            config={"topK": 5, "minThreshold": 0.4},
        )

        response = match_ads(request)

        # Only the compliant ad should pass filtering
        self.assertEqual(len(response.matchedAds), 1)
        self.assertEqual(response.matchedAds[0].ad.id, "compliant_ad_1")

    def test_ethical_signal_alignment(self):
        """Test that ethical signal alignment affects scoring"""
        # Create an ad that aligns with privacy and open-source signals
        ethical_ad = AdMetadata(
            id="ethical_ad_1",
            title="Privacy Email Client",
            description="Open source, privacy-focused email client",
            targetingConstraints={"platform": ["Android"], "license": ["open_source"]},
            forbiddenDimensions=[],
            qualityScore=0.8,
            ethicalTags=["privacy", "open_source", "no_tracking"],
        )

        # Create an ad that doesn't align with ethical signals but satisfies constraints
        non_ethical_ad = AdMetadata(
            id="non_ethical_ad_1",
            title="Basic Email Service",
            description="Standard email service",
            targetingConstraints={
                "platform": ["Android"],
                "license": ["open_source"],  # Satisfies the inclusion constraint
            },
            forbiddenDimensions=[],
            qualityScore=0.8,
            ethicalTags=["basic_service"],
        )

        request = AdMatchingRequest(
            intent=self.sample_intent, adInventory=[ethical_ad, non_ethical_ad], config={"topK": 5, "minThreshold": 0.4}
        )

        response = match_ads(request)

        # Both ads should match, but ethical ad should rank higher
        self.assertEqual(len(response.matchedAds), 2)
        self.assertGreater(response.matchedAds[0].adRelevanceScore, response.matchedAds[1].adRelevanceScore)
        self.assertEqual(response.matchedAds[0].ad.id, "ethical_ad_1")

    def test_semantic_relevance_scoring(self):
        """Test semantic relevance scoring"""
        # Create an ad highly relevant to the query
        relevant_ad = AdMetadata(
            id="relevant_ad_1",
            title="Android Email Setup Guide",
            description="Complete guide to setting up encrypted email on Android devices",
            targetingConstraints={"platform": ["Android"]},
            forbiddenDimensions=[],
            qualityScore=0.7,
            ethicalTags=["privacy"],
        )

        # Create an ad less relevant to the query
        less_relevant_ad = AdMetadata(
            id="less_relevant_ad_1",
            title="Calendar App",
            description="Schedule meetings and manage time",
            targetingConstraints={"platform": ["Android"]},
            forbiddenDimensions=[],
            qualityScore=0.7,
            ethicalTags=["organization"],
        )

        request = AdMatchingRequest(
            intent=self.sample_intent,
            adInventory=[relevant_ad, less_relevant_ad],
            config={"topK": 5, "minThreshold": 0.4},
        )

        response = match_ads(request)

        # Both ads satisfy constraints, but relevant ad should score higher
        if len(response.matchedAds) >= 2:
            self.assertGreater(response.matchedAds[0].adRelevanceScore, response.matchedAds[1].adRelevanceScore)
            self.assertEqual(response.matchedAds[0].ad.id, "relevant_ad_1")

    def test_edge_case_no_valid_ads(self):
        """Test behavior when no ads pass all checks"""
        # Create ads that all fail for various reasons
        bad_ads = [
            AdMetadata(
                id="bad_ad_1",
                title="Bad Ad",
                description="Has forbidden targeting",
                targetingConstraints={"platform": ["Android"]},
                forbiddenDimensions=["age"],  # Forbidden
                qualityScore=0.9,
                ethicalTags=["privacy"],
            ),
            AdMetadata(
                id="bad_ad_2",
                title="Wrong Platform",
                description="Targets wrong platform",
                targetingConstraints={"platform": ["iOS"]},  # Doesn't match Android requirement
                forbiddenDimensions=[],
                qualityScore=0.8,
                ethicalTags=["privacy"],
            ),
        ]

        request = AdMatchingRequest(
            intent=self.sample_intent, adInventory=bad_ads, config={"topK": 5, "minThreshold": 0.4}
        )

        response = match_ads(request)

        # No ads should be returned
        self.assertEqual(len(response.matchedAds), 0)
        self.assertEqual(response.metrics["adsPassedFairness"], 1)  # Only one ad passes fairness check

    def test_edge_case_all_ads_rejected(self):
        """Test behavior when all ads are rejected by constraints"""
        # Create ads that pass fairness but fail constraints
        constraint_failing_ads = [
            AdMetadata(
                id="failing_ad_1",
                title="Google Service",
                description="Email service from Google",
                targetingConstraints={"platform": ["Android"], "provider": ["Google"]},  # Violates exclusion
                forbiddenDimensions=[],  # Passes fairness
                qualityScore=0.9,
                ethicalTags=["privacy"],
            ),
            AdMetadata(
                id="failing_ad_2",
                title="Proprietary Solution",
                description="Closed source email",
                targetingConstraints={"platform": ["Android"], "license": ["proprietary"]},  # Violates inclusion
                forbiddenDimensions=[],  # Passes fairness
                qualityScore=0.8,
                ethicalTags=["basic"],
            ),
        ]

        request = AdMatchingRequest(
            intent=self.sample_intent, adInventory=constraint_failing_ads, config={"topK": 5, "minThreshold": 0.4}
        )

        response = match_ads(request)

        # No ads should pass constraints
        self.assertEqual(len(response.matchedAds), 0)
        self.assertEqual(response.metrics["adsPassedFairness"], 2)  # Both pass fairness
        self.assertEqual(response.metrics["adsFiltered"], 2)  # Both filtered by constraints

    def test_match_reasons_generation(self):
        """Test that match reasons are properly generated"""
        compliant_ad = AdMetadata(
            id="test_ad_1",
            title="Privacy Email",
            description="Open source encrypted email for Android",
            targetingConstraints={"platform": ["Android"], "license": ["open_source"]},
            forbiddenDimensions=[],
            qualityScore=0.85,
            ethicalTags=["privacy", "open_source"],
        )

        request = AdMatchingRequest(
            intent=self.sample_intent, adInventory=[compliant_ad], config={"topK": 5, "minThreshold": 0.4}
        )

        response = match_ads(request)

        # Should have one matched ad with reasons
        self.assertEqual(len(response.matchedAds), 1)
        matched_ad = response.matchedAds[0]

        # Check that reasons are provided
        self.assertGreater(len(matched_ad.matchReasons), 0)

        # Check for expected types of reasons
        has_constraint_reason = any("platform=" in reason for reason in matched_ad.matchReasons)
        has_ethical_reason = any("ethical alignment" in reason for reason in matched_ad.matchReasons)
        has_query_match = any(
            "query semantic match" in reason or "keyword match" in reason for reason in matched_ad.matchReasons
        )

        # At least one type of reason should be present
        self.assertTrue(
            has_constraint_reason or has_ethical_reason or has_query_match,
            f"Expected at least one type of reason, got: {matched_ad.matchReasons}",
        )


if __name__ == "__main__":
    unittest.main()
