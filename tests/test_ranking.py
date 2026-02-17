"""
Unit tests for the intent_ranker module
"""
import unittest
import copy
from ranking.ranker import (
    rank_results,
    RankingRequest,
    SearchResult,
    RankedResult,
)
from core.schema import (
    Constraint,
    ConstraintType,
    UseCase,
    EthicalDimension,
    TemporalHorizon,
    Recency,
    Frequency,
    SkillLevel,
    UniversalIntent,
    DeclaredIntent,
    InferredIntent,
    TemporalIntent,
    EthicalSignal
)


class TestIntentRanker(unittest.TestCase):

    def setUp(self):
        """Set up test data - use deepcopy to prevent fixture mutations"""
        # Create a sample intent for testing - deep copy for each test
        self.sample_intent_base = UniversalIntent(
            intentId="test-intent-123",
            context={
                "product": "search",
                "timestamp": "2026-01-23T12:00:00Z",
                "sessionId": "test-session",
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
        # Deep copy for each test to prevent mutation issues
        self.sample_intent = copy.deepcopy(self.sample_intent_base)
    
    def test_constraint_satisfaction_inclusion(self):
        """Test constraint satisfaction with inclusion constraints"""
        # Create a candidate that should pass inclusion constraint
        passing_candidate = SearchResult(
            id="1",
            title="Android Email Setup Guide",
            description="Learn how to set up email on Android devices",
            platform="Android",
            provider="ProtonMail",
            license="open-source",
            tags=["Android", "Email", "Setup", "Guide"]
        )
        
        # Create a candidate that should fail inclusion constraint
        failing_candidate = SearchResult(
            id="2",
            title="iOS Email Setup Guide",
            description="Learn how to set up email on iOS devices",
            platform="iOS",  # Wrong platform
            provider="ProtonMail",
            license="open-source",
            tags=["iOS", "Email", "Setup", "Guide"]
        )
        
        request = RankingRequest(
            intent=self.sample_intent,
            candidates=[passing_candidate, failing_candidate]
        )
        
        response = rank_results(request)
        
        # Only the passing candidate should remain
        self.assertEqual(len(response.rankedResults), 1)
        self.assertEqual(response.rankedResults[0].result.id, "1")
    
    def test_constraint_satisfaction_exclusion(self):
        """Test constraint satisfaction with exclusion constraints"""
        # Create a candidate that should pass exclusion constraint
        passing_candidate = SearchResult(
            id="1",
            title="ProtonMail Android Setup",
            description="Secure email setup on Android",
            platform="Android",
            provider="ProtonMail",  # Not Google or Microsoft
            license="open-source",
            tags=["Android", "Email", "Security", "Privacy"]
        )
        
        # Create a candidate that should fail exclusion constraint
        failing_candidate = SearchResult(
            id="2",
            title="Gmail Android Setup",
            description="Google email setup on Android",
            platform="Android",
            provider="Google",  # Excluded provider
            license="proprietary",
            tags=["Android", "Email", "Google"]
        )
        
        request = RankingRequest(
            intent=self.sample_intent,
            candidates=[passing_candidate, failing_candidate]
        )
        
        response = rank_results(request)
        
        # Only the passing candidate should remain
        self.assertEqual(len(response.rankedResults), 1)
        self.assertEqual(response.rankedResults[0].result.id, "1")
    
    def test_alignment_scoring(self):
        """Test that alignment scoring works correctly"""
        # Create a highly relevant candidate
        relevant_candidate = SearchResult(
            id="1",
            title="Android End-to-End Encrypted Email Setup Guide",
            description="Complete guide to setting up E2E encrypted email on Android devices",
            platform="Android",
            provider="ProtonMail",
            license="open-source",
            tags=["Android", "Email", "Encryption", "Setup", "Guide", "Privacy"],
            qualityScore=0.9,
            privacyRating=0.9,
            opensource=True,
            complexity="intermediate"
        )
        
        # Create a less relevant candidate
        less_relevant_candidate = SearchResult(
            id="2",
            title="General Email Overview",
            description="Overview of email protocols and concepts",
            platform="Web",
            provider="Generic",
            license="unknown",
            tags=["Email", "Protocols"],
            qualityScore=0.6,
            privacyRating=0.5,
            opensource=False,
            complexity="advanced"
        )
        
        request = RankingRequest(
            intent=self.sample_intent,
            candidates=[relevant_candidate, less_relevant_candidate]
        )

        response = rank_results(request)

        # Only the Android candidate should pass constraints (Web platform doesn't match inclusion constraint)
        self.assertEqual(len(response.rankedResults), 1)
        self.assertEqual(response.rankedResults[0].result.id, "1")

        # The relevant candidate should have a high score
        self.assertGreater(response.rankedResults[0].alignmentScore, 0.5)
    
    def test_empty_candidates(self):
        """Test behavior with empty candidate list"""
        request = RankingRequest(
            intent=self.sample_intent,
            candidates=[]
        )
        
        response = rank_results(request)
        
        # Should return empty list
        self.assertEqual(len(response.rankedResults), 0)
    
    def test_all_candidates_filtered_out(self):
        """Test behavior when all candidates fail constraints"""
        # Create candidates that all violate constraints
        violating_candidates = [
            SearchResult(
                id="1",
                title="Gmail Setup",
                description="Google email setup",
                platform="Android",
                provider="Google",  # Violates exclusion constraint
                license="proprietary"
            ),
            SearchResult(
                id="2",
                title="Outlook Setup",
                description="Microsoft email setup",
                platform="Android",
                provider="Microsoft",  # Violates exclusion constraint
                license="proprietary"
            )
        ]
        
        request = RankingRequest(
            intent=self.sample_intent,
            candidates=violating_candidates
        )
        
        response = rank_results(request)
        
        # All candidates should be filtered out
        self.assertEqual(len(response.rankedResults), 0)
    
    def test_edge_case_partial_match(self):
        """Test edge case with partial matches"""
        # Create a candidate that partially matches
        partial_match_candidate = SearchResult(
            id="1",
            title="Email Setup Guide",
            description="Guide to setting up email",
            platform="Android",
            provider="SmallerCompany",  # Not excluded
            license="open-source",
            tags=["Email", "Setup", "Guide"],
            qualityScore=0.7,
            privacyRating=0.6,
            opensource=True,
            complexity="intermediate"
        )
        
        request = RankingRequest(
            intent=self.sample_intent,
            candidates=[partial_match_candidate]
        )
        
        response = rank_results(request)
        
        # Should have one result that passes constraints
        self.assertEqual(len(response.rankedResults), 1)
        
        # Score should be reasonable but not perfect
        self.assertGreater(response.rankedResults[0].alignmentScore, 0.0)
        self.assertLessEqual(response.rankedResults[0].alignmentScore, 1.0)
    
    def test_reasons_generation(self):
        """Test that match reasons are properly generated"""
        relevant_candidate = SearchResult(
            id="1",
            title="Android End-to-End Encrypted Email Setup Guide",
            description="Complete guide to setting up E2E encrypted email on Android devices",
            platform="Android",
            provider="ProtonMail",
            license="open-source",
            tags=["Android", "Email", "Encryption", "Setup", "Guide", "Privacy"],
            qualityScore=0.9,
            privacyRating=0.9,
            opensource=True,
            complexity="intermediate"
        )
        
        request = RankingRequest(
            intent=self.sample_intent,
            candidates=[relevant_candidate]
        )
        
        response = rank_results(request)
        
        # Should have reasons for the match
        self.assertGreater(len(response.rankedResults[0].matchReasons), 0)
        # Check that some expected reasons are present
        reasons = response.rankedResults[0].matchReasons
        # At least one of these should be present
        expected_reasons_present = any(
            reason in ['query-content-match', 'use-case-learning-match', 'use-case-troubleshooting-match', 
                      'privacy-aligned', 'open-source-aligned', 'skill-level-match-intermediate']
            for reason in reasons
        )
        self.assertTrue(expected_reasons_present, f"Expected at least one of the known reasons, got: {reasons}")


if __name__ == '__main__':
    unittest.main()