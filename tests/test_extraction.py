"""
Unit tests for the intent_extractor module
"""
import unittest
import copy
from extraction.extractor import (
    extract_intent,
    IntentExtractionRequest,
)
from core.schema import (
    ConstraintType,
    IntentGoal,
    UseCase,
    EthicalDimension,
    ResultType,
    TemporalHorizon,
    Recency,
    Frequency,
    SkillLevel
)


class TestIntentExtractor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with deepcopy to prevent mutations"""
        self.base_request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up E2E encrypted email on Android, no big tech solutions'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        self.request = copy.deepcopy(self.base_request)
    
    def test_constraint_extraction(self):
        """Test constraint extraction functionality"""
        # Test inclusion constraint extraction
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up email on Android?'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent

        android_constraints = [c for c in intent.declared.constraints
                              if c.dimension == 'platform' and c.value == 'Android']
        self.assertTrue(len(android_constraints) > 0, "Should extract Android platform constraint")

        # Test exclusion constraint extraction
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'no Google or Microsoft email solutions'},
            context={'sessionId': 'test_session_2', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent

        exclusion_constraints = [c for c in intent.declared.constraints
                                if c.type == ConstraintType.EXCLUSION and c.dimension == 'provider']
        self.assertTrue(len(exclusion_constraints) > 0, "Should extract exclusion constraints")

        # Test exclusion constraint extraction
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up email, no Google solutions'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent

        exclusion_constraints = [c for c in intent.declared.constraints
                                if c.type == ConstraintType.EXCLUSION and c.value == 'Google']
        self.assertTrue(len(exclusion_constraints) > 0, "Should extract Google exclusion constraint")
        
        # Test multiple exclusions
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up email, no big tech solutions'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        exclusion_constraints = [c for c in intent.declared.constraints 
                                if c.type == ConstraintType.EXCLUSION and c.dimension == 'provider']
        self.assertTrue(len(exclusion_constraints) > 0, "Should extract big tech exclusion constraint")
        
        # Test feature constraint extraction
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up end-to-end encrypted email?'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        feature_constraints = [c for c in intent.declared.constraints 
                              if c.dimension == 'feature' and c.value == 'end-to-end_encryption']
        self.assertTrue(len(feature_constraints) > 0, "Should extract end-to-end encryption feature constraint")
    
    def test_goal_classification(self):
        """Test goal classification functionality"""
        # Test LEARN goal
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up encrypted email on Android?'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        self.assertEqual(intent.declared.goal, IntentGoal.LEARN, "Should classify as LEARN goal")
        
        # Test COMPARISON goal
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'Compare ProtonMail and Tutanota'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        self.assertIn(intent.declared.goal, [IntentGoal.COMPARISON, IntentGoal.FIND_INFORMATION], 
                     "Should classify as COMPARISON or FIND_INFORMATION goal")
        
        # Test TROUBLESHOOTING goal
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'Why is my email not working?'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        self.assertIn(intent.declared.goal, [IntentGoal.TROUBLESHOOTING, IntentGoal.FIND_INFORMATION],
                     "Should classify as TROUBLESHOOTING or FIND_INFORMATION goal")
        
        # Test PURCHASE goal
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'Where to buy encrypted email service?'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        self.assertIn(intent.declared.goal, [IntentGoal.PURCHASE, IntentGoal.FIND_INFORMATION],
                     "Should classify as PURCHASE or FIND_INFORMATION goal")
    
    def test_ethical_signal_detection(self):
        """Test ethical signal detection functionality"""
        # Test privacy ethical signal
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up privacy-focused encrypted email?'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        privacy_signals = [s for s in intent.inferred.ethicalSignals 
                          if s.dimension == EthicalDimension.PRIVACY]
        self.assertTrue(len(privacy_signals) > 0, "Should infer privacy ethical signal")
    
    def test_temporal_intent_parsing(self):
        """Test temporal intent parsing"""
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up encrypted email on Android, no Google'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        temporal_intent = intent.inferred.temporalIntent
        self.assertIsNotNone(temporal_intent, "Should have temporal intent")
        self.assertIsInstance(temporal_intent.horizon, TemporalHorizon, "Horizon should be TemporalHorizon enum")
        self.assertIsInstance(temporal_intent.recency, Recency, "Recency should be Recency enum")
        self.assertIsInstance(temporal_intent.frequency, Frequency, "Frequency should be Frequency enum")
    
    def test_full_schema_validation(self):
        """Test that the complete intent structure is properly formed"""
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up E2E encrypted email on Android, no big tech solutions'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        # Check that required fields are present
        self.assertIsNotNone(intent.intentId, "Should have intent ID")
        self.assertIsNotNone(intent.context, "Should have context")
        self.assertIsNotNone(intent.declared, "Should have declared intent")
        self.assertIsNotNone(intent.inferred, "Should have inferred intent")
        self.assertIsNotNone(intent.expiresAt, "Should have expiration time")
        
        # Check context fields
        self.assertEqual(intent.context["product"], "search", "Should have correct product")
        self.assertIn("sessionId", intent.context, "Should have session ID")
        self.assertIn("timestamp", intent.context, "Should have timestamp")
        
        # Check declared fields
        self.assertIsNotNone(intent.declared.query, "Should have query")
        self.assertIsNotNone(intent.declared.goal, "Should have goal")
        self.assertIsInstance(intent.declared.constraints, list, "Should have constraints list")
        self.assertIsInstance(intent.declared.negativePreferences, list, "Should have negative preferences list")
        self.assertIsInstance(intent.declared.skillLevel, SkillLevel, "Should have skill level")
        
        # Check inferred fields
        self.assertIsInstance(intent.inferred.useCases, list, "Should have use cases list")
        self.assertIsNotNone(intent.inferred.temporalIntent, "Should have temporal intent")
        self.assertIsInstance(intent.inferred.ethicalSignals, list, "Should have ethical signals list")
        self.assertIsNotNone(intent.inferred.resultType, "Should have result type")
        self.assertIsNotNone(intent.inferred.complexity, "Should have complexity")
    
    def test_negative_preference_extraction(self):
        """Test negative preference extraction"""
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up encrypted email, no big tech solutions'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        self.assertIn("no big tech", intent.declared.negativePreferences, "Should extract 'no big tech' preference")
        
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up open source encrypted email?'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        intent = response.intent
        
        self.assertIn("no proprietary", intent.declared.negativePreferences, "Should extract 'no proprietary' preference")
    
    def test_response_metrics(self):
        """Test that response includes proper metrics"""
        request = IntentExtractionRequest(
            product='search',
            input={'text': 'How to set up encrypted email?'},
            context={'sessionId': 'test_session', 'userLocale': 'en-US'}
        )
        response = extract_intent(request)
        
        self.assertIsNotNone(response.extractionMetrics, "Should have extraction metrics")
        self.assertIn('confidence', response.extractionMetrics, "Should have confidence metric")
        self.assertIn('extractedDimensions', response.extractionMetrics, "Should have extracted dimensions")
        self.assertIn('warnings', response.extractionMetrics, "Should have warnings field")
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(response.extractionMetrics['confidence'], 0.0, "Confidence should be >= 0")
        self.assertLessEqual(response.extractionMetrics['confidence'], 1.0, "Confidence should be <= 1")


if __name__ == '__main__':
    unittest.main()