"""
Simple test to verify the intent extractor works as expected
"""
from extraction.extractor import IntentExtractionRequest, extract_intent


def test_basic_functionality():
    """Test basic functionality of the intent extractor"""
    # Test case from the requirements
    request = IntentExtractionRequest(
        product='search',
        input={'text': 'How to set up E2E encrypted email on Android, no big tech'},
        context={'sessionId': 'test_session_123', 'userLocale': 'en-IN'}
    )
    
    response = extract_intent(request)
    intent = response.intent
    
    print("=== Intent Extraction Results ===")
    print(f"Intent ID: {intent.intentId}")
    print(f"Expires At: {intent.expiresAt}")
    print(f"Product: {intent.context['product']}")
    print(f"Session ID: {intent.context['sessionId']}")
    print(f"User Locale: {intent.context['userLocale']}")
    
    print(f"\nDeclared Goal: {intent.declared.goal}")
    print(f"Skill Level: {intent.declared.skillLevel}")
    print(f"Negative Preferences: {intent.declared.negativePreferences}")
    
    print("\nConstraints:")
    for constraint in intent.declared.constraints:
        print(f"  - Type: {constraint.type.value}, Dimension: {constraint.dimension}, Value: {constraint.value}")
    
    print(f"\nInferred Use Cases: {[uc.value for uc in intent.inferred.useCases]}")
    print(f"Result Type: {intent.inferred.resultType.value if intent.inferred.resultType else None}")
    print(f"Complexity: {intent.inferred.complexity.value}")
    
    print(f"\nEthical Signals:")
    for signal in intent.inferred.ethicalSignals:
        print(f"  - Dimension: {signal.dimension.value}, Preference: {signal.preference}")
    
    print(f"\nTemporal Intent:")
    if intent.inferred.temporalIntent:
        print(f"  - Horizon: {intent.inferred.temporalIntent.horizon.value}")
        print(f"  - Recency: {intent.inferred.temporalIntent.recency.value}")
        print(f"  - Frequency: {intent.inferred.temporalIntent.frequency.value}")
    
    print(f"\nExtraction Metrics: {response.extractionMetrics}")


if __name__ == "__main__":
    test_basic_functionality()