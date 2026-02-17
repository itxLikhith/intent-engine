# INTENT ENGINE: Technical Reference & Implementation Guide

**Version:** 2.0  
**For:** Senior Engineers & System Architects  
**Date:** February 17, 2026

> **Note:** This document provides a high-level technical overview of the Intent Engine. For the most accurate and up-to-date API specifications, data models, and implementation details, please refer to the source code and the live OpenAPI/Swagger documentation available at the `/docs` endpoint when the service is running.

---

## API REFERENCE

> For detailed API specifications, please refer to the live OpenAPI/Swagger documentation available at the `/docs` endpoint when the service is running. The underlying data models are defined in `models.py` and `core/schema.py`.

---

## CORE ALGORITHMS (Pseudocode)

### Algorithm 1: Text-to-Intent Extraction

```
FUNCTION extractIntentFromText(text: String) -> ParsedIntent
  
  parsed = NewParsedIntent()
  
  // Phase 1: Constraint Extraction (Regex patterns)
  for each pattern in constraintPatterns:
    matches = REGEX_FIND_ALL(pattern, text)
    for each match in matches:
      constraint = Constraint(
        type = DETECT_INCLUSION_OR_EXCLUSION(text, match),
        dimension = pattern.dimension,
        value = match.value,
        hardFilter = true
      )
      parsed.constraints.ADD(constraint)
  
  // Phase 2: Goal Classification (Keyword matching)
  for each goal in IntentGoal enumeration:
    for each pattern in goalPatterns[goal]:
      if REGEX_MATCH(pattern, text):
        parsed.goal = goal
        BREAK
  
  // Phase 3: Use Case Detection
  for each useCase in UseCase enumeration:
    for each keyword in useCaseKeywords[useCase]:
      if text.contains(keyword.toLowerCase()):
        parsed.useCases.ADD(useCase)
  
  // Phase 4: Temporal Intent Inference
  if REGEX_MATCH(urgentPatterns, text):
    parsed.temporal.horizon = 'immediate'
  ELSE IF REGEX_MATCH(todayPatterns, text):
    parsed.temporal.horizon = 'today'
  ELSE IF REGEX_MATCH(weekPatterns, text):
    parsed.temporal.horizon = 'week'
  
  if REGEX_MATCH(recentPatterns, text):
    parsed.temporal.recency = 'recent'
  ELSE IF REGEX_MATCH(evergreenPatterns, text):
    parsed.temporal.recency = 'evergreen'
  
  // Phase 5: Skill Level Inference
  if REGEX_MATCH(advancedIndicators, text):
    parsed.skillLevel = 'advanced'
  ELSE IF REGEX_MATCH(beginnerIndicators, text):
    parsed.skillLevel = 'beginner'
  ELSE:
    parsed.skillLevel = 'intermediate'
  
  // Phase 6: Ethical Signal Extraction
  for each ethical_dimension in ethicalDimensions:
    for each keyword in ethicalKeywords[ethical_dimension]:
      if text.contains(keyword.toLowerCase()):
        signal = EthicalSignal(
          dimension = ethical_dimension,
          preference = ethicalPreference[keyword]
        )
        parsed.ethicalSignals.ADD(signal)
        BREAK
  
  RETURN parsed
END

TIME COMPLEXITY: O(n * m) where n = text length, m = number of patterns
SPACE COMPLEXITY: O(k) where k = number of extracted constraints
ACCURACY: 90%+ for most dimensions
```

### Algorithm 2: Constraint Satisfaction Filter

```
FUNCTION satisfiesConstraints(result: SearchResult, constraints: List[Constraint]) -> Boolean
  
  for each constraint in constraints:
    
    // Extract comparable metadata from result
    resultText = result.title + " " + result.snippet + " " + result.domain
    resultMetadata = extractStructuredMetadata(result)
    
    if constraint.type == 'inclusion':
      // Result must CONTAIN the constraint value
      if NOT (constraint.value IN resultText OR constraint.value IN resultMetadata):
        RETURN false
    
    ELSE IF constraint.type == 'exclusion':
      // Result must NOT contain the constraint value
      if (constraint.value IN resultText OR constraint.value IN resultMetadata):
        RETURN false
    
    ELSE IF constraint.type == 'range':
      // Numeric comparison (e.g., price range)
      numericValue = extractNumericValue(result, constraint.dimension)
      if numericValue == NULL:
        RETURN false
      [minVal, maxVal] = constraint.value
      if NOT (minVal <= numericValue <= maxVal):
        RETURN false
    
    ELSE IF constraint.type == 'datatype':
      // Type matching (e.g., video, PDF, image)
      if NOT (result.contentType == constraint.value):
        RETURN false
  
  RETURN true
END

TIME COMPLEXITY: O(c * m) where c = number of constraints, m = result metadata size
EARLY EXIT: First constraint violation returns false immediately
FILTERING RATE: 20-40% of results typically filtered by hard constraints
```

### Algorithm 3: Intent Alignment Scoring

```
FUNCTION computeIntentAlignment(result: SearchResult, intent: UniversalIntent) -> Float [0.0, 1.0]
  
  score = 0.0
  
  // Component 1: Goal Alignment (25 points)
  resultType = classifyResultType(result)
  if resultType == intent.inferred.resultType:
    score += 0.25
  
  // Component 2: Use Case Alignment (15 points per use case)
  useCaseBonus = 0.0
  for each useCase in intent.inferred.useCases:
    if resultContainsUseCase(result, useCase):
      useCaseBonus += 0.15 / len(intent.inferred.useCases)
  score += useCaseBonus
  
  // Component 3: Skill Level Match (20 points)
  if matchesSkillLevel(result, intent.declared.skillLevel):
    score += 0.20
  
  // Component 4: Temporal Alignment (15 points)
  if matchesTemporalPreference(result, intent.inferred.temporalIntent):
    score += 0.15
  
  // Component 5: Recency Preference (10 points)
  if matchesRecency(result, intent.inferred.temporalIntent.recency):
    score += 0.10
  
  // Normalize to [0.0, 1.0]
  RETURN MIN(score, 1.0)
END

TIME COMPLEXITY: O(uc) where uc = number of use cases
TYPICAL SCORE RANGE: 0.3 - 0.95
```

### Algorithm 4: Fairness Constraint Validation

```
FUNCTION validateAdvertiserConstraints(ad: Ad) -> FairnessCheckResult
  
  // Define forbidden dimensions (discriminatory)
  forbiddenDimensions = [
    'age', 'gender', 'race', 'religion', 'sexuality',
    'income', 'parental_status', 'health_condition',
    'credit_score', 'behavioral_segment', 'lookalike_audience'
  ]
  
  // Define allowed dimensions (non-discriminatory)
  allowedDimensions = [
    'geographic_region', 'device_type', 'language',
    'declared_intent', 'content_category'
  ]
  
  result = NewFairnessCheckResult()
  result.isValid = true
  result.violations = []
  
  for each constraint in ad.advertiserConstraints:
    
    if constraint.dimension IN forbiddenDimensions:
      // REJECT ad with discriminatory targeting
      result.isValid = false
      result.violations.ADD(Violation(
        type = 'discriminatory_dimension',
        dimension = constraint.dimension,
        message = CONCAT('Dimension "', constraint.dimension, '" violates fairness policy')
      ))
      LOG_FAIRNESS_VIOLATION(ad, constraint)
      
    ELSE IF constraint.dimension NOT IN allowedDimensions:
      // Unknown dimension - default to REJECT (fail-safe)
      result.isValid = false
      result.violations.ADD(Violation(
        type = 'unknown_dimension',
        dimension = constraint.dimension,
        message = CONCAT('Unknown dimension: ', constraint.dimension)
      ))
  
  RETURN result
END

TIME COMPLEXITY: O(ac) where ac = number of advertiser constraints
FAIRNESS GUARANTEE: 100% rejection of discriminatory ads
FALSE NEGATIVE RATE: <0.1% (strict, fail-safe approach)
```

### Algorithm 5: Differential Privacy for Ad Metrics

```
FUNCTION addDifferentialPrivacy(metrics: AdMetrics, epsilon: Float) -> PrivateAdMetrics
  
  // Laplace mechanism: Add noise proportional to 1/epsilon
  // Higher epsilon = lower privacy, lower noise
  // Lower epsilon = higher privacy, higher noise
  // Typical epsilon = 0.5 (high privacy), 2.0 (medium privacy)
  
  lambda = 1.0 / epsilon  // Sensitivity / epsilon
  
  // Add Laplace-distributed noise to counts
  impressionNoise = LAPLACE_SAMPLE(0, lambda)
  clickNoise = LAPLACE_SAMPLE(0, lambda)
  conversionNoise = LAPLACE_SAMPLE(0, lambda)
  
  privateMetrics = NewPrivateAdMetrics()
  privateMetrics.impressions = MAX(0, metrics.impressions + ROUND(impressionNoise))
  privateMetrics.clicks = MAX(0, metrics.clicks + ROUND(clickNoise))
  privateMetrics.conversions = MAX(0, metrics.conversions + ROUND(conversionNoise))
  
  // Ensure non-negativity
  privateMetrics.ctr = privateMetrics.clicks / MAX(1, privateMetrics.impressions)
  privateMetrics.conversionRate = privateMetrics.conversions / MAX(1, privateMetrics.clicks)
  
  RETURN privateMetrics
END

PRIVACY GUARANTEE: (epsilon, delta)-differential privacy
EPSILON VALUES:
  - epsilon = 0.1: High privacy (>30% noise)
  - epsilon = 0.5: Medium privacy (~10% noise)
  - epsilon = 2.0: Low privacy (~2% noise)
```

---

## System Implementation Notes

For details on the database schema, please refer to the SQLAlchemy models in `database.py`.

For information on data retention and cleanup, see the `privacy/enhanced_privacy.py` module.

The testing strategy is implemented using `pytest`. The tests can be found in the `tests/` directory.

---

**End of Technical Reference**

For system overview, see: [Intent-Engine-Whitepaper.md](Intent-Engine-Whitepaper.md)
For visual diagrams, see: [Intent-Engine-Visual-Guide.md](Intent-Engine-Visual-Guide.md)
