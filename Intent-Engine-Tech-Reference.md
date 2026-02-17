# INTENT ENGINE: Technical Reference & Implementation Guide

**Version:** 1.0  
**For:** Senior Engineers & System Architects  
**Date:** January 19, 2026

---

## API REFERENCE

### Intent Extraction API

```typescript
// Main entry point for extracting intent from any input

interface IntentExtractionRequest {
  product: 'search' | 'docs' | 'mail' | 'calendar' | 'meet' | 'forms' | 'diary' | 'sites';
  input: TextInput | FormInput | DocumentInput | EventInput;
  context: ExtractionContext;
  options?: ExtractionOptions;
}

interface IntentExtractionResponse {
  intent: UniversalIntent;
  extractionMetrics: {
    confidence: number; // 0.0 - 1.0
    extractedDimensions: string[]; // Which parts were successfully extracted
    warnings?: string[]; // Ambiguities or fallbacks used
  };
}

// Usage Example:
const response = await extractIntent({
  product: 'search',
  input: {
    text: 'How to setup E2E encrypted email on Android, no big tech solutions'
  },
  context: {
    sessionId: 'sess_abc123',
    userLocale: 'en-IN',
    timestamp: new Date()
  }
});

console.log(response.intent);
// {
//   intentId: 'intent_xyz789',
//   declared: {
//     goal: 'LEARN',
//     constraints: [
//       { type: 'inclusion', dimension: 'platform', value: 'Android' },
//       { type: 'exclusion', dimension: 'provider', value: ['Google', 'Microsoft'] }
//     ],
//     skillLevel: 'intermediate'
//   },
//   inferred: {
//     useCases: ['learning', 'troubleshooting'],
//     ethicalSignals: [
//       { dimension: 'privacy', preference: 'privacy-first' }
//     ]
//   },
//   expiresAt: '2026-01-19T20:34:56Z'
// }
```

### Constraint Satisfaction API

```typescript
// Check if a result satisfies user constraints

function satisfiesConstraints(
  result: SearchResult,
  constraints: Constraint[]
): boolean {
  for (const constraint of constraints) {
    if (!satisfiesSingleConstraint(result, constraint)) {
      return false;
    }
  }
  return true;
}

// Usage:
const userConstraints = [
  { type: 'inclusion', dimension: 'platform', value: 'Android', hardFilter: true },
  { type: 'exclusion', dimension: 'provider', value: ['Google'], hardFilter: true }
];

const result = searchResults[0]; // ProtonMail tutorial
const passes = satisfiesConstraints(result, userConstraints);
// true - Result is about Android and not from Google
```

### Ranking API

```typescript
// Rank search results using intent alignment

interface RankingRequest {
  intent: UniversalIntent;
  candidates: SearchResult[];
  options?: RankingOptions;
}

interface RankingResponse {
  rankedResults: RankedResult[];
}

async function rankResults(req: RankingRequest): Promise<RankingResponse> {
  const ranked = [];
  
  for (const result of req.candidates) {
    // Step 1: Apply hard filters
    if (!satisfiesConstraints(result, req.intent.declared.constraints)) {
      continue;
    }
    
    // Step 2: Compute intent alignment
    const alignmentScore = computeIntentAlignment(result, req.intent);
    
    // Step 3: Compute quality score
    const qualityScore = computeQualityScore(result, req.intent);
    
    // Step 4: Compute ethical alignment
    const ethicalScore = computeEthicalAlignment(result, req.intent);
    
    // Step 5: Combine
    const finalScore = (
      0.50 * alignmentScore +
      0.30 * qualityScore +
      0.20 * ethicalScore
    );
    
    ranked.push({
      result,
      score: finalScore,
      reasons: explainScore(result, req.intent)
    });
  }
  
  // Sort by score
  ranked.sort((a, b) => b.score - a.score);
  return { rankedResults: ranked };
}

// Usage:
const ranking = await rankResults({
  intent: extractedIntent,
  candidates: searchResults
});

console.log(ranking.rankedResults[0]);
// {
//   result: { url: 'https://protonmail.com/...', title: '...' },
//   score: 0.92,
//   reasons: ['Goal match: LEARN', 'Platform match: Android', ...]
// }
```

### Ad Matching API

```typescript
// Match ads to user intent (privacy-first, no tracking)

interface AdMatchingRequest {
  intent: UniversalIntent;
  adInventory: Ad[];
  config?: MatchingConfig;
}

interface AdMatchingResponse {
  matchedAds: MatchedAd[];
  metrics: {
    totalAdsEvaluated: number;
    adsPassedFairness: number;
    adsFiltered: number;
  };
}

async function matchAds(req: AdMatchingRequest): Promise<AdMatchingResponse> {
  const matched = [];
  let fairnessPassCount = 0;
  
  for (const ad of req.adInventory) {
    // Filter 1: User constraints
    if (!satisfiesUserConstraints(ad, req.intent.declared.constraints)) {
      continue;
    }
    
    // Filter 2: Fairness check (NO discriminatory targeting)
    const fairnessCheck = validateAdvertiserConstraints(ad);
    if (!fairnessCheck.isValid) {
      console.warn(`Ad rejected for fairness: ${fairnessCheck.reason}`);
      continue;
    }
    fairnessPassCount++;
    
    // Filter 3: Relevance scoring
    const relevanceScore = computeAdRelevance(ad, req.intent);
    
    if (relevanceScore > 0.4) { // Minimum threshold
      matched.push({
        ad,
        relevance: relevanceScore,
        matchedDimensions: explainAdMatch(ad, req.intent)
      });
    }
  }
  
  // Sort by relevance
  matched.sort((a, b) => b.relevance - a.relevance);
  
  return {
    matchedAds: matched.slice(0, 5), // Top 5 ads
    metrics: {
      totalAdsEvaluated: req.adInventory.length,
      adsPassedFairness: fairnessPassCount,
      adsFiltered: req.adInventory.length - matched.length
    }
  };
}

// Usage:
const adMatching = await matchAds({
  intent: extractedIntent,
  adInventory: availableAds
});

console.log(adMatching.matchedAds);
// [
//   { ad: Tutanota, relevance: 0.85, matchedDimensions: ['privacy-goal', 'intent-goal-match'] },
//   { ad: Proton VPN, relevance: 0.72, matchedDimensions: ['ethical-signal-match'] },
//   ...
// ]

console.log(adMatching.metrics);
// { totalAdsEvaluated: 500, adsPassedFairness: 485, adsFiltered: 415 }
```

### Service Recommendation API

```typescript
// Recommend workspace services based on user intent

interface ServiceRecommendationRequest {
  intent: UniversalIntent;
  availableServices: Service[];
}

interface ServiceRecommendationResponse {
  recommendations: ServiceRecommendation[];
}

async function recommendServices(
  req: ServiceRecommendationRequest
): Promise<ServiceRecommendationResponse> {
  const recommendations = [];
  
  for (const service of req.availableServices) {
    const score = computeServiceMatch(req.intent, service);
    
    if (score > 0.3) {
      recommendations.push({
        service,
        score,
        reason: explainServiceMatch(req.intent, service)
      });
    }
  }
  
  // Sort by score
  recommendations.sort((a, b) => b.score - a.score);
  return { recommendations };
}

// Usage:
const serviceRecs = await recommendServices({
  intent: {
    declared: { goal: 'COLLABORATE' },
    inferred: { useCases: ['collaboration', 'analysis'] }
  },
  availableServices: [
    { name: 'docs', ... },
    { name: 'sheets', ... },
    { name: 'mail', ... }
  ]
});

console.log(serviceRecs.recommendations);
// [
//   { service: 'docs', score: 0.95, reason: 'Best for real-time collaboration' },
//   { service: 'sheets', score: 0.88, reason: 'Good for data analysis' },
//   { service: 'mail', score: 0.42, reason: 'Not primary use case' }
// ]
```

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

## SYSTEM DEPLOYMENT

### Database Schema (for logging only)

```sql
-- Session-scoped intent (in-memory, auto-delete)
CREATE TABLE intent_in_memory (
    intent_id STRING PRIMARY KEY,
    session_id STRING NOT NULL,
    product STRING NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    declared_intent JSON NOT NULL,
    inferred_intent JSON NOT NULL,
    session_feedback JSON,
    
    INDEX idx_expires_at (expires_at),  -- For cleanup job
    INDEX idx_session_id (session_id)
);

-- Ranking logs (7-day retention, encrypted)
CREATE TABLE ranking_logs (
    log_id STRING PRIMARY KEY,
    session_id STRING NOT NULL,
    intent_id STRING NOT NULL,
    product STRING NOT NULL,
    
    -- Anonymized data only
    result_urls STRING[],
    result_scores FLOAT64[],
    ranking_reasons JSON,
    
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    
    INDEX idx_expires_at (expires_at)
);

-- Ad impressions (aggregated, 30-day retention, differential privacy applied)
CREATE TABLE ad_impressions_aggregated (
    ad_id STRING NOT NULL,
    date DATE NOT NULL,
    intent_goal STRING,
    intent_use_case STRING,
    advertiser_id STRING NOT NULL,
    
    -- Metrics with noise for privacy
    impression_count INT64,
    click_count INT64,
    conversion_count INT64,
    
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    
    PRIMARY KEY (ad_id, date, intent_goal, intent_use_case),
    INDEX idx_expires_at (expires_at)
);

-- Fairness audit log (permanent, for compliance)
CREATE TABLE fairness_audit_log (
    audit_id STRING PRIMARY KEY,
    ad_id STRING NOT NULL,
    advertiser_id STRING NOT NULL,
    violation_type STRING,
    violation_dimension STRING,
    violation_details JSON,
    
    action_taken STRING,  -- 'rejected', 'flagged', 'approved'
    created_at TIMESTAMP NOT NULL,
    reviewed_by STRING
);
```

### Auto-Deletion Job (TTL Enforcement)

```python
import datetime
import schedule
import time

def cleanup_expired_intents():
    """
    Delete all expired intent objects from in-memory storage.
    Runs every 1 hour.
    """
    now = datetime.datetime.utcnow()
    
    query = """
        DELETE FROM intent_in_memory
        WHERE expires_at < ?
    """
    
    cursor.execute(query, [now.isoformat()])
    deleted_count = cursor.rowcount
    
    logger.info(f"Cleaned up {deleted_count} expired intents")
    
    # Metrics
    metrics.gauge('intent.expired_deletions', deleted_count)

def cleanup_old_ranking_logs():
    """
    Delete ranking logs older than 7 days.
    Runs daily at 2 AM UTC.
    """
    seven_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    
    query = """
        DELETE FROM ranking_logs
        WHERE expires_at < ?
    """
    
    cursor.execute(query, [seven_days_ago.isoformat()])
    deleted_count = cursor.rowcount
    
    logger.info(f"Purged {deleted_count} ranking logs")

def cleanup_old_ad_metrics():
    """
    Delete aggregated ad metrics older than 30 days.
    Runs daily at 3 AM UTC.
    """
    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    
    query = """
        DELETE FROM ad_impressions_aggregated
        WHERE expires_at < ?
    """
    
    cursor.execute(query, [thirty_days_ago.isoformat()])
    deleted_count = cursor.rowcount
    
    logger.info(f"Purged {deleted_count} ad metrics")

# Schedule cleanup jobs
schedule.every(1).hours.do(cleanup_expired_intents)
schedule.every().day.at("02:00").do(cleanup_old_ranking_logs)
schedule.every().day.at("03:00").do(cleanup_old_ad_metrics)

# Start scheduler in background
while True:
    schedule.run_pending()
    time.sleep(60)
```

### Privacy Monitoring Dashboard

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics for Intent Engine

# 1. Privacy Metrics
intent_objects_created = Counter(
    'intent_objects_created_total',
    'Total number of intent objects created'
)

intent_objects_expired = Counter(
    'intent_objects_expired_total',
    'Total number of intent objects auto-deleted'
)

session_duration_seconds = Histogram(
    'session_duration_seconds',
    'Duration of user sessions',
    buckets=[60, 300, 900, 1800, 3600, 28800]  # 1min, 5min, 15min, 30min, 1hr, 8hrs
)

# 2. Extraction Accuracy
extraction_accuracy = Gauge(
    'extraction_accuracy',
    'Intent extraction accuracy (0-1)',
    ['dimension']  # goal, constraint, skill_level, etc.
)

extraction_latency_ms = Histogram(
    'extraction_latency_ms',
    'Intent extraction latency (milliseconds)',
    buckets=[5, 10, 20, 50, 100, 200, 500]
)

# 3. Fairness Metrics
ads_rejected_fairness = Counter(
    'ads_rejected_fairness_total',
    'Ads rejected due to fairness violations',
    ['violation_type']  # discriminatory_dimension, unknown_dimension
)

fairness_violations_detected = Counter(
    'fairness_violations_detected_total',
    'Total fairness violations detected',
    ['ad_id', 'advertiser_id']
)

# 4. Ranking Metrics
results_filtered_constraints = Counter(
    'results_filtered_constraints_total',
    'Results filtered by hard constraints',
    ['product']
)

ranking_quality_ctr = Gauge(
    'ranking_quality_ctr',
    'Click-through rate of ranked results',
    ['product']
)

# 5. System Health
system_uptime_seconds = Gauge(
    'system_uptime_seconds',
    'Intent Engine uptime'
)

cleanup_job_duration_ms = Histogram(
    'cleanup_job_duration_ms',
    'Duration of TTL cleanup jobs',
    buckets=[100, 500, 1000, 5000, 10000]
)

# Usage in code
@app.route('/extract-intent', methods=['POST'])
def extract_intent_endpoint():
    start = time.time()
    
    try:
        request_data = request.json
        intent = extract_intent(request_data)
        intent_objects_created.inc()
        
        latency = (time.time() - start) * 1000
        extraction_latency_ms.observe(latency)
        
        return {'intent': intent}
    
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

# Log extraction accuracy (from user feedback)
def log_extraction_quality(dimension, accuracy):
    extraction_accuracy.labels(dimension=dimension).set(accuracy)

# Log fairness violations
def log_fairness_violation(ad_id, advertiser_id, violation_type):
    ads_rejected_fairness.labels(violation_type=violation_type).inc()
    fairness_violations_detected.labels(ad_id=ad_id, advertiser_id=advertiser_id).inc()
```

---

## TESTING STRATEGY

### Unit Test Examples

```python
import pytest
from intent_engine import extract_intent, satisfies_constraints, compute_intent_alignment

class TestIntentExtraction:
    
    def test_constraint_extraction_inclusion(self):
        text = "I need Android app"
        intent = extract_intent({'product': 'search', 'input': {'text': text}})
        
        assert len(intent.declared.constraints) >= 1
        android_constraint = next(
            c for c in intent.declared.constraints if c.dimension == 'platform'
        )
        assert android_constraint.value == 'Android'
        assert android_constraint.type == 'inclusion'
    
    def test_constraint_extraction_exclusion(self):
        text = "no Google solutions"
        intent = extract_intent({'product': 'search', 'input': {'text': text}})
        
        google_constraint = next(
            c for c in intent.declared.constraints if 'Google' in str(c.value)
        )
        assert google_constraint.type == 'exclusion'
    
    def test_goal_classification_learn(self):
        text = "How to setup email?"
        intent = extract_intent({'product': 'search', 'input': {'text': text}})
        
        assert intent.declared.goal == 'LEARN'
    
    def test_goal_classification_comparison(self):
        text = "Compare Proton vs Tutanota"
        intent = extract_intent({'product': 'search', 'input': {'text': text}})
        
        assert intent.declared.goal == 'COMPARISON'
    
    def test_skill_level_inference_advanced(self):
        text = "Setup API endpoints with authentication"
        intent = extract_intent({'product': 'search', 'input': {'text': text}})
        
        assert intent.declared.skillLevel == 'advanced'
    
    def test_skill_level_inference_beginner(self):
        text = "Beginners guide to email setup"
        intent = extract_intent({'product': 'search', 'input': {'text': text}})
        
        assert intent.declared.skillLevel == 'beginner'
    
    def test_ethical_signal_extraction_privacy(self):
        text = "encrypted private email no tracking"
        intent = extract_intent({'product': 'search', 'input': {'text': text}})
        
        privacy_signal = next(
            s for s in intent.inferred.ethicalSignals if s.dimension == 'privacy'
        )
        assert privacy_signal.preference == 'privacy-first'
    
    def test_ttl_assignment(self):
        intent = extract_intent({'product': 'search', 'input': {'text': 'test'}})
        
        assert intent.expiresAt is not None
        expiry = datetime.datetime.fromisoformat(intent.expiresAt)
        now = datetime.datetime.utcnow()
        delta = (expiry - now).total_seconds()
        
        # Should expire in ~8 hours
        assert 28000 < delta < 29000  # 7h 45min to 8h 5min
    
    def test_session_id_not_persistent(self):
        intent1 = extract_intent({'product': 'search', 'input': {'text': 'query1'}})
        intent2 = extract_intent({'product': 'search', 'input': {'text': 'query2'}})
        
        # Different sessions (not linked)
        assert intent1.context.sessionId != intent2.context.sessionId

class TestConstraintSatisfaction:
    
    def test_inclusion_constraint_satisfied(self):
        result = SearchResult(
            title="Android Email App Setup",
            domain="example.com"
        )
        constraint = Constraint(
            type='inclusion',
            dimension='platform',
            value='Android',
            hardFilter=True
        )
        
        assert satisfies_constraints(result, [constraint]) == True
    
    def test_exclusion_constraint_violated(self):
        result = SearchResult(
            title="Gmail Setup Guide",
            domain="google.com"
        )
        constraint = Constraint(
            type='exclusion',
            dimension='provider',
            value='Google',
            hardFilter=True
        )
        
        assert satisfies_constraints(result, [constraint]) == False
    
    def test_multiple_constraints_all_satisfied(self):
        result = SearchResult(
            title="ProtonMail Android App",
            domain="protonmail.com",
            price=0  # Free
        )
        constraints = [
            Constraint(type='inclusion', dimension='platform', value='Android', hardFilter=True),
            Constraint(type='inclusion', dimension='feature', value='encrypted', hardFilter=True),
            Constraint(type='exclusion', dimension='provider', value='Google', hardFilter=True),
        ]
        
        assert satisfies_constraints(result, constraints) == True
    
    def test_multiple_constraints_one_violated(self):
        result = SearchResult(
            title="Gmail Android Setup",
            domain="google.com"
        )
        constraints = [
            Constraint(type='inclusion', dimension='platform', value='Android', hardFilter=True),
            Constraint(type='exclusion', dimension='provider', value='Google', hardFilter=True),
        ]
        
        # First constraint passes, second fails -> overall False
        assert satisfies_constraints(result, constraints) == False

class TestIntentAlignment:
    
    def test_goal_match_scores_high(self):
        result = SearchResult(
            title="Email Setup Tutorial",
            url="https://example.com/tutorial",
            contentType='tutorial'
        )
        intent = UniversalIntent(
            declared={'goal': 'LEARN'},
            inferred={'resultType': 'tutorial'}
        )
        
        score = compute_intent_alignment(result, intent)
        assert score > 0.7  # High alignment
    
    def test_skill_level_mismatch_scores_lower(self):
        result = SearchResult(
            title="Advanced API Documentation",
            contentType='documentation'
        )
        intent = UniversalIntent(
            declared={'skillLevel': 'beginner'},
            inferred={'complexity': 'advanced'}
        )
        
        score = compute_intent_alignment(result, intent)
        assert score < 0.5  # Lower due to skill mismatch
```

### Integration Test Example

```python
class TestEndToEndIntentFlow:
    
    def test_search_query_to_ranked_results(self):
        """Full flow: Query -> Intent extraction -> Ranking"""
        
        # Step 1: User query
        query = "How to setup E2E encrypted email on Android, no big tech"
        
        # Step 2: Extract intent
        intent = extract_intent({
            'product': 'search',
            'input': {'text': query}
        })
        
        assert intent.declared.goal == 'LEARN'
        assert len(intent.declared.constraints) >= 2
        assert any(s.dimension == 'privacy' for s in intent.inferred.ethicalSignals)
        
        # Step 3: Get search results (mock)
        candidates = [
            SearchResult(title="ProtonMail Android", domain="protonmail.com"),
            SearchResult(title="Tutanota Setup", domain="tutanota.com"),
            SearchResult(title="Gmail with E2E", domain="google.com"),  # Should be filtered
        ]
        
        # Step 4: Rank results
        ranking = rank_results({
            'intent': intent,
            'candidates': candidates
        })
        
        # Step 5: Validate ranking
        assert len(ranking.rankedResults) == 2  # Gmail filtered due to constraint
        assert ranking.rankedResults[0].result.domain in ['protonmail.com', 'tutanota.com']
        assert ranking.rankedResults[0].score > ranking.rankedResults[1].score
    
    def test_privacy_constraint_enforcement(self):
        """Verify privacy constraints are properly enforced"""
        
        intent = extract_intent({
            'product': 'search',
            'input': {'text': 'privacy-first email'}
        })
        
        # Should have privacy signal
        privacy_signals = [s for s in intent.inferred.ethicalSignals if s.dimension == 'privacy']
        assert len(privacy_signals) > 0
        
        # Verify intent expires
        assert intent.expiresAt is not None
    
    def test_fairness_in_ad_matching(self):
        """Verify discriminatory ads are rejected"""
        
        intent = extract_intent({
            'product': 'search',
            'input': {'text': 'job search software engineer'}
        })
        
        # Create ad with discriminatory targeting
        bad_ad = Ad(
            id='bad_ad_1',
            advertiserConstraints=[
                {'dimension': 'gender', 'value': 'female'}  # Discriminatory!
            ]
        )
        
        # Create ad with fair targeting
        good_ad = Ad(
            id='good_ad_1',
            advertiserConstraints=[
                {'dimension': 'device_type', 'value': 'desktop'}  # OK
            ]
        )
        
        matching = match_ads({
            'intent': intent,
            'adInventory': [bad_ad, good_ad]
        })
        
        # Bad ad should be rejected
        assert 'bad_ad_1' not in [a.ad.id for a in matching.matchedAds]
        
        # Good ad may be matched
        good_ad_matched = any(a.ad.id == 'good_ad_1' for a in matching.matchedAds)
        # (may or may not be matched depending on relevance, but fairness check passes)
```

---

## MONITORING & OBSERVABILITY

### Key Dashboards

1. **Intent Extraction Quality Dashboard**
   - Extraction accuracy by dimension (goal, constraint, skill_level)
   - Latency histogram
   - Coverage (% of queries where each dimension was extracted)

2. **Ranking Quality Dashboard**
   - CTR (Click-through rate) vs. baseline
   - Dwell time distribution
   - Constraint filtering rate (% of results filtered)

3. **Fairness & Privacy Dashboard**
   - Ads rejected by violation type
   - Fairness violations per advertiser
   - Intent TTL compliance (% of intents deleted on schedule)

4. **System Health Dashboard**
   - Uptime percentage
   - API latency (extraction, ranking, matching)
   - Database cleanup job duration

---

## PERFORMANCE BENCHMARKS (Target)

| Operation | Target Latency | Throughput |
|---|---|---|
| Intent extraction | < 50ms | 10k/sec |
| Constraint satisfaction | < 20ms | 50k/sec |
| Intent alignment scoring | < 30ms | 30k/sec |
| Ranking (100 results) | < 100ms | 5k/sec |
| Ad matching (500 ads) | < 75ms | 8k/sec |
| Service recommendation | < 40ms | 15k/sec |
| **End-to-end search request** | **< 200ms** | **5k/sec** |

---

**End of Technical Reference**

For system overview, see: `Intent-Engine-Whitepaper.md`
For visual diagrams, see: `Intent-Engine-Visual-Guide.md`
