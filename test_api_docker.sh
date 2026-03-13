#!/bin/bash
# Comprehensive API Test Script for Docker Deployment

BASE_URL="http://localhost:8000"
PASS=0
FAIL=0

echo "=============================================="
echo "Intent Engine API - Docker Test Suite"
echo "=============================================="
echo ""

# Test 1: Health Check
echo "Test 1: Health Check"
RESPONSE=$(curl -s "$BASE_URL/")
if echo "$RESPONSE" | grep -q "healthy"; then
    echo "[PASS] Health check passed"
    ((PASS++))
else
    echo "[FAIL] Health check failed: $RESPONSE"
    ((FAIL++))
fi

# Test 2: Status Endpoint
echo ""
echo "Test 2: Status Endpoint"
RESPONSE=$(curl -s "$BASE_URL/status")
if echo "$RESPONSE" | grep -q "uptime"; then
    echo "[PASS] Status endpoint working"
    ((PASS++))
else
    echo "[FAIL] Status endpoint failed: $RESPONSE"
    ((FAIL++))
fi

# Test 3: Intent Extraction
echo ""
echo "Test 3: Intent Extraction"
RESPONSE=$(curl -s -X POST "$BASE_URL/extract-intent" \
  -H "Content-Type: application/json" \
  -d '{"product": "search", "input": {"text": "best laptop for programming under 50000 rupees"}}')
if echo "$RESPONSE" | grep -q "intent"; then
    echo "[PASS] Intent extraction working"
    ((PASS++))
else
    echo "[FAIL] Intent extraction failed: $RESPONSE"
    ((FAIL++))
fi

# Test 4: Ad Matching
echo ""
echo "Test 4: Ad Matching"
RESPONSE=$(curl -s -X POST "$BASE_URL/match-ads" \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "declared": {"query": "best laptop", "goal": "PURCHASE"},
      "inferred": {"useCases": [], "ethicalSignals": []}
    },
    "adInventory": [
      {"id": "ad1", "title": "Budget Laptop", "description": "Affordable", "targetingConstraints": {}, "forbiddenDimensions": [], "qualityScore": 0.8, "ethicalTags": ["privacy"]}
    ]
  }')
if echo "$RESPONSE" | grep -q "matchedAds"; then
    echo "[PASS] Ad matching working"
    ((PASS++))
else
    echo "[FAIL] Ad matching failed: $RESPONSE"
    ((FAIL++))
fi

# Test 5: Service Recommendation
echo ""
echo "Test 5: Service Recommendation"
RESPONSE=$(curl -s -X POST "$BASE_URL/recommend-services" \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "declared": {"query": "best laptop", "goal": "SEARCH"},
      "inferred": {"useCases": [], "ethicalSignals": []}
    },
    "availableServices": [
      {"id": "search", "name": "Search Engine", "supportedGoals": ["SEARCH"], "primaryUseCases": ["research"], "temporalPatterns": ["immediate"], "ethicalAlignment": ["privacy"]}
    ]
  }')
if echo "$RESPONSE" | grep -q "recommendations"; then
    echo "[PASS] Service recommendation working"
    ((PASS++))
else
    echo "[FAIL] Service recommendation failed: $RESPONSE"
    ((FAIL++))
fi

# Test 6: Campaign Creation
echo ""
echo "Test 6: Campaign Creation"
RESPONSE=$(curl -s -X POST "$BASE_URL/campaigns" \
  -H "Content-Type: application/json" \
  -d '{
    "advertiserId": 1,
    "name": "Test Campaign",
    "startDate": "2026-02-18",
    "endDate": "2026-03-18",
    "budget": 5000.0,
    "dailyBudget": 200.0,
    "status": "active"
  }')
if echo "$RESPONSE" | grep -q "id"; then
    echo "[PASS] Campaign creation working"
    ((PASS++))
else
    echo "[FAIL] Campaign creation failed: $RESPONSE"
    ((FAIL++))
fi

# Test 7: List Campaigns
echo ""
echo "Test 7: List Campaigns"
RESPONSE=$(curl -s "$BASE_URL/campaigns?limit=10")
if echo "$RESPONSE" | grep -q "items"; then
    echo "[PASS] List campaigns working"
    ((PASS++))
else
    echo "[FAIL] List campaigns failed: $RESPONSE"
    ((FAIL++))
fi

# Test 8: Consent Record
echo ""
echo "Test 8: Consent Record"
RESPONSE=$(curl -s -X POST "$BASE_URL/consent/record" \
  -H "Content-Type: application/json" \
  -d '{"userId": "test-user-123", "consentType": "analytics", "granted": true}')
if echo "$RESPONSE" | grep -q "consent"; then
    echo "[PASS] Consent record working"
    ((PASS++))
else
    echo "[FAIL] Consent record failed: $RESPONSE"
    ((FAIL++))
fi

# Test 9: Fraud Detection
echo ""
echo "Test 9: Fraud Detection"
RESPONSE=$(curl -s -X POST "$BASE_URL/fraud-detection" \
  -H "Content-Type: application/json" \
  -d '{
    "adId": 1,
    "eventType": "click",
    "ipAddress": "192.168.1.1",
    "userAgent": "Mozilla/5.0",
    "timestamp": "2026-02-18T10:00:00Z",
    "metadata": {"clickCount": 5, "timeWindowSeconds": 60}
  }')
if echo "$RESPONSE" | grep -q "fraud"; then
    echo "[PASS] Fraud detection working"
    ((PASS++))
else
    echo "[FAIL] Fraud detection failed: $RESPONSE"
    ((FAIL++))
fi

# Test 10: A/B Test Creation
echo ""
echo "Test 10: A/B Test Creation"
RESPONSE=$(curl -s -X POST "$BASE_URL/abtests" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Experiment",
    "description": "Testing new ad format",
    "status": "active",
    "startDate": "2026-02-18",
    "endDate": "2026-03-18",
    "variants": [
      {"name": "control", "weight": 0.5, "metadata": {}},
      {"name": "treatment", "weight": 0.5, "metadata": {}}
    ]
  }')
if echo "$RESPONSE" | grep -q "id"; then
    echo "[PASS] A/B test creation working"
    ((PASS++))
else
    echo "[FAIL] A/B test creation failed: $RESPONSE"
    ((FAIL++))
fi

# Summary
echo ""
echo "=============================================="
echo "TEST SUMMARY"
echo "=============================================="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "Total:  $((PASS + FAIL))"
echo "=============================================="

if [ $FAIL -eq 0 ]; then
    echo "[SUCCESS] All tests passed!"
    exit 0
else
    echo "[FAILURE] Some tests failed"
    exit 1
fi
