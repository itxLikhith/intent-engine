#!/bin/bash
# Comprehensive API Test Script - Simulates Real User Workflows
# Tests all major functionality end-to-end

BASE_URL="http://localhost:8000"
PASS=0
FAIL=0

echo "============================================================"
echo "Intent Engine API - Comprehensive User Workflow Tests"
echo "============================================================"
echo ""

# Helper function for colored output
test_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

test_result() {
    if [ $1 -eq 0 ]; then
        echo "[PASS] $2"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $2"
        FAIL=$((FAIL + 1))
    fi
}

# Test 1: Health & Status
test_header "TEST 1: Health Check & Service Status"

RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)
if [ "$HTTP_CODE" = "200" ] && echo "$BODY" | grep -q "healthy"; then
    test_result 0 "Health check endpoint"
else
    test_result 1 "Health check endpoint (HTTP $HTTP_CODE)"
fi

RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/status")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ]; then
    test_result 0 "Status endpoint"
else
    test_result 1 "Status endpoint (HTTP $HTTP_CODE)"
fi

# Test 2: Intent Extraction Workflow
test_header "TEST 2: Intent Extraction (User Search Queries)"

# Test case 1: Shopping query
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/extract-intent" \
  -H "Content-Type: application/json" \
  -d '{"product": "search", "input": {"text": "best laptop for programming under 50000 rupees"}}')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ] && echo "$RESPONSE" | grep -q "intent"; then
    test_result 0 "Extract intent - Shopping query"
else
    test_result 1 "Extract intent - Shopping query (HTTP $HTTP_CODE)"
fi

# Test case 2: Privacy-focused query
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/extract-intent" \
  -H "Content-Type: application/json" \
  -d '{"product": "search", "input": {"text": "privacy-focused VPN no logs policy"}}')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ]; then
    test_result 0 "Extract intent - Privacy query"
else
    test_result 1 "Extract intent - Privacy query (HTTP $HTTP_CODE)"
fi

# Test case 3: Developer tools query
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/extract-intent" \
  -H "Content-Type: application/json" \
  -d '{"product": "search", "input": {"text": "open source developer tools free"}}')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ]; then
    test_result 0 "Extract intent - Developer tools query"
else
    test_result 1 "Extract intent - Developer tools query (HTTP $HTTP_CODE)"
fi

# Test 3: Ad Matching Workflow
test_header "TEST 3: Ad Matching (Ethical Ad Selection)"

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/match-ads" \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "declared": {"query": "best laptop for programming", "goal": "PURCHASE"},
      "inferred": {"useCases": [{"value": "programming"}], "ethicalSignals": []}
    },
    "adInventory": [
      {"id": "ad1", "title": "Programming Laptop", "description": "High-performance", "targetingConstraints": {"use_case": ["programming"]}, "forbiddenDimensions": [], "qualityScore": 0.9, "ethicalTags": ["quality"]}
    ]
  }')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ] && echo "$RESPONSE" | grep -q "matchedAds"; then
    test_result 0 "Ad matching endpoint"
else
    test_result 1 "Ad matching endpoint (HTTP $HTTP_CODE)"
fi

# Test 4: Service Recommendation Workflow
test_header "TEST 4: Service Recommendation"

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/recommend-services" \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "declared": {"query": "best laptop", "goal": "SEARCH"},
      "inferred": {"useCases": [], "ethicalSignals": []}
    },
    "availableServices": [
      {"id": "search", "name": "Search Engine", "supportedGoals": ["SEARCH", "LEARN"], "primaryUseCases": ["research"], "temporalPatterns": ["immediate"], "ethicalAlignment": ["privacy"], "description": "Privacy search"}
    ]
  }')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ] && echo "$RESPONSE" | grep -q "recommendations"; then
    test_result 0 "Service recommendation endpoint"
else
    test_result 1 "Service recommendation endpoint (HTTP $HTTP_CODE)"
fi

# Test 5: Campaign Management (Advertiser Workflow)
test_header "TEST 5: Campaign Management (Advertiser)"

# Create advertiser
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/advertisers" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Advertiser Co", "contact_email": "test@example.com"}')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
ADVERTISER_ID=$(echo "$RESPONSE" | head -n -1 | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
if [ "$HTTP_CODE" = "200" ] && [ -n "$ADVERTISER_ID" ]; then
    test_result 0 "Create advertiser (ID: $ADVERTISER_ID)"
else
    test_result 1 "Create advertiser (HTTP $HTTP_CODE)"
fi

# Create campaign
if [ -n "$ADVERTISER_ID" ]; then
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/campaigns" \
      -H "Content-Type: application/json" \
      -d "{
        \"advertiserId\": $ADVERTISER_ID,
        \"name\": \"Test Campaign $(date +%s)\",
        \"startDate\": \"$(date -u +%Y-%m-%d)\",
        \"endDate\": \"$(date -u -d '+30 days' +%Y-%m-%d 2>/dev/null || date -u -v+30d +%Y-%m-%d)\",
        \"budget\": 5000.0,
        \"dailyBudget\": 200.0,
        \"status\": \"active\"
      }")
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    CAMPAIGN_ID=$(echo "$RESPONSE" | head -n -1 | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
    if [ "$HTTP_CODE" = "200" ] && [ -n "$CAMPAIGN_ID" ]; then
        test_result 0 "Create campaign (ID: $CAMPAIGN_ID)"
    else
        test_result 1 "Create campaign (HTTP $HTTP_CODE)"
    fi
fi

# List campaigns
RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/campaigns?limit=10")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ] && echo "$RESPONSE" | grep -q "items"; then
    test_result 0 "List campaigns"
else
    test_result 1 "List campaigns (HTTP $HTTP_CODE)"
fi

# Test 6: Ad Group & Ad Management
test_header "TEST 6: Ad Group & Ad Management"

# Create ad group
if [ -n "$CAMPAIGN_ID" ]; then
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/adgroups" \
      -H "Content-Type: application/json" \
      -d "{
        \"campaignId\": $CAMPAIGN_ID,
        \"name\": \"Test Ad Group\",
        \"targetingSettings\": {\"device_type\": [\"mobile\"]},
        \"bidStrategy\": \"automatic\"
      }")
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    ADGROUP_ID=$(echo "$RESPONSE" | head -n -1 | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
    if [ "$HTTP_CODE" = "200" ] && [ -n "$ADGROUP_ID" ]; then
        test_result 0 "Create ad group (ID: $ADGROUP_ID)"
    else
        test_result 1 "Create ad group (HTTP $HTTP_CODE)"
    fi
fi

# Create ad
if [ -n "$ADGROUP_ID" ] && [ -n "$ADVERTISER_ID" ]; then
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/ads" \
      -H "Content-Type: application/json" \
      -d "{
        \"advertiserId\": $ADVERTISER_ID,
        \"adGroupId\": $ADGROUP_ID,
        \"title\": \"Test Ad Title\",
        \"description\": \"Test ad description for testing purposes\",
        \"url\": \"https://example.com/test\",
        \"targetingConstraints\": {\"category\": \"test\"},
        \"ethicalTags\": [\"test\"],
        \"qualityScore\": 0.8,
        \"creativeFormat\": \"banner\",
        \"bidAmount\": 1.50,
        \"status\": \"active\",
        \"approvalStatus\": \"pending\"
      }")
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    AD_ID=$(echo "$RESPONSE" | head -n -1 | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
    if [ "$HTTP_CODE" = "200" ] && [ -n "$AD_ID" ]; then
        test_result 0 "Create ad (ID: $AD_ID)"
    else
        test_result 1 "Create ad (HTTP $HTTP_CODE)"
    fi
fi

# List ads
RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/ads?limit=10")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ]; then
    test_result 0 "List ads"
else
    test_result 1 "List ads (HTTP $HTTP_CODE)"
fi

# Test 7: Creative Asset Management
test_header "TEST 7: Creative Asset Management"

# Upload creative
if [ -n "$AD_ID" ]; then
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/creatives" \
      -H "Content-Type: application/json" \
      -d "{
        \"adId\": $AD_ID,
        \"assetType\": \"image\",
        \"assetUrl\": \"https://example.com/assets/test-banner.jpg\",
        \"dimensions\": {\"width\": 728, \"height\": 90},
        \"checksum\": \"test123abc\"
      }")
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    CREATIVE_ID=$(echo "$RESPONSE" | head -n -1 | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
    if [ "$HTTP_CODE" = "200" ] && [ -n "$CREATIVE_ID" ]; then
        test_result 0 "Upload creative (ID: $CREATIVE_ID)"
    else
        test_result 1 "Upload creative (HTTP $HTTP_CODE)"
    fi
fi

# Test 8: Click & Conversion Tracking
test_header "TEST 8: Click & Conversion Tracking"

# Record click
if [ -n "$AD_ID" ]; then
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/click-tracking" \
      -H "Content-Type: application/json" \
      -d "{
        \"adId\": $AD_ID,
        \"sessionId\": \"test-session-$(date +%s)\",
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"userAgent\": \"Mozilla/5.0 Test Browser\",
        \"ipAddress\": \"192.168.1.100\",
        \"referrer\": \"https://example.com\"
      }")
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    CLICK_ID=$(echo "$RESPONSE" | head -n -1 | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
    if [ "$HTTP_CODE" = "200" ] && [ -n "$CLICK_ID" ]; then
        test_result 0 "Record click (ID: $CLICK_ID)"
    else
        test_result 1 "Record click (HTTP $HTTP_CODE)"
    fi
fi

# Record conversion
if [ -n "$CLICK_ID" ]; then
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/conversion-tracking" \
      -H "Content-Type: application/json" \
      -d "{
        \"clickId\": $CLICK_ID,
        \"conversionType\": \"purchase\",
        \"conversionValue\": 99.99,
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"status\": \"completed\"
      }")
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    CONVERSION_ID=$(echo "$RESPONSE" | head -n -1 | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
    if [ "$HTTP_CODE" = "200" ] && [ -n "$CONVERSION_ID" ]; then
        test_result 0 "Record conversion (ID: $CONVERSION_ID)"
    else
        test_result 1 "Record conversion (HTTP $HTTP_CODE)"
    fi
fi

# Test 9: Fraud Detection
test_header "TEST 9: Fraud Detection"

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/fraud-detection" \
  -H "Content-Type: application/json" \
  -d "{
    \"adId\": 1,
    \"eventType\": \"click\",
    \"ipAddress\": \"192.168.1.1\",
    \"userAgent\": \"Mozilla/5.0\",
    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
    \"metadata\": {\"clickCount\": 5, \"timeWindowSeconds\": 60}
  }")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ] && echo "$RESPONSE" | grep -q "fraud"; then
    test_result 0 "Fraud detection endpoint"
else
    test_result 1 "Fraud detection endpoint (HTTP $HTTP_CODE)"
fi

# Test 10: Privacy & Consent Management
test_header "TEST 10: Privacy & Consent Management"

# Record consent
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/consent/record" \
  -H "Content-Type: application/json" \
  -d "{\"userId\": \"test-user-123\", \"consentType\": \"analytics\", \"granted\": true}")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ] && echo "$RESPONSE" | grep -q "consent"; then
    test_result 0 "Record user consent"
else
    test_result 1 "Record user consent (HTTP $HTTP_CODE)"
fi

# Get consent status
RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/consent/test-user-123/analytics")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ]; then
    test_result 0 "Get consent status"
else
    test_result 1 "Get consent status (HTTP $HTTP_CODE)"
fi

# Test 11: Reporting & Analytics
test_header "TEST 11: Reporting & Analytics"

RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/reports/campaign-performance?startDate=$(date -u +%Y-%m-%d -d '-7 days' 2>/dev/null || date -u -v-7d +%Y-%m-%d)&endDate=$(date -u +%Y-%m-%d)")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ]; then
    test_result 0 "Campaign performance report"
else
    test_result 1 "Campaign performance report (HTTP $HTTP_CODE)"
fi

# Test 12: A/B Testing
test_header "TEST 12: A/B Testing"

# Create A/B test
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/abtests" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"Test Experiment $(date +%s)\",
    \"description\": \"Testing new ad format\",
    \"status\": \"active\",
    \"startDate\": \"$(date -u +%Y-%m-%d)\",
    \"endDate\": \"$(date -u -d '+30 days' +%Y-%m-%d 2>/dev/null || date -u -v+30d +%Y-%m-%d)\",
    \"variants\": [
      {\"name\": \"control\", \"weight\": 0.5, \"metadata\": {}},
      {\"name\": \"treatment\", \"weight\": 0.5, \"metadata\": {}}
    ]
  }")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
ABTEST_ID=$(echo "$RESPONSE" | head -n -1 | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
if [ "$HTTP_CODE" = "200" ] && [ -n "$ABTEST_ID" ]; then
    test_result 0 "Create A/B test (ID: $ABTEST_ID)"
else
    test_result 1 "Create A/B test (HTTP $HTTP_CODE)"
fi

# List A/B tests
RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/abtests?limit=10")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ]; then
    test_result 0 "List A/B tests"
else
    test_result 1 "List A/B tests (HTTP $HTTP_CODE)"
fi

# Test 13: URL Ranking (Privacy-Focused Search)
test_header "TEST 13: URL Ranking (Privacy Search)"

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/rank-urls" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "privacy-focused search engines",
    "urls": [
      {"url": "https://duckduckgo.com", "title": "DuckDuckGo", "description": "Privacy search"},
      {"url": "https://startpage.com", "title": "Startpage", "description": "Private search"}
    ],
    "options": {"minPrivacyScore": 0.7}
  }')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ] && echo "$RESPONSE" | grep -q "ranked_urls"; then
    test_result 0 "URL ranking endpoint"
else
    test_result 1 "URL ranking endpoint (HTTP $HTTP_CODE)"
fi

# Summary
echo ""
echo "============================================================"
echo "TEST SUMMARY"
echo "============================================================"
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "Total:  $((PASS + FAIL))"
echo "Success Rate: $((PASS * 100 / (PASS + FAIL)))%"
echo "============================================================"

if [ $FAIL -eq 0 ]; then
    echo "[SUCCESS] All tests passed!"
    exit 0
else
    echo "[WARNING] Some tests failed. Check output above."
    exit 1
fi
