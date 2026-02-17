#!/usr/bin/env python3
"""
API Integration Test Script for Intent Engine Docker Container
Tests all major endpoints of the running API
"""

import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://localhost:8000"

# Test results tracking
tests_passed = 0
tests_failed = 0
test_results = []

def call_api_endpoint(name, method, endpoint, payload=None, expected_status=200):
    """Test an API endpoint"""
    global tests_passed, tests_failed
    
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=payload, timeout=10)
        else:
            response = requests.request(method, url, json=payload, timeout=10)
        
        success = response.status_code == expected_status
        
        if success:
            tests_passed += 1
            test_results.append((name, "PASS", response.status_code))
            return True, response
        else:
            tests_failed += 1
            test_results.append((name, f"FAIL (Expected {expected_status}, got {response.status_code})", None))
            return False, response
            
    except Exception as e:
        tests_failed += 1
        test_results.append((name, f"ERROR: {str(e)}", None))
        return False, None

def print_results():
    """Print test results"""
    print("\n" + "="*80)
    print("API TEST RESULTS")
    print("="*80)
    for name, status, code in test_results:
        status_str = str(status)
        print(f"{name:<50} {status_str}")
    print("="*80)
    print(f"\nTotal: {tests_passed + tests_failed} | Passed: {tests_passed} | Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{tests_failed} test(s) failed")
        return 1

def main():
    print("Testing Intent Engine API (Docker)")
    print(f"Base URL: {BASE_URL}")
    print("="*80)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check (GET /)")
    success, response = call_api_endpoint("Health Check", "GET", "/")
    if success:
        print(f"   Status: {response.json()}")
    
    # Test 2: API Status
    print("\n2. Testing Status Endpoint (GET /status)")
    success, response = call_api_endpoint("Status Check", "GET", "/status")
    if success:
        print(f"   Status: {response.json()}")
    
    # Test 3: Metrics Endpoint
    print("\n3. Testing Metrics Endpoint (GET /metrics)")
    success, response = call_api_endpoint("Metrics", "GET", "/metrics")
    if success:
        print(f"   Response length: {len(response.text)} characters")
    
    # Test 4: Intent Extraction
    print("\n4. Testing Intent Extraction (POST /extract-intent)")
    intent_payload = {
        "query": "How to setup E2E encrypted email on Android, no big tech solutions",
        "context": {
            "timestamp": "2026-02-16T10:00:00Z",
            "sessionId": "test-session-123",
            "userLocale": "en-US"
        }
    }
    success, response = call_api_endpoint("Intent Extraction", "POST", "/extract-intent", intent_payload)
    if success:
        data = response.json()
        intent = data.get('intent', {})
        print(f"   Intent ID: {intent.get('intentId', 'N/A')}")
        declared = intent.get('declared', {})
        print(f"   Declared Goal: {declared.get('goal', 'N/A')}")
    
    # Test 5: URL Ranking
    print("\n5. Testing URL Ranking (POST /rank-urls)")
    url_ranking_payload = {
        "query": "privacy email android",
        "urls": [
            "https://proton.me/mail",
            "https://tutanota.com",
            "https://mail.google.com",
            "https://outlook.live.com"
        ],
        "weights": {
            "relevance": 0.4,
            "privacy": 0.35,
            "quality": 0.25
        },
        "exclude_big_tech": True,
        "min_privacy_score": 0.5
    }
    success, response = call_api_endpoint("URL Ranking", "POST", "/rank-urls", url_ranking_payload)
    if success:
        data = response.json()
        ranked_urls = data.get('ranked_urls', [])
        print(f"   Ranked {len(ranked_urls)} URLs")
        if ranked_urls:
            top_result = ranked_urls[0]
            print(f"   Top result: {top_result.get('url', 'N/A')} (Score: {top_result.get('final_score', 0):.3f})")
    
    # Test 6: Service Recommendations
    print("\n6. Testing Service Recommendations (POST /recommend-services)")
    service_payload = {
        "intent": {
            "intentId": "test-intent-456",
            "context": {
                "product": "search",
                "timestamp": "2026-02-16T10:00:00Z",
                "sessionId": "test-session-456",
                "userLocale": "en-US"
            },
            "declared": {
                "query": "How to setup E2E encrypted email",
                "goal": "learn",
                "constraints": [
                    {"type": "inclusion", "dimension": "platform", "value": "Android", "hardFilter": True}
                ],
                "negativePreferences": ["no big tech"],
                "skillLevel": "intermediate"
            },
            "inferred": {
                "useCases": ["learning"],
                "ethicalSignals": [
                    {"dimension": "privacy", "preference": "privacy-first"}
                ]
            }
        },
        "available_services": [
            {
                "id": "protonmail",
                "name": "ProtonMail",
                "description": "Encrypted email service",
                "type": "API",
                "ethicalTags": ["privacy", "encryption"],
                "features": ["E2E encryption", "Open source"],
                "qualityScore": 0.9
            },
            {
                "id": "gmail",
                "name": "Gmail",
                "description": "Google email service",
                "type": "API",
                "ethicalTags": ["ad_supported"],
                "features": ["15GB storage"],
                "qualityScore": 0.8
            }
        ]
    }
    success, response = call_api_endpoint("Service Recommendations", "POST", "/recommend-services", service_payload)
    if success:
        data = response.json()
        recommendations = data.get('recommendations', [])
        print(f"   Recommended {len(recommendations)} services")
        if recommendations:
            for rec in recommendations[:3]:
                svc = rec.get('service', {})
                print(f"   - {svc.get('name', 'N/A')}: {rec.get('relevanceScore', 0):.3f}")
    
    # Test 7: Ad Matching
    print("\n7. Testing Ad Matching (POST /match-ads)")
    from datetime import datetime, timedelta
    future_time = (datetime.utcnow() + timedelta(hours=8)).isoformat() + "Z"
    ad_matching_payload = {
        "intent": {
            "intentId": "test-intent-789",
            "context": {
                "product": "search",
                "timestamp": "2026-02-16T10:00:00Z",
                "sessionId": "test-session-789",
                "userLocale": "en-US"
            },
            "declared": {
                "query": "privacy VPN service",
                "goal": "learn",
                "constraints": [],
                "skillLevel": "intermediate"
            },
            "inferred": {
                "useCases": ["learning"],
                "ethicalSignals": [
                    {"dimension": "privacy", "preference": "privacy-first"}
                ]
            },
            "expiresAt": future_time
        },
        "ad_inventory": [
            {
                "id": "ad_privacy_vpn_1",
                "title": "Privacy-Focused VPN",
                "description": "No-logs VPN service with strong encryption",
                "targetingConstraints": {"category": ["privacy", "security"]},
                "forbiddenDimensions": [],
                "qualityScore": 0.9,
                "ethicalTags": ["privacy", "no_logs"]
            }
        ],
        "config": {
            "topK": 5,
            "minThreshold": 0.4
        }
    }
    success, response = call_api_endpoint("Ad Matching", "POST", "/match-ads", ad_matching_payload)
    if success:
        data = response.json()
        matched_ads = data.get('matched_ads', [])
        print(f"   Matched {len(matched_ads)} ads")
        if matched_ads:
            print(f"   Top ad: {matched_ads[0]['ad']['title']}")
    
    # Test 8: Get Campaigns (empty list expected)
    print("\n8. Testing Get Campaigns (GET /campaigns)")
    success, response = call_api_endpoint("Get Campaigns", "GET", "/campaigns")
    if success:
        data = response.json()
        print(f"   Found {len(data)} campaigns")
    
    # Test 9: Get Ads (empty list expected)
    print("\n9. Testing Get Ads (GET /ads)")
    success, response = call_api_endpoint("Get Ads", "GET", "/ads")
    if success:
        data = response.json()
        print(f"   Found {len(data)} ads")
    
    # Test 10: Consent Summary
    print("\n10. Testing Consent Summary (GET /consent-summary)")
    success, response = call_api_endpoint("Consent Summary", "GET", "/consent-summary")
    if success:
        data = response.json()
        print(f"   Total consents: {data.get('totalConsents', 0)}")
    
    # Test 11: Audit Stats
    print("\n11. Testing Audit Stats (GET /audit-stats)")
    success, response = call_api_endpoint("Audit Stats", "GET", "/audit-stats")
    if success:
        data = response.json()
        print(f"   Total events: {data.get('totalEvents', 0)}")
    
    # Print final results
    return print_results()

if __name__ == "__main__":
    sys.exit(main())
