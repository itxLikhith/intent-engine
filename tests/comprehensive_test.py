"""
Comprehensive test script for the Intent Engine with Advertising System
"""


import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        print("[OK] Health endpoint working")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        raise e


def test_campaigns():
    """Test campaign management endpoints"""
    print("\nTesting campaign management endpoints...")

    # Test GET campaigns
    response = requests.get(f"{BASE_URL}/campaigns")
    assert response.status_code == 200
    campaigns = response.json()
    assert isinstance(campaigns, list)
    print(f"[OK] Retrieved {len(campaigns)} campaigns")

    if campaigns:
        campaign_id = campaigns[0]["id"]
        # Test GET specific campaign
        response = requests.get(f"{BASE_URL}/campaigns/{campaign_id}")
        assert response.status_code == 200
        campaign = response.json()
        assert campaign["id"] == campaign_id
        print(f"[OK] Retrieved campaign {campaign_id}")


def test_ad_groups():
    """Test ad group management endpoints"""
    print("\nTesting ad group management endpoints...")

    # Test GET ad groups
    response = requests.get(f"{BASE_URL}/adgroups")
    assert response.status_code == 200
    ad_groups = response.json()
    assert isinstance(ad_groups, list)
    print(f"[OK] Retrieved {len(ad_groups)} ad groups")

    if ad_groups:
        ad_group_id = ad_groups[0]["id"]
        # Test GET specific ad group
        response = requests.get(f"{BASE_URL}/adgroups/{ad_group_id}")
        assert response.status_code == 200
        ad_group = response.json()
        assert ad_group["id"] == ad_group_id
        print(f"[OK] Retrieved ad group {ad_group_id}")


def test_creatives():
    """Test creative management endpoints"""
    print("\nTesting creative management endpoints...")

    # Test GET specific creative
    response = requests.get(f"{BASE_URL}/creatives/1")
    if response.status_code == 404:
        print("[OK] Creative 1 not found, which is expected in a clean DB")
    elif response.status_code == 200:
        creative = response.json()
        assert creative["id"] == 1
        print(f"[OK] Retrieved creative {creative['id']}")
    else:
        assert False, f"Unexpected status code {response.status_code}"


def test_reporting():
    """Test reporting endpoints"""
    print("\nTesting reporting endpoints...")

    # Test campaign performance report
    response = requests.get(f"{BASE_URL}/reports/campaign-performance")
    assert response.status_code == 200
    reports = response.json()
    assert isinstance(reports, list)
    print(f"[OK] Retrieved {len(reports)} campaign performance reports")


def test_intent_extraction():
    """Test intent extraction"""
    print("\nTesting intent extraction...")

    payload = {
        "product": "search",
        "input": {"text": "How to set up encrypted email?"},
        "context": {"sessionId": "test-session", "userLocale": "en-US"},
    }

    response = requests.post(f"{BASE_URL}/extract-intent", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "intent" in result
    print("[OK] Intent extraction working")


def test_ad_matching():
    """Test ad matching"""
    print("\nTesting ad matching...")

    # First extract an intent to use for matching
    intent_payload = {
        "product": "search",
        "input": {"text": "privacy focused email service"},
        "context": {"sessionId": "test-session", "userLocale": "en-US"},
    }

    response = requests.post(f"{BASE_URL}/extract-intent", json=intent_payload)
    assert response.status_code == 200
    intent_data = response.json()

    # Now test ad matching with the extracted intent
    ad_match_payload = {"intent": intent_data["intent"], "ad_inventory": [], "config": {"minThreshold": 0.4, "topK": 5}}

    response = requests.post(f"{BASE_URL}/match-ads", json=ad_match_payload)
    assert response.status_code == 200
    result = response.json()
    assert "matched_ads" in result
    print(f"[OK] Ad matching working, found {len(result['matched_ads'])} ads")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running comprehensive tests for Intent Engine with Advertising System")
    print("=" * 60)

    try:
        test_health()
        test_campaigns()
        test_ad_groups()
        test_creatives()
        test_reporting()
        test_intent_extraction()
        test_ad_matching()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("The Intent Engine with Advertising System is working correctly!")
        print("=" * 60)

        return True
    except AssertionError as e:
        print(f"\nTEST FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    run_all_tests()
