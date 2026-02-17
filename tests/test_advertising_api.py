"""
Test script for the new advertising system API endpoints
"""

import json
import time
from datetime import datetime, timedelta

import requests

BASE_URL = "http://localhost:8000"


def test_campaign_endpoints():
    """Test campaign management endpoints"""
    print("Testing Campaign Management Endpoints...")

    # Create an advertiser first
    advertiser_data = {"name": "Test Advertiser", "contact_email": "test@example.com"}

    try:
        # Create advertiser (this would require an endpoint that we may need to add)
        # For now, we'll assume an advertiser exists with ID 1
        advertiser_id = 1

        # Create a campaign
        campaign_data = {
            "advertiser_id": advertiser_id,
            "name": "Test Campaign",
            "start_date": (datetime.now() + timedelta(days=1)).isoformat(),
            "end_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "budget": 1000.0,
            "daily_budget": 50.0,
            "status": "active",
        }

        response = requests.post(f"{BASE_URL}/campaigns", json=campaign_data)
        if response.status_code == 200:
            campaign = response.json()
            print(f"[OK] Created campaign: {campaign['name']} (ID: {campaign['id']})")
            campaign_id = campaign["id"]
        else:
            print(f"[ERROR] Failed to create campaign: {response.text}")
            return False

        # Get the campaign
        response = requests.get(f"{BASE_URL}/campaigns/{campaign_id}")
        if response.status_code == 200:
            print(f"[OK] Retrieved campaign: {response.json()['name']}")
        else:
            print(f"[ERROR] Failed to retrieve campaign: {response.text}")
            return False

        # Update the campaign
        update_data = {"name": "Updated Test Campaign", "budget": 1500.0}
        response = requests.put(f"{BASE_URL}/campaigns/{campaign_id}", json=update_data)
        if response.status_code == 200:
            print(f"[OK] Updated campaign: {response.json()['name']}")
        else:
            print(f"[ERROR] Failed to update campaign: {response.text}")
            return False

        # List campaigns
        response = requests.get(f"{BASE_URL}/campaigns")
        if response.status_code == 200:
            campaigns = response.json()
            print(f"[OK] Found {len(campaigns)} campaigns")
        else:
            print(f"[ERROR] Failed to list campaigns: {response.text}")
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Exception in campaign tests: {str(e)}")
        return False


def test_adgroup_endpoints():
    """Test ad group management endpoints"""
    print("\nTesting Ad Group Management Endpoints...")

    try:
        # Create a campaign first (or assume one exists)
        # For this test, we'll assume a campaign exists with ID 1
        campaign_id = 1

        # Create an ad group
        adgroup_data = {
            "campaign_id": campaign_id,
            "name": "Test Ad Group",
            "targeting_settings": {"device_type": ["mobile", "desktop"], "location": ["US", "CA"]},
            "bid_strategy": "manual",
        }

        response = requests.post(f"{BASE_URL}/adgroups", json=adgroup_data)
        if response.status_code == 200:
            adgroup = response.json()
            print(f"[OK] Created ad group: {adgroup['name']} (ID: {adgroup['id']})")
            adgroup_id = adgroup["id"]
        else:
            print(f"[ERROR] Failed to create ad group: {response.text}")
            return False

        # Get the ad group
        response = requests.get(f"{BASE_URL}/adgroups/{adgroup_id}")
        if response.status_code == 200:
            print(f"[OK] Retrieved ad group: {response.json()['name']}")
        else:
            print(f"[ERROR] Failed to retrieve ad group: {response.text}")
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Exception in ad group tests: {str(e)}")
        return False


def test_creative_endpoints():
    """Test creative management endpoints"""
    print("\nTesting Creative Management Endpoints...")

    try:
        # First, create an ad to associate with the creative
        # Create an advertiser first
        advertiser_data = {"name": "Test Advertiser for Creative", "contact_email": "creative@example.com"}

        response = requests.post(f"{BASE_URL}/advertisers", json=advertiser_data)
        if response.status_code != 200:
            print(f"[ERROR] Failed to create advertiser: {response.text}")
            return False

        advertiser = response.json()
        print(f"[OK] Created advertiser: {advertiser['name']} (ID: {advertiser['id']})")
        advertiser_id = advertiser["id"]

        # Create a campaign
        campaign_data = {
            "advertiser_id": advertiser_id,
            "name": "Test Campaign for Creative",
            "start_date": (datetime.now() + timedelta(days=1)).isoformat(),
            "end_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "budget": 1000.0,
            "daily_budget": 50.0,
            "status": "active",
        }

        response = requests.post(f"{BASE_URL}/campaigns", json=campaign_data)
        if response.status_code != 200:
            print(f"[ERROR] Failed to create campaign: {response.text}")
            return False

        campaign = response.json()
        print(f"[OK] Created campaign: {campaign['name']} (ID: {campaign['id']})")
        campaign_id = campaign["id"]

        # Create an ad group
        adgroup_data = {
            "campaign_id": campaign_id,
            "name": "Test Ad Group for Creative",
            "targeting_settings": {"device_type": ["mobile", "desktop"], "location": ["US", "CA"]},
            "bid_strategy": "manual",
        }

        response = requests.post(f"{BASE_URL}/adgroups", json=adgroup_data)
        if response.status_code != 200:
            print(f"[ERROR] Failed to create ad group: {response.text}")
            return False

        adgroup = response.json()
        print(f"[OK] Created ad group: {adgroup['name']} (ID: {adgroup['id']})")
        adgroup_id = adgroup["id"]

        # Create an ad
        ad_data = {
            "advertiser_id": advertiser_id,
            "ad_group_id": adgroup_id,
            "title": "Test Ad for Creative",
            "description": "Test ad description",
            "url": "https://example.com",
            "targeting_constraints": {},
            "ethical_tags": ["privacy"],
            "quality_score": 0.8,
            "creative_format": "banner",
            "bid_amount": 1.0,
            "status": "active",
            "approval_status": "approved",
        }

        response = requests.post(f"{BASE_URL}/ads", json=ad_data)
        if response.status_code != 200:
            print(f"[ERROR] Failed to create ad: {response.text}")
            return False

        ad = response.json()
        print(f"[OK] Created ad: {ad['title']} (ID: {ad['id']})")
        ad_id = ad["id"]

        # Now create a creative asset
        creative_data = {
            "ad_id": ad_id,
            "asset_type": "image",
            "asset_url": "https://example.com/creative.jpg",
            "dimensions": {"width": 300, "height": 250},
            "checksum": "abc123",
        }

        response = requests.post(f"{BASE_URL}/creatives", json=creative_data)
        if response.status_code == 200:
            creative = response.json()
            print(f"[OK] Created creative: {creative['asset_type']} (ID: {creative['id']})")
            creative_id = creative["id"]
        else:
            print(f"[ERROR] Failed to create creative: {response.text}")
            return False

        # Get the creative
        response = requests.get(f"{BASE_URL}/creatives/{creative_id}")
        if response.status_code == 200:
            print(f"[OK] Retrieved creative: {response.json()['asset_type']}")
        else:
            print(f"[ERROR] Failed to retrieve creative: {response.text}")
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Exception in creative tests: {str(e)}")
        return False


def test_reporting_endpoints():
    """Test reporting endpoints"""
    print("\nTesting Reporting Endpoints...")

    try:
        # Get campaign performance report
        response = requests.get(f"{BASE_URL}/reports/campaign-performance")
        if response.status_code == 200:
            reports = response.json()
            print(f"[OK] Retrieved {len(reports)} campaign performance reports")
        else:
            print(f"[ERROR] Failed to retrieve campaign performance reports: {response.text}")
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Exception in reporting tests: {str(e)}")
        return False


def test_advanced_ad_matching():
    """Test advanced ad matching with campaign context"""
    print("\nTesting Advanced Ad Matching...")

    try:
        # First, extract an intent using the working endpoint
        intent_payload = {
            "product": "search",
            "input": {"text": "privacy focused email service"},
            "context": {"sessionId": "test-session", "userLocale": "en-US"},
        }

        response = requests.post(f"{BASE_URL}/extract-intent", json=intent_payload)
        if response.status_code != 200:
            print(f"[ERROR] Failed to extract intent: {response.text}")
            return False

        intent_data = response.json()["intent"]

        # Advanced ad matching request with campaign context
        request_data = {
            "intent": intent_data,
            "ad_inventory": [],
            "campaign_context": {"campaign_ids": [1], "budget_constraint": True},
        }

        response = requests.post(f"{BASE_URL}/match-ads-advanced", json=request_data)
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Advanced ad matching returned {len(result['matched_ads'])} ads")
        else:
            print(f"[ERROR] Failed advanced ad matching: {response.text}")
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Exception in advanced ad matching tests: {str(e)}")
        return False


def run_all_tests():
    """Run all tests"""
    print("Starting Advertising System API Tests...\n")

    success = True

    success &= test_campaign_endpoints()
    success &= test_adgroup_endpoints()
    success &= test_creative_endpoints()
    success &= test_reporting_endpoints()
    success &= test_advanced_ad_matching()

    print(f"\n{'='*50}")
    if success:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print(f"{'='*50}")

    return success


if __name__ == "__main__":
    run_all_tests()
