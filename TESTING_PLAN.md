# Intent Engine API - Comprehensive Testing Plan

## Overview

This document outlines a detailed plan to test all currently failing/500-error endpoints by properly setting up required data dependencies and testing complete user workflows.

---

## Problem Analysis

| Endpoint Category | Current Status | Root Cause |
|------------------|----------------|------------|
| `/creatives` | 500 Error | Missing proper asset upload handling |
| `/click-tracking` | 500 Error | Missing ad/click data relationships |
| `/conversion-tracking` | 500 Error | Depends on click-tracking |
| `/fraud-detection` | 500 Error | Missing scan configuration |
| `/analytics/*` | 500 Error | Requires metrics data |
| `/reports/*` | 500 Error | Requires campaign/ad metrics |
| `/ab-tests` | 500 Error | Missing test variant data |

---

## Phase 1: Data Setup (Prerequisites)

### 1.1 Create Complete Data Hierarchy

```
Advertiser → Campaign → AdGroup → Ad → CreativeAsset
                                    ↓
                              ClickTracking
                                    ↓
                          ConversionTracking
                                    ↓
                               AdMetric
```

### 1.2 Required Data Flow

```python
# Step 1: Create advertiser
POST /advertisers
→ advertiser_id

# Step 2: Create campaign
POST /campaigns {advertiser_id}
→ campaign_id

# Step 3: Create ad group
POST /adgroups {campaign_id}
→ adgroup_id

# Step 4: Create ad
POST /ads {advertiser_id, adgroup_id}
→ ad_id

# Step 5: Create creative
POST /creatives {ad_id}
→ creative_id

# Step 6: Record click
POST /click-tracking {ad_id}
→ click_id

# Step 7: Record conversion
POST /conversion-tracking {click_id}
→ conversion_id

# Step 8: Create metrics
(Insert directly via DB or trigger via events)
```

---

## Phase 2: Endpoint-Specific Test Plans

### 2.1 Creative Asset Management (`/creatives`)

**Current Issue:** 500 Internal Server Error

**Test Plan:**

```bash
# 1. Create advertiser
curl -X POST http://localhost:8000/advertisers \
  -H "Content-Type: application/json" \
  -d '{"name": "Creative Test Corp", "contact_email": "test@creative.com"}'
# Save: advertiser_id

# 2. Create campaign
curl -X POST http://localhost:8000/campaigns \
  -H "Content-Type: application/json" \
  -d '{
    "advertiser_id": <advertiser_id>,
    "name": "Creative Test Campaign",
    "start_date": "2026-02-18T00:00:00Z",
    "end_date": "2026-03-18T00:00:00Z",
    "budget": 5000,
    "daily_budget": 150,
    "status": "active"
  }'
# Save: campaign_id

# 3. Create ad group
curl -X POST http://localhost:8000/adgroups \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": <campaign_id>,
    "name": "Creative Ad Group",
    "targeting_settings": {"device": ["desktop"]},
    "bid_strategy": "automatic"
  }'
# Save: adgroup_id

# 4. Create ad
curl -X POST http://localhost:8000/ads \
  -H "Content-Type: application/json" \
  -d '{
    "advertiser_id": <advertiser_id>,
    "ad_group_id": <adgroup_id>,
    "title": "Test Ad for Creative",
    "description": "Testing creative upload",
    "url": "https://example.com/test",
    "targeting_constraints": {},
    "ethical_tags": ["test"],
    "quality_score": 0.8,
    "creative_format": "banner",
    "bid_amount": 1.5,
    "status": "active",
    "approval_status": "approved"
  }'
# Save: ad_id

# 5. Upload creative (CORRECT FIELD NAMES)
curl -X POST http://localhost:8000/creatives \
  -H "Content-Type: application/json" \
  -d '{
    "ad_id": <ad_id>,
    "asset_type": "image",
    "asset_url": "https://example.com/banner.jpg",
    "dimensions": {"width": 728, "height": 90},
    "checksum": "abc123"
  }'
# Expected: 200 OK with creative_id
```

**Success Criteria:**
- [ ] Creative created successfully
- [ ] Creative linked to ad
- [ ] GET /creatives/{id} returns creative data
- [ ] GET /creatives?ad_id={ad_id} lists creatives

---

### 2.2 Click Tracking (`/click-tracking`)

**Current Issue:** 500 Internal Server Error

**Test Plan:**

```bash
# Prerequisites: ad_id from Phase 2.1

# 1. Record click (CORRECT FIELD NAMES - snake_case)
curl -X POST http://localhost:8000/click-tracking \
  -H "Content-Type: application/json" \
  -d '{
    "ad_id": <ad_id>,
    "session_id": "test-session-'"$(date +%s)"'",
    "timestamp": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "ip_address": "192.168.1.100",
    "referrer": "https://google.com/search?q=vpn"
  }'
# Expected: 200 OK with click_id

# 2. List clicks for ad
curl "http://localhost:8000/click-tracking?ad_id=<ad_id>&limit=10"
# Expected: 200 OK with clicks array

# 3. Get specific click
curl "http://localhost:8000/click-tracking/<click_id>"
# Expected: 200 OK with click details
```

**Success Criteria:**
- [ ] Click recorded successfully
- [ ] Click linked to ad
- [ ] Click list endpoint works
- [ ] Individual click retrieval works

---

### 2.3 Conversion Tracking (`/conversion-tracking`)

**Current Issue:** Depends on click-tracking

**Test Plan:**

```bash
# Prerequisites: click_id from Phase 2.2

# 1. Record conversion (CORRECT FIELD NAMES)
curl -X POST http://localhost:8000/conversion-tracking \
  -H "Content-Type: application/json" \
  -d '{
    "click_id": <click_id>,
    "conversion_type": "purchase",
    "conversion_value": 99.99,
    "timestamp": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",
    "status": "completed",
    "metadata": {"product_id": "vpn-premium", "quantity": 1}
  }'
# Expected: 200 OK with conversion_id

# 2. List conversions
curl "http://localhost:8000/conversion-tracking?click_id=<click_id>"
# Expected: 200 OK with conversions array

# 3. Get specific conversion
curl "http://localhost:8000/conversion-tracking/<conversion_id>"
# Expected: 200 OK with conversion details
```

**Success Criteria:**
- [ ] Conversion recorded successfully
- [ ] Conversion linked to click
- [ ] Conversion list endpoint works
- [ ] Individual conversion retrieval works

---

### 2.4 Fraud Detection (`/fraud-detection`)

**Current Issue:** 500 Internal Server Error

**Test Plan:**

```bash
# Prerequisites: ad_id, click_id from previous phases

# 1. Report fraud event (CORRECT FIELD NAMES)
curl -X POST http://localhost:8000/fraud-detection \
  -H "Content-Type: application/json" \
  -d '{
    "ad_id": <ad_id>,
    "event_type": "click",
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0",
    "timestamp": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",
    "metadata": {
      "click_count": 50,
      "time_window_seconds": 60,
      "reason": "suspicious_velocity"
    }
  }'
# Expected: 200 OK with fraud_id and analysis

# 2. List fraud events
curl "http://localhost:8000/fraud-detection?ad_id=<ad_id>&event_type=click&limit=10"
# Expected: 200 OK with events array

# 3. Get fraud stats (CORRECT ENDPOINT)
curl "http://localhost:8000/fraud/analyze-click?ad_id=<ad_id>&ip_address=192.168.1.1"
# Expected: 200 OK with fraud analysis

# 4. Run fraud scan
curl -X POST http://localhost:8000/fraud/run-scan \
  -H "Content-Type: application/json" \
  -d '{
    "scan_type": "clicks",
    "time_window_hours": 24,
    "threshold": 0.7
  }'
# Expected: 200 OK with scan results
```

**Success Criteria:**
- [ ] Fraud event recorded
- [ ] Fraud list endpoint works
- [ ] Fraud analysis works
- [ ] Fraud scan executes successfully

---

### 2.5 Analytics Endpoints (`/analytics/*`)

**Current Issue:** 500 Error - Requires metrics data

**Test Plan:**

```bash
# Step 1: Seed metrics data directly via database
docker exec intent-engine-intent-engine-api-1 python << 'EOF'
from database import db_manager
from database import AdMetric
from datetime import datetime, timedelta

db = next(db_manager.get_db())

# Create metrics for existing ads
for ad_id in [1, 2, 3, 4]:
    for i in range(7):
        metric_date = (datetime.now() - timedelta(days=i)).date()
        metric = AdMetric(
            ad_id=ad_id,
            date=metric_date,
            intent_goal="PURCHASE",
            intent_use_case="shopping",
            impression_count=1000 + (i * 100),
            click_count=50 + (i * 5),
            conversion_count=5 + i,
            ctr=0.05 + (i * 0.005),
            cpc=1.50 - (i * 0.1),
            roas=3.5 - (i * 0.2),
            engagement_rate=0.12 + (i * 0.01),
            expires_at=datetime.now() + timedelta(days=90)
        )
        db.add(metric)

db.commit()
db.close()
print("Metrics seeded successfully")
EOF

# Step 2: Test analytics endpoints

# Top ads by CTR
curl "http://localhost:8000/analytics/top-ads?limit=10&metric=ctr"
# Expected: 200 OK with ads array sorted by CTR

# Top ads by conversions
curl "http://localhost:8000/analytics/top-ads?limit=10&metric=conversions"
# Expected: 200 OK with ads array sorted by conversions

# Trend data for clicks
curl "http://localhost:8000/analytics/trends/clicks?start_date=2026-02-11&end_date=2026-02-18"
# Expected: 200 OK with trend data points

# Trend data for impressions
curl "http://localhost:8000/analytics/trends/impressions?start_date=2026-02-11&end_date=2026-02-18"
# Expected: 200 OK with trend data points

# Campaign ROI
curl "http://localhost:8000/analytics/campaign-roi/1"
# Expected: 200 OK with ROI metrics

# Conversion attribution
curl "http://localhost:8000/analytics/attribution/<conversion_id>"
# Expected: 200 OK with attribution data
```

**Success Criteria:**
- [ ] Metrics data seeded
- [ ] Top ads endpoint works
- [ ] Trends endpoint works
- [ ] Campaign ROI endpoint works
- [ ] Attribution endpoint works

---

### 2.6 Reporting Endpoints (`/reports/*`)

**Current Issue:** 500 Error - Requires metrics data

**Test Plan:**

```bash
# Prerequisites: Metrics from Phase 2.5

# 1. Campaign performance report
curl "http://localhost:8000/reports/campaign-performance?start_date=2026-02-11&end_date=2026-02-18"
# Expected: 200 OK with performance data

# 2. Campaign performance with filters
curl "http://localhost:8000/reports/campaign-performance?start_date=2026-02-11&end_date=2026-02-18&campaign_id=1"
# Expected: 200 OK with filtered performance data

# 3. Ad performance report (if available)
curl "http://localhost:8000/reports/ad-performance?start_date=2026-02-11&end_date=2026-02-18&ad_id=1"
# Expected: 200 OK with ad performance data
```

**Success Criteria:**
- [ ] Campaign performance report works
- [ ] Report filtering works
- [ ] Data aggregation is correct

---

### 2.7 A/B Testing (`/ab-tests`)

**Current Issue:** 500 Error - Endpoint path is `/ab-tests` not `/abtests`

**Test Plan:**

```bash
# 1. Create A/B test (CORRECT ENDPOINT PATH)
curl -X POST http://localhost:8000/ab-tests \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ad Format Test",
    "description": "Testing banner vs native ad formats",
    "status": "active",
    "start_date": "2026-02-18T00:00:00Z",
    "end_date": "2026-03-18T00:00:00Z",
    "variants": [
      {
        "name": "control",
        "weight": 0.5,
        "metadata": {"format": "banner", "color": "blue"}
      },
      {
        "name": "treatment",
        "weight": 0.5,
        "metadata": {"format": "native", "color": "green"}
      }
    ]
  }'
# Expected: 200 OK with test_id

# 2. List A/B tests
curl "http://localhost:8000/ab-tests?limit=10&status=active"
# Expected: 200 OK with tests array

# 3. Get specific test
curl "http://localhost:8000/ab-tests/<test_id>"
# Expected: 200 OK with test details

# 4. Get test variants
curl "http://localhost:8000/ab-tests/<test_id>/variants"
# Expected: 200 OK with variants array

# 5. Get test results
curl "http://localhost:8000/ab-tests/<test_id>/results"
# Expected: 200 OK with statistical results

# 6. Assign user to variant
curl -X POST "http://localhost:8000/ab-tests/<test_id>/assign?user_id=user-123"
# Expected: 200 OK with assigned variant

# 7. Start/pause test
curl -X POST "http://localhost:8000/ab-tests/<test_id>/start"
curl -X POST "http://localhost:8000/ab-tests/<test_id>/pause"
# Expected: 200 OK with status update

# 8. Complete test
curl -X POST "http://localhost:8000/ab-tests/<test_id>/complete"
# Expected: 200 OK with final results
```

**Success Criteria:**
- [ ] A/B test created successfully
- [ ] Test variants created
- [ ] User assignment works
- [ ] Test results calculated
- [ ] Test lifecycle (start/pause/complete) works

---

## Phase 3: Integration Test Script

Create `test_complete_workflow.py`:

```python
#!/usr/bin/env python3
"""
Complete API Workflow Test
Tests all endpoints in proper sequence with data dependencies
"""

import requests
from datetime import datetime, timedelta, timezone

BASE_URL = "http://localhost:8000"

def test_complete_workflow():
    results = {"passed": 0, "failed": 0, "details": []}
    
    # 1. Create advertiser
    r = requests.post(f"{BASE_URL}/advertisers", json={
        "name": f"Workflow Test {datetime.now().strftime('%H%M%S')}",
        "contact_email": "workflow@test.com"
    })
    if r.status_code == 200:
        results["passed"] += 1
        advertiser_id = r.json()["id"]
    else:
        results["failed"] += 1
        results["details"].append(("Create advertiser", r.status_code, r.text))
        return results
    
    # 2. Create campaign
    now = datetime.now(timezone.utc)
    r = requests.post(f"{BASE_URL}/campaigns", json={
        "advertiser_id": advertiser_id,
        "name": "Workflow Campaign",
        "start_date": now.isoformat(),
        "end_date": (now + timedelta(days=30)).isoformat(),
        "budget": 5000.0,
        "daily_budget": 150.0,
        "status": "active"
    })
    if r.status_code == 200:
        results["passed"] += 1
        campaign_id = r.json()["id"]
    else:
        results["failed"] += 1
        results["details"].append(("Create campaign", r.status_code, r.text))
        return results
    
    # 3. Create ad group
    r = requests.post(f"{BASE_URL}/adgroups", json={
        "campaign_id": campaign_id,
        "name": "Workflow AdGroup",
        "targeting_settings": {"device": ["desktop"]},
        "bid_strategy": "automatic"
    })
    if r.status_code == 200:
        results["passed"] += 1
        adgroup_id = r.json()["id"]
    else:
        results["failed"] += 1
        results["details"].append(("Create ad group", r.status_code, r.text))
        return results
    
    # 4. Create ad
    r = requests.post(f"{BASE_URL}/ads", json={
        "advertiser_id": advertiser_id,
        "ad_group_id": adgroup_id,
        "title": "Workflow Test Ad",
        "description": "Testing complete workflow",
        "url": "https://example.com/workflow",
        "targeting_constraints": {},
        "ethical_tags": ["test"],
        "quality_score": 0.85,
        "creative_format": "banner",
        "bid_amount": 2.0,
        "status": "active",
        "approval_status": "approved"
    })
    if r.status_code == 200:
        results["passed"] += 1
        ad_id = r.json()["id"]
    else:
        results["failed"] += 1
        results["details"].append(("Create ad", r.status_code, r.text))
        return results
    
    # 5. Create creative
    r = requests.post(f"{BASE_URL}/creatives", json={
        "ad_id": ad_id,
        "asset_type": "image",
        "asset_url": "https://example.com/banner.jpg",
        "dimensions": {"width": 728, "height": 90},
        "checksum": "workflow123"
    })
    if r.status_code == 200:
        results["passed"] += 1
        creative_id = r.json()["id"]
    else:
        results["failed"] += 1
        results["details"].append(("Create creative", r.status_code, r.text))
    
    # 6. Record click
    r = requests.post(f"{BASE_URL}/click-tracking", json={
        "ad_id": ad_id,
        "session_id": f"workflow-{datetime.now().strftime('%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_agent": "Workflow Test Browser",
        "ip_address": "192.168.1.100",
        "referrer": "https://google.com"
    })
    if r.status_code == 200:
        results["passed"] += 1
        click_id = r.json()["id"]
    else:
        results["failed"] += 1
        results["details"].append(("Record click", r.status_code, r.text))
        return results
    
    # 7. Record conversion
    r = requests.post(f"{BASE_URL}/conversion-tracking", json={
        "click_id": click_id,
        "conversion_type": "purchase",
        "conversion_value": 99.99,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "completed"
    })
    if r.status_code == 200:
        results["passed"] += 1
        conversion_id = r.json()["id"]
    else:
        results["failed"] += 1
        results["details"].append(("Record conversion", r.status_code, r.text))
    
    # 8. Report fraud
    r = requests.post(f"{BASE_URL}/fraud-detection", json={
        "ad_id": ad_id,
        "event_type": "click",
        "ip_address": "192.168.1.1",
        "user_agent": "Fraud Test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {"click_count": 50, "time_window_seconds": 60}
    })
    if r.status_code == 200:
        results["passed"] += 1
    else:
        results["failed"] += 1
        results["details"].append(("Report fraud", r.status_code, r.text))
    
    # 9. Create A/B test
    r = requests.post(f"{BASE_URL}/ab-tests", json={
        "name": f"Workflow AB Test {datetime.now().strftime('%H%M%S')}",
        "description": "Workflow test",
        "status": "active",
        "start_date": now.isoformat(),
        "end_date": (now + timedelta(days=30)).isoformat(),
        "variants": [
            {"name": "control", "weight": 0.5, "metadata": {}},
            {"name": "treatment", "weight": 0.5, "metadata": {}}
        ]
    })
    if r.status_code == 200:
        results["passed"] += 1
        abtest_id = r.json()["id"]
    else:
        results["failed"] += 1
        results["details"].append(("Create A/B test", r.status_code, r.text))
        return results
    
    # 10. Get A/B test results
    r = requests.get(f"{BASE_URL}/ab-tests/{abtest_id}/results")
    if r.status_code == 200:
        results["passed"] += 1
    else:
        results["failed"] += 1
        results["details"].append(("Get A/B results", r.status_code, r.text))
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("COMPLETE WORKFLOW TEST")
    print("=" * 70)
    
    results = test_complete_workflow()
    
    print(f"\nResults: {results['passed']} passed, {results['failed']} failed")
    print(f"Success Rate: {results['passed']*100/(results['passed']+results['failed']):.1f}%")
    
    if results["details"]:
        print("\nFailed Tests:")
        for name, status, error in results["details"]:
            print(f"  - {name}: HTTP {status}")
            print(f"    {error[:100]}...")
    
    print("=" * 70)
```

---

## Phase 4: Execution Schedule

| Phase | Task | Estimated Time | Dependencies |
|-------|------|----------------|--------------|
| 1 | Data setup script | 30 min | None |
| 2.1 | Test creatives | 15 min | Phase 1 |
| 2.2 | Test click tracking | 15 min | Phase 2.1 |
| 2.3 | Test conversion tracking | 15 min | Phase 2.2 |
| 2.4 | Test fraud detection | 20 min | Phase 2.2 |
| 2.5 | Test analytics | 30 min | Phase 2.3 |
| 2.6 | Test reporting | 20 min | Phase 2.5 |
| 2.7 | Test A/B testing | 30 min | Phase 1 |
| 3 | Integration test | 30 min | All Phase 2 |
| 4 | Bug fixes & retest | 60 min | All tests |

**Total Estimated Time:** ~4.5 hours

---

## Success Metrics

- [ ] All 7 endpoint categories return 200 OK
- [ ] Complete workflow test passes 90%+
- [ ] No 500 errors in logs
- [ ] Data relationships properly maintained
- [ ] Analytics calculations accurate
- [ ] A/B test statistical significance calculated correctly

---

## Next Steps

1. Run Phase 1 data setup
2. Execute each Phase 2 test sequentially
3. Fix any 500 errors discovered
4. Run Phase 3 integration test
5. Document any remaining issues
6. Update API documentation with correct field names
