#!/usr/bin/env python
"""Check database status"""

from database import Ad, AdGroup, AdMetric, Advertiser, Campaign, CreativeAsset, SessionLocal

db = SessionLocal()
try:
    print(f"Advertisers: {db.query(Advertiser).count()}")
    print(f"Campaigns: {db.query(Campaign).count()}")
    print(f"Ad Groups: {db.query(AdGroup).count()}")
    print(f"Ads: {db.query(Ad).count()}")
    print(f"Creative Assets: {db.query(CreativeAsset).count()}")
    print(f"Metrics: {db.query(AdMetric).count()}")
finally:
    db.close()
