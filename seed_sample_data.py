"""
Seed Data Script for Intent Engine Advertising System

This script populates the database with realistic sample data for testing.
Run this after the database is initialized.
"""

import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import models
from database import (
    Ad,
    AdGroup,
    AdMetric,
    Advertiser,
    Campaign,
    CreativeAsset,
    db_manager,
)


def seed_data():
    """Seed the database with sample data"""
    print("=" * 60)
    print("Intent Engine - Seeding Sample Data")
    print("=" * 60)

    db = next(db_manager.get_db())

    try:
        # Check if data already exists
        existing_advertisers = db.query(Advertiser).count()
        if existing_advertisers > 0:
            print("Database already has data. Skipping seed.")
            return

        print("\n[1/6] Creating advertisers...")

        # Create advertisers
        advertisers = [
            Advertiser(name="TechGadgets Inc.", contact_email="ads@techgadgets.com"),
            Advertiser(name="Privacy Solutions Ltd.", contact_email="marketing@privacysolutions.io"),
            Advertiser(name="EcoFriendly Products", contact_email="hello@ecofriendly.shop"),
            Advertiser(name="OpenSource Tools", contact_email="community@opensource.tools"),
        ]

        for adv in advertisers:
            db.add(adv)
        db.commit()

        for adv in advertisers:
            db.refresh(adv)
            print(f"  ✓ Created advertiser: {adv.name} (ID: {adv.id})")

        print("\n[2/6] Creating campaigns...")

        # Create campaigns
        today = datetime.now()
        campaigns = [
            Campaign(
                advertiser_id=advertisers[0].id,
                name="Summer Laptop Sale 2026",
                start_date=today,
                end_date=today + timedelta(days=60),
                budget=10000.0,
                daily_budget=200.0,
                status="active",
            ),
            Campaign(
                advertiser_id=advertisers[1].id,
                name="Privacy-First VPN Campaign",
                start_date=today,
                end_date=today + timedelta(days=90),
                budget=15000.0,
                daily_budget=150.0,
                status="active",
            ),
            Campaign(
                advertiser_id=advertisers[2].id,
                name="Sustainable Living Products",
                start_date=today - timedelta(days=15),
                end_date=today + timedelta(days=45),
                budget=5000.0,
                daily_budget=100.0,
                status="active",
            ),
            Campaign(
                advertiser_id=advertisers[3].id,
                name="Developer Tools Promotion",
                start_date=today,
                end_date=today + timedelta(days=30),
                budget=8000.0,
                daily_budget=250.0,
                status="active",
            ),
        ]

        for camp in campaigns:
            db.add(camp)
        db.commit()

        for camp in campaigns:
            db.refresh(camp)
            print(f"  ✓ Created campaign: {camp.name} (ID: {camp.id})")

        print("\n[3/6] Creating ad groups...")

        # Create ad groups
        ad_groups = [
            AdGroup(
                campaign_id=campaigns[0].id,
                name="Programming Laptops",
                targeting_settings={
                    "device_type": ["desktop", "laptop"],
                    "interests": ["programming", "software development"],
                    "location": ["US", "UK", "CA", "IN"],
                },
                bid_strategy="automatic",
            ),
            AdGroup(
                campaign_id=campaigns[0].id,
                name="Budget Laptops",
                targeting_settings={"device_type": ["laptop"], "price_range": ["budget"], "location": ["US", "IN"]},
                bid_strategy="manual",
            ),
            AdGroup(
                campaign_id=campaigns[1].id,
                name="Privacy Conscious Users",
                targeting_settings={"interests": ["privacy", "security", "vpn"], "location": ["US", "UK", "DE", "FR"]},
                bid_strategy="automatic",
            ),
            AdGroup(
                campaign_id=campaigns[2].id,
                name="Eco Products",
                targeting_settings={
                    "interests": ["sustainability", "eco-friendly", "green living"],
                    "location": ["US", "UK", "CA"],
                },
                bid_strategy="automatic",
            ),
            AdGroup(
                campaign_id=campaigns[3].id,
                name="Developer Tools",
                targeting_settings={
                    "interests": ["software development", "devops", "cloud"],
                    "location": ["US", "UK", "IN", "DE"],
                },
                bid_strategy="automatic",
            ),
        ]

        for ag in ad_groups:
            db.add(ag)
        db.commit()

        for ag in ad_groups:
            db.refresh(ag)
            print(f"  ✓ Created ad group: {ag.name} (ID: {ag.id})")

        print("\n[4/6] Creating ads...")

        # Create ads
        ads = [
            Ad(
                advertiser_id=advertisers[0].id,
                ad_group_id=ad_groups[0].id,
                title="Best Programming Laptops 2026",
                description="High-performance laptops for developers. Starting at ₹45,999",
                url="https://techgadgets.com/laptops/programming",
                targeting_constraints={"category": "electronics", "use_case": "programming"},
                ethical_tags=["quality", "performance"],
                quality_score=0.85,
                creative_format="native",
                bid_amount=2.50,
                status="active",
                approval_status="approved",
            ),
            Ad(
                advertiser_id=advertisers[0].id,
                ad_group_id=ad_groups[1].id,
                title="Budget Laptops Under ₹30,000",
                description="Affordable laptops perfect for students and everyday use",
                url="https://techgadgets.com/laptops/budget",
                targeting_constraints={"category": "electronics", "price_range": "budget"},
                ethical_tags=["affordable", "value"],
                quality_score=0.78,
                creative_format="banner",
                bid_amount=1.50,
                status="active",
                approval_status="approved",
            ),
            Ad(
                advertiser_id=advertisers[1].id,
                ad_group_id=ad_groups[2].id,
                title="Secure VPN - No Logs Policy",
                description="Privacy-first VPN with military-grade encryption. Free trial available.",
                url="https://privacysolutions.io/vpn",
                targeting_constraints={"category": "security", "privacy_focused": True},
                ethical_tags=["privacy", "security", "no_tracking"],
                quality_score=0.92,
                creative_format="native",
                bid_amount=3.00,
                status="active",
                approval_status="approved",
            ),
            Ad(
                advertiser_id=advertisers[2].id,
                ad_group_id=ad_groups[3].id,
                title="Eco-Friendly Home Products",
                description="Sustainable products for a greener lifestyle. Free shipping on orders over ₹500",
                url="https://ecofriendly.shop/home",
                targeting_constraints={"category": "lifestyle", "sustainability": True},
                ethical_tags=["eco_friendly", "sustainable", "organic"],
                quality_score=0.88,
                creative_format="banner",
                bid_amount=2.00,
                status="active",
                approval_status="approved",
            ),
            Ad(
                advertiser_id=advertisers[3].id,
                ad_group_id=ad_groups[4].id,
                title="Open Source Dev Tools",
                description="Free and open-source developer tools. Join 50,000+ developers.",
                url="https://opensource.tools",
                targeting_constraints={"category": "software", "open_source": True},
                ethical_tags=["open_source", "free", "community"],
                quality_score=0.90,
                creative_format="native",
                bid_amount=1.80,
                status="active",
                approval_status="approved",
            ),
        ]

        for ad in ads:
            db.add(ad)
        db.commit()

        for ad in ads:
            db.refresh(ad)
            print(f"  ✓ Created ad: {ad.title} (ID: {ad.id})")

        print("\n[5/6] Creating creative assets...")

        # Create creative assets
        creatives = [
            CreativeAsset(
                ad_id=ads[0].id,
                asset_type="image",
                asset_url="https://techgadgets.com/assets/laptop-programming-banner.jpg",
                dimensions={"width": 728, "height": 90},
                checksum="abc123def456",
            ),
            CreativeAsset(
                ad_id=ads[1].id,
                asset_type="image",
                asset_url="https://techgadgets.com/assets/budget-laptop-banner.jpg",
                dimensions={"width": 300, "height": 250},
                checksum="def456ghi789",
            ),
            CreativeAsset(
                ad_id=ads[2].id,
                asset_type="video",
                asset_url="https://privacysolutions.io/assets/vpn-promo.mp4",
                dimensions={"width": 640, "height": 480},
                checksum="ghi789jkl012",
            ),
            CreativeAsset(
                ad_id=ads[3].id,
                asset_type="image",
                asset_url="https://ecofriendly.shop/assets/eco-products.jpg",
                dimensions={"width": 728, "height": 90},
                checksum="jkl012mno345",
            ),
            CreativeAsset(
                ad_id=ads[4].id,
                asset_type="image",
                asset_url="https://opensource.tools/assets/dev-tools-banner.png",
                dimensions={"width": 970, "height": 250},
                checksum="mno345pqr678",
            ),
        ]

        for creative in creatives:
            db.add(creative)
        db.commit()

        for creative in creatives:
            db.refresh(creative)
            print(f"  ✓ Created creative asset (ID: {creative.id})")

        print("\n[6/6] Creating sample metrics...")

        # Create sample metrics for the past 7 days
        for ad in ads:
            for i in range(7):
                metric_date = (today - timedelta(days=i)).date()
                metric = AdMetric(
                    ad_id=ad.id,
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
                    expires_at=today + timedelta(days=90),
                )
                db.add(metric)

        db.commit()
        print("  ✓ Created metrics for past 7 days")

        print("\n" + "=" * 60)
        print("Seed data completed successfully!")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - {len(advertisers)} advertisers")
        print(f"  - {len(campaigns)} campaigns")
        print(f"  - {len(ad_groups)} ad groups")
        print(f"  - {len(ads)} ads")
        print(f"  - {len(creatives)} creative assets")
        print(f"  - {len(ads) * 7} metric records")
        print("=" * 60)

    except Exception as e:
        db.rollback()
        print(f"Error seeding data: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_data()
