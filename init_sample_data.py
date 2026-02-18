"""
Script to initialize the database with sample data for the advertising system
"""

from datetime import datetime, timedelta

from database import (
    Ad,
    AdGroup,
    AdMetric,
    Advertiser,
    Campaign,
    CreativeAsset,
    db_manager,
)


def init_sample_data():
    """Initialize the database with sample data"""
    print("Initializing sample data for advertising system...")

    # Get database session
    db = next(db_manager.get_db())

    try:
        # Create sample advertiser
        advertiser = Advertiser(name="Sample Advertiser Inc.", contact_email="contact@sample-advertiser.com")
        db.add(advertiser)
        db.commit()
        db.refresh(advertiser)
        print(f"Created advertiser: {advertiser.name} (ID: {advertiser.id})")

        # Create sample campaign
        campaign = Campaign(
            advertiser_id=advertiser.id,
            name="Summer Sale Campaign",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            budget=5000.0,
            daily_budget=200.0,
            status="active",
        )
        db.add(campaign)
        db.commit()
        db.refresh(campaign)
        print(f"Created campaign: {campaign.name} (ID: {campaign.id})")

        # Create sample ad group
        ad_group = AdGroup(
            campaign_id=campaign.id,
            name="Mobile Ads Group",
            targeting_settings={
                "device_type": ["mobile"],
                "location": ["US", "CA", "UK"],
                "language": ["en"],
            },
            bid_strategy="manual",
        )
        db.add(ad_group)
        db.commit()
        db.refresh(ad_group)
        print(f"Created ad group: {ad_group.name} (ID: {ad_group.id})")

        # Create sample ad
        ad = Ad(
            advertiser_id=advertiser.id,
            ad_group_id=ad_group.id,
            title="Premium VPN Service",
            description="Secure and private browsing with our premium VPN service",
            url="https://example.com/vpn",
            targeting_constraints={"privacy_conscious": True, "security_focused": True},
            ethical_tags=["privacy", "security", "no_tracking"],
            quality_score=0.85,
            creative_format="banner",
            bid_amount=2.50,
            status="active",
            approval_status="approved",
        )
        db.add(ad)
        db.commit()
        db.refresh(ad)
        print(f"Created ad: {ad.title} (ID: {ad.id})")

        # Create sample creative asset
        creative = CreativeAsset(
            ad_id=ad.id,
            asset_type="image",
            asset_url="https://example.com/assets/vpn-banner-300x250.png",
            dimensions={"width": 300, "height": 250},
            checksum="a1b2c3d4e5f6",
        )
        db.add(creative)
        db.commit()
        db.refresh(creative)
        print(f"Created creative: {creative.asset_type} (ID: {creative.id})")

        # Create sample metrics
        today = datetime.now().date()
        metric = AdMetric(
            ad_id=ad.id,
            date=today,
            intent_goal="PURCHASE",
            intent_use_case="privacy",
            impression_count=1500,
            click_count=45,
            conversion_count=3,
            ctr=0.03,  # 3%
            cpc=0.85,  # $0.85
            roas=3.2,  # Return on ad spend
            engagement_rate=0.12,  # 12%
            expires_at=datetime.now() + timedelta(days=30),
        )
        db.add(metric)
        db.commit()
        print(f"Created metrics for ad {ad.id}")

        print("\nSample data initialization completed successfully!")
        print("\nSample data includes:")
        print(f"- 1 Advertiser: {advertiser.name}")
        print(f"- 1 Campaign: {campaign.name}")
        print(f"- 1 Ad Group: {ad_group.name}")
        print(f"- 1 Ad: {ad.title}")
        print(f"- 1 Creative Asset: {creative.asset_type}")
        print("- 1 Metric record")

    except Exception as e:
        print(f"Error initializing sample data: {str(e)}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    init_sample_data()
