"""
Intent Engine - Seed Data Script

This script populates the database with sample advertisers, campaigns, ad groups, and ads
for testing and demonstration purposes.

Usage:
    python seed_data.py [--reset]
    
Options:
    --reset    Drop and recreate all tables before seeding
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import db_manager, Advertiser, Campaign, AdGroup, Ad, CreativeAsset, Base, engine
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reset_database():
    """Drop and recreate all tables"""
    logger.info("Resetting database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database reset complete")


def seed_advertisers(db: Session) -> List[Advertiser]:
    """Create sample advertisers"""
    logger.info("Seeding advertisers...")
    
    advertisers_data = [
        {"name": "TechCorp Solutions", "contact_email": "ads@techcorp.com"},
        {"name": "GreenEnergy Inc", "contact_email": "marketing@greenenergy.com"},
        {"name": "EduLearn Platform", "contact_email": "partnerships@edulearn.com"},
        {"name": "HealthFirst Medical", "contact_email": "ads@healthfirst.com"},
        {"name": "FinanceWise", "contact_email": "marketing@financewise.com"},
        {"name": "TravelBug Adventures", "contact_email": "ads@travelbug.com"},
        {"name": "FoodieDelights", "contact_email": "partnerships@foodiedelights.com"},
        {"name": "StyleHub Fashion", "contact_email": "ads@stylehub.com"},
        {"name": "GameZone Entertainment", "contact_email": "marketing@gamezone.com"},
        {"name": "HomeComfort Living", "contact_email": "ads@homecomfort.com"},
    ]
    
    advertisers = []
    for data in advertisers_data:
        advertiser = Advertiser(**data)
        db.add(advertiser)
        advertisers.append(advertiser)
    
    db.commit()
    logger.info(f"Created {len(advertisers)} advertisers")
    return advertisers


def seed_campaigns(db: Session, advertisers: List[Advertiser]) -> List[Campaign]:
    """Create sample campaigns"""
    logger.info("Seeding campaigns...")
    
    now = datetime.utcnow()
    campaigns_data = [
        # TechCorp campaigns
        {"advertiser_id": advertisers[0].id, "name": "Q1 2026 Product Launch", "budget": 50000, "daily_budget": 500},
        {"advertiser_id": advertisers[0].id, "name": "Developer Tools Promotion", "budget": 25000, "daily_budget": 250},
        
        # GreenEnergy campaigns
        {"advertiser_id": advertisers[1].id, "name": "Solar Panel Awareness", "budget": 30000, "daily_budget": 300},
        {"advertiser_id": advertisers[1].id, "name": "EV Charging Network", "budget": 20000, "daily_budget": 200},
        
        # EduLearn campaigns
        {"advertiser_id": advertisers[2].id, "name": "Online Courses 2026", "budget": 40000, "daily_budget": 400},
        {"advertiser_id": advertisers[2].id, "name": "Professional Certifications", "budget": 35000, "daily_budget": 350},
        
        # HealthFirst campaigns
        {"advertiser_id": advertisers[3].id, "name": "Telemedicine Services", "budget": 25000, "daily_budget": 250},
        {"advertiser_id": advertisers[3].id, "name": "Health Screening Campaign", "budget": 15000, "daily_budget": 150},
        
        # FinanceWise campaigns
        {"advertiser_id": advertisers[4].id, "name": "Investment Platform Launch", "budget": 45000, "daily_budget": 450},
        {"advertiser_id": advertisers[4].id, "name": "Retirement Planning", "budget": 20000, "daily_budget": 200},
        
        # TravelBug campaigns
        {"advertiser_id": advertisers[5].id, "name": "Summer Vacation Deals", "budget": 35000, "daily_budget": 350},
        {"advertiser_id": advertisers[5].id, "name": "Adventure Tours 2026", "budget": 25000, "daily_budget": 250},
        
        # FoodieDelights campaigns
        {"advertiser_id": advertisers[6].id, "name": "Recipe App Promotion", "budget": 15000, "daily_budget": 150},
        {"advertiser_id": advertisers[6].id, "name": "Meal Kit Delivery", "budget": 20000, "daily_budget": 200},
        
        # StyleHub campaigns
        {"advertiser_id": advertisers[7].id, "name": "Spring Collection 2026", "budget": 30000, "daily_budget": 300},
        {"advertiser_id": advertisers[7].id, "name": "Sustainable Fashion", "budget": 18000, "daily_budget": 180},
        
        # GameZone campaigns
        {"advertiser_id": advertisers[8].id, "name": "New Game Releases", "budget": 40000, "daily_budget": 400},
        {"advertiser_id": advertisers[8].id, "name": "Gaming Hardware Sale", "budget": 25000, "daily_budget": 250},
        
        # HomeComfort campaigns
        {"advertiser_id": advertisers[9].id, "name": "Smart Home Devices", "budget": 35000, "daily_budget": 350},
        {"advertiser_id": advertisers[9].id, "name": "Furniture Collection", "budget": 28000, "daily_budget": 280},
    ]
    
    campaigns = []
    for data in campaigns_data:
        campaign = Campaign(
            advertiser_id=data["advertiser_id"],
            name=data["name"],
            start_date=now - timedelta(days=30),
            end_date=now + timedelta(days=60),
            budget=data["budget"],
            daily_budget=data["daily_budget"],
            status="active"
        )
        db.add(campaign)
        campaigns.append(campaign)
    
    db.commit()
    logger.info(f"Created {len(campaigns)} campaigns")
    return campaigns


def seed_ad_groups(db: Session, campaigns: List[Campaign]) -> List[AdGroup]:
    """Create sample ad groups"""
    logger.info("Seeding ad groups...")
    
    ad_groups_data = [
        # For each campaign, create 2-3 ad groups
        {"campaign_id": campaigns[0].id, "name": "Search Ads", "bid_strategy": "automated"},
        {"campaign_id": campaigns[0].id, "name": "Display Ads", "bid_strategy": "manual"},
        {"campaign_id": campaigns[0].id, "name": "Video Ads", "bid_strategy": "automated"},
        
        {"campaign_id": campaigns[1].id, "name": "Developer Targeting", "bid_strategy": "automated"},
        {"campaign_id": campaigns[1].id, "name": "Enterprise Targeting", "bid_strategy": "manual"},
    ]
    
    # Generate ad groups for all campaigns
    for i, campaign in enumerate(campaigns):
        if i < 2:  # Already defined above
            continue
        
        ad_groups_data.extend([
            {"campaign_id": campaign.id, "name": f"{campaign.name} - Group A", "bid_strategy": "automated"},
            {"campaign_id": campaign.id, "name": f"{campaign.name} - Group B", "bid_strategy": "manual"},
        ])
    
    ad_groups = []
    for data in ad_groups_data:
        ad_group = AdGroup(
            campaign_id=data["campaign_id"],
            name=data["name"],
            bid_strategy=data["bid_strategy"],
            targeting_settings={"demographics": "all", "interests": []}
        )
        db.add(ad_group)
        ad_groups.append(ad_group)
    
    db.commit()
    logger.info(f"Created {len(ad_groups)} ad groups")
    return ad_groups


def seed_ads(db: Session, advertisers: List[Advertiser], ad_groups: List[AdGroup]) -> List[Ad]:
    """Create sample ads"""
    logger.info("Seeding ads...")
    
    ads_data = [
        # TechCorp ads
        {
            "advertiser_id": advertisers[0].id,
            "ad_group_id": ad_groups[0].id if ad_groups else None,
            "title": "Professional Developer Tools",
            "description": "Boost your productivity with our suite of developer tools. Free trial available.",
            "url": "https://techcorp.com/dev-tools",
            "quality_score": 0.85,
            "creative_format": "banner",
            "ethical_tags": ["open_source", "privacy"],
            "targeting_constraints": {"device_type": ["desktop"], "interests": ["programming", "technology"]}
        },
        {
            "advertiser_id": advertisers[0].id,
            "ad_group_id": ad_groups[1].id if len(ad_groups) > 1 else None,
            "title": "Cloud Infrastructure Solutions",
            "description": "Scalable, secure, and cost-effective cloud infrastructure for modern businesses.",
            "url": "https://techcorp.com/cloud",
            "quality_score": 0.90,
            "creative_format": "native",
            "ethical_tags": ["enterprise", "security"],
            "targeting_constraints": {"industry": ["technology", "finance"]}
        },
        
        # GreenEnergy ads
        {
            "advertiser_id": advertisers[1].id,
            "ad_group_id": ad_groups[2].id if len(ad_groups) > 2 else None,
            "title": "Solar Panels for Your Home",
            "description": "Reduce your electricity bills with clean solar energy. Get a free quote today.",
            "url": "https://greenenergy.com/solar",
            "quality_score": 0.82,
            "creative_format": "banner",
            "ethical_tags": ["sustainability", "green"],
            "targeting_constraints": {"location": ["suburban", "rural"]}
        },
        {
            "advertiser_id": advertisers[1].id,
            "ad_group_id": ad_groups[3].id if len(ad_groups) > 3 else None,
            "title": "EV Charging Network",
            "description": "Find charging stations near you. Fast, reliable, and eco-friendly.",
            "url": "https://greenenergy.com/ev-charging",
            "quality_score": 0.78,
            "creative_format": "native",
            "ethical_tags": ["sustainability", "transportation"],
            "targeting_constraints": {"interests": ["electric_vehicles", "sustainability"]}
        },
        
        # EduLearn ads
        {
            "advertiser_id": advertisers[2].id,
            "ad_group_id": ad_groups[4].id if len(ad_groups) > 4 else None,
            "title": "Learn New Skills Online",
            "description": "Over 5000 courses in technology, business, and creative fields. Start learning today.",
            "url": "https://edulearn.com/courses",
            "quality_score": 0.88,
            "creative_format": "video",
            "ethical_tags": ["education", "career"],
            "targeting_constraints": {"age_range": ["18-34", "35-54"]}
        },
        
        # HealthFirst ads
        {
            "advertiser_id": advertisers[3].id,
            "ad_group_id": ad_groups[5].id if len(ad_groups) > 5 else None,
            "title": "Online Doctor Consultations",
            "description": "Speak with licensed physicians from the comfort of your home. Available 24/7.",
            "url": "https://healthfirst.com/telemedicine",
            "quality_score": 0.91,
            "creative_format": "banner",
            "ethical_tags": ["healthcare", "accessibility"],
            "targeting_constraints": {"interests": ["health", "wellness"]}
        },
        
        # FinanceWise ads
        {
            "advertiser_id": advertisers[4].id,
            "ad_group_id": ad_groups[6].id if len(ad_groups) > 6 else None,
            "title": "Smart Investment Platform",
            "description": "AI-powered investment strategies for everyone. Start with as little as $100.",
            "url": "https://financewise.com/invest",
            "quality_score": 0.84,
            "creative_format": "native",
            "ethical_tags": ["finance", "technology"],
            "targeting_constraints": {"income_range": ["middle", "high"]}
        },
        
        # TravelBug ads
        {
            "advertiser_id": advertisers[5].id,
            "ad_group_id": ad_groups[7].id if len(ad_groups) > 7 else None,
            "title": "Adventure Awaits - Book Now",
            "description": "Exclusive deals on adventure tours worldwide. Limited time offers available.",
            "url": "https://travelbug.com/adventures",
            "quality_score": 0.79,
            "creative_format": "video",
            "ethical_tags": ["travel", "adventure"],
            "targeting_constraints": {"interests": ["travel", "outdoor_activities"]}
        },
        
        # GameZone ads
        {
            "advertiser_id": advertisers[8].id,
            "ad_group_id": ad_groups[8].id if len(ad_groups) > 8 else None,
            "title": "New RPG Release - Pre-order Now",
            "description": "The most anticipated RPG of 2026. Pre-order and get exclusive bonus content.",
            "url": "https://gamezone.com/rpg-2026",
            "quality_score": 0.87,
            "creative_format": "video",
            "ethical_tags": ["gaming", "entertainment"],
            "targeting_constraints": {"interests": ["gaming", "rpg"]}
        },
        
        # HomeComfort ads
        {
            "advertiser_id": advertisers[9].id,
            "ad_group_id": ad_groups[9].id if len(ad_groups) > 9 else None,
            "title": "Smart Home Starter Kit",
            "description": "Transform your home with our easy-to-use smart devices. Special bundle pricing.",
            "url": "https://homecomfort.com/smart-kit",
            "quality_score": 0.83,
            "creative_format": "banner",
            "ethical_tags": ["technology", "home"],
            "targeting_constraints": {"interests": ["smart_home", "technology"]}
        },
    ]
    
    ads = []
    for data in ads_data:
        ad = Ad(
            advertiser_id=data["advertiser_id"],
            ad_group_id=data["ad_group_id"],
            title=data["title"],
            description=data["description"],
            url=data["url"],
            quality_score=data["quality_score"],
            creative_format=data["creative_format"],
            ethical_tags=data["ethical_tags"],
            targeting_constraints=data["targeting_constraints"],
            bid_amount=1.50,
            status="active",
            approval_status="approved"
        )
        db.add(ad)
        ads.append(ad)
    
    db.commit()
    logger.info(f"Created {len(ads)} ads")
    return ads


def seed_creative_assets(db: Session, ads: List[Ad]) -> List[CreativeAsset]:
    """Create sample creative assets"""
    logger.info("Seeding creative assets...")
    
    assets_data = []
    for i, ad in enumerate(ads):
        if ad.creative_format == "banner":
            assets_data.append({
                "ad_id": ad.id,
                "asset_type": "image",
                "asset_url": f"/assets/banners/ad_{ad.id}_300x250.jpg",
                "dimensions": {"width": 300, "height": 250}
            })
            assets_data.append({
                "ad_id": ad.id,
                "asset_type": "image",
                "asset_url": f"/assets/banners/ad_{ad.id}_728x90.jpg",
                "dimensions": {"width": 728, "height": 90}
            })
        elif ad.creative_format == "video":
            assets_data.append({
                "ad_id": ad.id,
                "asset_type": "video",
                "asset_url": f"/assets/videos/ad_{ad.id}_30s.mp4",
                "dimensions": {"width": 1920, "height": 1080, "duration": 30}
            })
        elif ad.creative_format == "native":
            assets_data.append({
                "ad_id": ad.id,
                "asset_type": "html",
                "asset_url": f"/assets/native/ad_{ad.id}.html",
                "dimensions": {"responsive": True}
            })
    
    creative_assets = []
    for data in assets_data:
        asset = CreativeAsset(
            ad_id=data["ad_id"],
            asset_type=data["asset_type"],
            asset_url=data["asset_url"],
            dimensions=data["dimensions"],
            checksum=f"checksum_{data['ad_id']}_{data['asset_type']}"
        )
        db.add(asset)
        creative_assets.append(asset)
    
    db.commit()
    logger.info(f"Created {len(creative_assets)} creative assets")
    return creative_assets


def main():
    """Main function to seed all data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Seed Intent Engine database with sample data')
    parser.add_argument('--reset', action='store_true', help='Drop and recreate all tables before seeding')
    args = parser.parse_args()
    
    logger.info("Starting data seeding...")
    
    # Reset database if requested
    if args.reset:
        reset_database()
    else:
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)
    
    # Create database session
    db = next(db_manager.get_db())
    
    try:
        # Seed all data
        advertisers = seed_advertisers(db)
        campaigns = seed_campaigns(db, advertisers)
        ad_groups = seed_ad_groups(db, campaigns)
        ads = seed_ads(db, advertisers, ad_groups)
        creative_assets = seed_creative_assets(db, ads)
        
        logger.info("=" * 60)
        logger.info("Seeding complete!")
        logger.info("=" * 60)
        logger.info(f"Advertisers: {len(advertisers)}")
        logger.info(f"Campaigns: {len(campaigns)}")
        logger.info(f"Ad Groups: {len(ad_groups)}")
        logger.info(f"Ads: {len(ads)}")
        logger.info(f"Creative Assets: {len(creative_assets)}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error during seeding: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
