"""
Intent Engine - Database Layer

This module implements the database layer using SQLAlchemy ORM with support for
both SQLite (for local development) and PostgreSQL (for production).
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import JSON, Column, Date, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)


# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./intent_engine.db")

# Connection pool settings from environment
POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "10"))
POOL_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.getenv("DATABASE_POOL_RECYCLE", "1800"))
PGBOUNCER_ENABLED = os.getenv("PGBOUNCER_ENABLED", "false").lower() in ("true", "1", "t")

# Configure engine based on database type
if "postgresql" in DATABASE_URL:
    if PGBOUNCER_ENABLED:
        engine = create_engine(
            DATABASE_URL,
            poolclass=NullPool,
            echo=False,
        )
        logger.info("PostgreSQL engine initialized with PgBouncer (NullPool)")
    else:
        # PostgreSQL configuration with connection pooling
        engine = create_engine(
            DATABASE_URL,
            pool_size=POOL_SIZE,
            max_overflow=POOL_MAX_OVERFLOW,
            pool_timeout=POOL_TIMEOUT,
            pool_recycle=POOL_RECYCLE,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info(f"PostgreSQL engine initialized with pool_size={POOL_SIZE}, max_overflow={POOL_MAX_OVERFLOW}")
elif "sqlite" in DATABASE_URL:
    # SQLite configuration (for local development only)
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, pool_pre_ping=True, echo=False)
    logger.warning("SQLite engine initialized (use PostgreSQL for production)")
else:
    # Default configuration
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
    logger.info(f"Database engine initialized for: {DATABASE_URL.split(':')[0]}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Advertiser(Base):
    """
    Advertisers (publishers) table
    """

    __tablename__ = "advertisers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    contact_email = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to ads
    ads = relationship("Ad", back_populates="advertiser")
    # New relationships for advertising system
    campaigns = relationship("Campaign", back_populates="advertiser")


class Campaign(Base):
    """
    Campaigns table
    """

    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, index=True)
    advertiser_id = Column(Integer, ForeignKey("advertisers.id"), nullable=False)
    name = Column(String, nullable=False)
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    budget = Column(Float, default=0.0)  # Total campaign budget
    daily_budget = Column(Float, default=0.0)  # Daily spending limit
    status = Column(String, default="active")  # active, paused, completed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    advertiser = relationship("Advertiser", back_populates="campaigns")
    ad_groups = relationship("AdGroup", back_populates="campaign")


class AdGroup(Base):
    """
    Ad Groups table
    """

    __tablename__ = "ad_groups"  # Using "ad_groups" to avoid conflicts with SQL reserved word

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False)
    name = Column(String, nullable=False)
    targeting_settings = Column(JSON)  # JSON for targeting criteria
    bid_strategy = Column(String, default="manual")  # manual or automated bidding
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    campaign = relationship("Campaign", back_populates="ad_groups")
    ads = relationship("Ad", back_populates="ad_group")


class Ad(Base):
    """
    Ads (with fairness constraints) table
    """

    __tablename__ = "ads"

    id = Column(Integer, primary_key=True, index=True)
    advertiser_id = Column(Integer, ForeignKey("advertisers.id"), nullable=False)
    ad_group_id = Column(Integer, ForeignKey("ad_groups.id"))  # NEW: Link to ad group
    title = Column(String, nullable=False)
    description = Column(Text)
    url = Column(String, nullable=False)
    targeting_constraints = Column(JSON)  # e.g., [{"dimension": "device_type", "value": "mobile"}]
    ethical_tags = Column(JSON)  # e.g., ["privacy", "open_source"]
    quality_score = Column(Float, default=0.5)
    creative_format = Column(String)  # NEW: Banner, native, video, etc.
    bid_amount = Column(Float, default=0.0)  # NEW: Current bid amount
    status = Column(String, default="active")  # NEW: active, paused, disapproved
    approval_status = Column(String, default="pending")  # NEW: pending, approved, rejected
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to advertiser
    advertiser = relationship("Advertiser", back_populates="ads")
    # NEW: Relationship to ad group
    ad_group = relationship("AdGroup", back_populates="ads")
    # NEW: Relationship to creative assets
    creative_assets = relationship("CreativeAsset", back_populates="ad")
    # Relationship to metrics
    metrics = relationship("AdMetric", back_populates="ad")


class CreativeAsset(Base):
    """
    Creative Assets table
    """

    __tablename__ = "creative_assets"

    id = Column(Integer, primary_key=True, index=True)
    ad_id = Column(Integer, ForeignKey("ads.id"), nullable=False)
    asset_type = Column(String, nullable=False)  # image, video, text, html
    asset_url = Column(String, nullable=False)  # Location of creative asset
    dimensions = Column(JSON)  # {"width": 300, "height": 250} for display ads
    checksum = Column(String)  # Integrity verification
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    ad = relationship("Ad", back_populates="creative_assets")


class AdMetric(Base):
    """
    Aggregated Metrics (with differential privacy applied later) table
    """

    __tablename__ = "ad_metrics"

    id = Column(Integer, primary_key=True, index=True)
    ad_id = Column(Integer, ForeignKey("ads.id"), nullable=False)
    date = Column(Date, nullable=False)
    intent_goal = Column(String)  # e.g., "LEARN"
    intent_use_case = Column(String)  # e.g., "learning"
    impression_count = Column(Integer, default=0)
    click_count = Column(Integer, default=0)
    conversion_count = Column(Integer, default=0)
    ctr = Column(Float)  # NEW: Click-through rate
    cpm = Column(Float)  # NEW: Cost per thousand impressions
    cpc = Column(Float)  # NEW: Cost per click
    roas = Column(Float)  # NEW: Return on ad spend
    engagement_rate = Column(Float)  # NEW: Interaction rate
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)  # 30 days from creation

    # Relationship to ad
    ad = relationship("Ad", back_populates="metrics")


class ClickTracking(Base):
    """
    Click Tracking table for recording ad clicks
    """

    __tablename__ = "click_tracking"

    id = Column(Integer, primary_key=True, index=True)
    ad_id = Column(Integer, ForeignKey("ads.id"), nullable=False)
    session_id = Column(String)  # Anonymous session identifier
    timestamp = Column(DateTime(timezone=True), server_default=func.now())  # Click timestamp
    ip_hash = Column(String)  # Hashed IP for fraud detection
    user_agent_hash = Column(String)  # Hashed user agent
    referring_url = Column(String)  # Source of click

    # Relationship to ad
    ad = relationship("Ad")


class ConversionTracking(Base):
    """
    Conversion Tracking table for recording ad conversions
    """

    __tablename__ = "conversion_tracking"

    id = Column(Integer, primary_key=True, index=True)
    click_id = Column(Integer, ForeignKey("click_tracking.id"), nullable=False)  # Foreign key to click tracking
    conversion_type = Column(String)  # Purchase, signup, download, etc.
    value = Column(Float)  # Conversion value (if applicable)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())  # Conversion timestamp
    status = Column(String, default="pending")  # Verified, pending, rejected

    # Relationship to click
    click = relationship("ClickTracking")


class FraudDetection(Base):
    """
    Fraud Detection table for tracking suspicious activities
    """

    __tablename__ = "fraud_detection"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer)  # ID of suspicious event
    event_type = Column(String)  # Click, impression, conversion
    reason = Column(String)  # Reason for flagging
    severity = Column(String)  # Low, medium, high risk
    review_status = Column(String, default="pending")  # Pending, reviewed, action_taken
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # Timestamp

    # Relationship to ad (optional, could be linked to ad_id if needed)
    ad_id = Column(Integer, ForeignKey("ads.id"), nullable=True)
    ad = relationship("Ad")


# Create tables if they don't exist
# Moved this to happen inside the DatabaseManager to avoid import-time issues


class DatabaseManager:
    """
    Manager class for database operations and maintenance
    """

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.start_cleanup_job()

    def initialize_database(self):
        pass

    def start_cleanup_job(self):
        """
        Start the background job to clean up expired metrics every hour
        """
        self.scheduler.add_job(
            self.cleanup_expired_metrics,
            trigger=IntervalTrigger(hours=1),
            id="cleanup_expired_metrics",
            name="Cleanup expired ad metrics",
            replace_existing=True,
        )
        if not self.scheduler.running:
            self.scheduler.start()

        # Also schedule shutdown handler
        import atexit

        atexit.register(lambda: self.scheduler.shutdown())

    def get_db(self):
        """
        Dependency to get database session
        """
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def cleanup_expired_metrics(self):
        """
        Delete expired metrics from the database
        """
        db = SessionLocal()
        try:
            # Find and delete expired metrics
            expired_before = datetime.now(timezone.utc)
            deleted_count = db.query(AdMetric).filter(AdMetric.expires_at < expired_before).delete()
            db.commit()
            print(f"Cleaned up {deleted_count} expired ad metrics")
        except Exception as e:
            print(f"Error cleaning up expired metrics: {e}")
            db.rollback()
        finally:
            db.close()


# Initialize the database manager
db_manager = DatabaseManager()
