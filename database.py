"""
Intent Engine - Database Layer

This module implements the database layer using SQLAlchemy ORM with support for
both SQLite (for local development) and PostgreSQL (for production).
"""

import logging
import os
from datetime import UTC, datetime
from enum import Enum

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)


class ABTestStatus(Enum):
    """A/B Test status options"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./intent_engine.db")
PGBOUNCER_ENABLED = os.getenv("PGBOUNCER_ENABLED", "false").lower() == "true"

# Connection pool settings from environment
POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "10"))
POOL_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.getenv("DATABASE_POOL_RECYCLE", "1800"))

# Configure engine based on database type
if "postgresql" in DATABASE_URL:
    # PostgreSQL configuration with connection pooling
    # When using PgBouncer, disable pool_pre_ping and use NullPool for DDL operations
    if PGBOUNCER_ENABLED:
        from sqlalchemy.pool import NullPool

        engine = create_engine(
            DATABASE_URL,
            poolclass=NullPool,
            pool_pre_ping=True,
            echo=False,
            isolation_level="AUTOCOMMIT",  # Required for DDL with PgBouncer
        )
        logger.info("PostgreSQL engine initialized with PgBouncer (session mode)")
    else:
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
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
        echo=False,
    )
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
    advertiser_id = Column(Integer, ForeignKey("advertisers.id"), nullable=False, index=True)
    name = Column(String, nullable=False, index=True)
    start_date = Column(DateTime(timezone=True), index=True)
    end_date = Column(DateTime(timezone=True), index=True)
    budget = Column(Float, default=0.0)  # Total campaign budget
    daily_budget = Column(Float, default=0.0)  # Daily spending limit
    status = Column(String, default="active", index=True)  # active, paused, completed
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    advertiser = relationship("Advertiser", back_populates="campaigns")
    ad_groups = relationship("AdGroup", back_populates="campaign")

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_campaigns_advertiser_id_status", "advertiser_id", "status"),
        Index("ix_campaigns_status_dates", "status", "start_date", "end_date"),
    )


class AdGroup(Base):
    """
    Ad Groups table
    """

    __tablename__ = "ad_groups"  # Using "ad_groups" to avoid conflicts with SQL reserved word

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False, index=True)
    name = Column(String, nullable=False, index=True)
    targeting_settings = Column(JSON)  # JSON for targeting criteria
    bid_strategy = Column(String, default="manual")  # manual or automated bidding
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    campaign = relationship("Campaign", back_populates="ad_groups")
    ads = relationship("Ad", back_populates="ad_group")

    # Composite indexes for common queries
    __table_args__ = (Index("ix_ad_groups_campaign_id_bid_strategy", "campaign_id", "bid_strategy"),)


class Ad(Base):
    """
    Ads (with fairness constraints) table
    """

    __tablename__ = "ads"

    id = Column(Integer, primary_key=True, index=True)
    advertiser_id = Column(Integer, ForeignKey("advertisers.id"), nullable=False, index=True)
    ad_group_id = Column(Integer, ForeignKey("ad_groups.id"), index=True)  # NEW: Link to ad group
    title = Column(String, nullable=False)
    description = Column(Text)
    url = Column(String, nullable=False)
    targeting_constraints = Column(JSON)  # e.g., [{"dimension": "device_type", "value": "mobile"}]
    ethical_tags = Column(JSON)  # e.g., ["privacy", "open_source"]
    quality_score = Column(Float, default=0.5)
    creative_format = Column(String)  # NEW: Banner, native, video, etc.
    bid_amount = Column(Float, default=0.0)  # NEW: Current bid amount
    status = Column(String, default="active", index=True)  # NEW: active, paused, disapproved
    approval_status = Column(String, default="pending", index=True)  # NEW: pending, approved, rejected
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationship to advertiser
    advertiser = relationship("Advertiser", back_populates="ads")
    # NEW: Relationship to ad group
    ad_group = relationship("AdGroup", back_populates="ads")
    # NEW: Relationship to creative assets
    creative_assets = relationship("CreativeAsset", back_populates="ad")
    # Relationship to metrics
    metrics = relationship("AdMetric", back_populates="ad")

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_ads_advertiser_status", "advertiser_id", "status"),
        Index("ix_ads_approval_status", "approval_status", "status"),
    )


class CreativeAsset(Base):
    """
    Creative Assets table
    """

    __tablename__ = "creative_assets"

    id = Column(Integer, primary_key=True, index=True)
    ad_id = Column(Integer, ForeignKey("ads.id"), nullable=False, index=True)
    asset_type = Column(String, nullable=False, index=True)  # image, video, text, html
    asset_url = Column(String, nullable=False)  # Location of creative asset
    dimensions = Column(JSON)  # {"width": 300, "height": 250} for display ads
    checksum = Column(String)  # Integrity verification
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    ad = relationship("Ad", back_populates="creative_assets")

    # Composite indexes for common queries
    __table_args__ = (Index("ix_creative_assets_ad_id_asset_type", "ad_id", "asset_type"),)


class AdMetric(Base):
    """
    Aggregated Metrics (with differential privacy applied later) table
    """

    __tablename__ = "ad_metrics"

    id = Column(Integer, primary_key=True, index=True)
    ad_id = Column(Integer, ForeignKey("ads.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
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
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)  # 30 days from creation

    # Relationship to ad
    ad = relationship("Ad", back_populates="metrics")

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_ad_metrics_ad_id_date", "ad_id", "date"),
        Index("ix_ad_metrics_date_goal", "date", "intent_goal"),
    )


class ClickTracking(Base):
    """
    Click Tracking table for recording ad clicks
    """

    __tablename__ = "click_tracking"

    id = Column(Integer, primary_key=True, index=True)
    ad_id = Column(Integer, ForeignKey("ads.id"), nullable=False, index=True)
    session_id = Column(String, index=True)  # Anonymous session identifier
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)  # Click timestamp
    ip_hash = Column(String, index=True)  # Hashed IP for fraud detection
    user_agent_hash = Column(String)  # Hashed user agent
    referring_url = Column(String)  # Source of click

    # Relationship to ad
    ad = relationship("Ad")

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_click_tracking_ad_id_timestamp", "ad_id", "timestamp"),
        Index("ix_click_tracking_session_id_timestamp", "session_id", "timestamp"),
    )


class ConversionTracking(Base):
    """
    Conversion Tracking table for recording ad conversions
    """

    __tablename__ = "conversion_tracking"

    id = Column(Integer, primary_key=True, index=True)
    click_id = Column(
        Integer, ForeignKey("click_tracking.id"), nullable=False, index=True
    )  # Foreign key to click tracking
    conversion_type = Column(String, index=True)  # Purchase, signup, download, etc.
    value = Column(Float)  # Conversion value (if applicable)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)  # Conversion timestamp
    status = Column(String, default="pending", index=True)  # Verified, pending, rejected

    # Relationship to click
    click = relationship("ClickTracking")

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_conversion_tracking_click_id_timestamp", "click_id", "timestamp"),
        Index("ix_conversion_tracking_type_status", "conversion_type", "status"),
    )


class FraudDetection(Base):
    """
    Fraud Detection table for tracking suspicious activities
    """

    __tablename__ = "fraud_detection"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer)  # ID of suspicious event
    event_type = Column(String, index=True)  # Click, impression, conversion
    reason = Column(String)  # Reason for flagging
    severity = Column(String, index=True)  # Low, medium, high risk
    review_status = Column(String, default="pending", index=True)  # Pending, reviewed, action_taken
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)  # Timestamp

    # Relationship to ad (optional, could be linked to ad_id if needed)
    ad_id = Column(Integer, ForeignKey("ads.id"), nullable=True, index=True)
    ad = relationship("Ad")

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_fraud_detection_event_type_severity", "event_type", "severity"),
        Index("ix_fraud_detection_review_status_created", "review_status", "created_at"),
    )


# A/B Testing Tables
class ABTest(Base):
    """
    A/B Test configuration table
    """

    __tablename__ = "ab_tests"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), index=True)
    status = Column(String, default=ABTestStatus.DRAFT.value, index=True)  # draft, running, paused, completed, cancelled
    start_date = Column(DateTime(timezone=True), index=True)
    end_date = Column(DateTime(timezone=True), index=True)
    traffic_allocation = Column(Float, default=1.0)  # Percentage of traffic to include in test
    min_sample_size = Column(Integer, default=1000)  # Minimum samples before significance
    confidence_level = Column(Float, default=0.95)  # Statistical confidence level
    primary_metric = Column(String, default="ctr")  # Primary metric to optimize
    winner_variant_id = Column(Integer, nullable=True)  # ID of winning variant (if determined)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)

    # Relationships
    variants = relationship("ABTestVariant", back_populates="test", cascade="all, delete-orphan")

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_ab_tests_campaign_id_status", "campaign_id", "status"),
        Index("ix_ab_tests_status_dates", "status", "start_date", "end_date"),
    )


class ABTestVariant(Base):
    """
    A/B Test variant table
    """

    __tablename__ = "ab_test_variants"

    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(Integer, ForeignKey("ab_tests.id"), nullable=False, index=True)
    name = Column(String, nullable=False, index=True)  # e.g., "Control", "Variant A"
    ad_id = Column(Integer, ForeignKey("ads.id"), index=True)  # The ad to show for this variant
    traffic_weight = Column(Float, default=0.5)  # Weight for traffic splitting
    is_control = Column(Boolean, default=False, index=True)  # Is this the control variant
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    revenue = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    test = relationship("ABTest", back_populates="variants")

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_ab_test_variants_test_id_ad_id", "test_id", "ad_id"),
        Index("ix_ab_test_variants_is_control", "is_control"),
    )


class ABTestAssignment(Base):
    """
    Tracks which variant a user was assigned to
    """

    __tablename__ = "ab_test_assignments"

    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(Integer, ForeignKey("ab_tests.id"), nullable=False, index=True)
    variant_id = Column(Integer, ForeignKey("ab_test_variants.id"), nullable=False, index=True)
    user_hash = Column(String, nullable=False, index=True)  # Hashed user/session identifier
    assigned_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_ab_test_assignments_test_id_user_hash", "test_id", "user_hash"),
        Index("ix_ab_test_assignments_variant_id", "variant_id"),
    )


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
        """Initialize database tables"""
        try:
            Base.metadata.create_all(bind=engine, checkfirst=True)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.warning(f"Table creation warning (safe to ignore if tables exist): {str(e)}")

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
        Dependency to get database session (generator version for FastAPI Depends)
        """
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def get_db_session(self):
        """
        Context manager for database sessions.

        Usage:
            with db_manager.get_db_session() as db:
                # use db session
        """
        from contextlib import contextmanager

        @contextmanager
        def session_context():
            db = SessionLocal()
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

        return session_context()

    def cleanup_expired_metrics(self):
        """
        Delete expired metrics from the database
        """
        db = SessionLocal()
        try:
            # Find and delete expired metrics
            expired_before = datetime.now(UTC)
            deleted_count = db.query(AdMetric).filter(AdMetric.expires_at < expired_before).delete()
            db.commit()
            logger.info(f"Cleaned up {deleted_count} expired ad metrics")
        except Exception as e:
            logger.error(f"Error cleaning up expired metrics: {e}")
            db.rollback()
        finally:
            db.close()


# Initialize the database manager
db_manager = DatabaseManager()
