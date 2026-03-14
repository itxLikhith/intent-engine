#!/usr/bin/env python
"""Initialize database tables using a fresh SQLAlchemy setup - avoids caching issues"""

import os
import sys

# Configure database URL directly
user = os.getenv("POSTGRES_USER", "intent_user")
password = os.getenv("POSTGRES_PASSWORD", "intent_secure_password_change_in_prod")
db = os.getenv("POSTGRES_DB", "intent_engine")
host = "postgres"
port = "5432"

DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}"

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
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import func

print("1. Wiping database schema...")
engine = create_engine(DATABASE_URL, poolclass=NullPool, isolation_level="AUTOCOMMIT")
with engine.connect() as conn:
    conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
    conn.commit()
conn.close()
engine.dispose()


print("   Recreating schema...")
engine = create_engine(DATABASE_URL, poolclass=NullPool, isolation_level="AUTOCOMMIT")
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA public"))
    conn.execute(text(f"GRANT ALL ON SCHEMA public TO {user}"))
    conn.execute(text(f"GRANT ALL ON ALL TABLES IN SCHEMA public TO {user}"))
    conn.execute(text(f"GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO {user}"))
    conn.commit()
conn.close()
engine.dispose()
print("   Schema wiped successfully.")

print("2. Creating tables...")
from sqlalchemy.orm import clear_mappers

clear_mappers()
metadata = MetaData()
metadata.clear()

# Define all tables inline to avoid any caching issues
tables = {}

# Advertisers
tables["advertisers"] = Table(
    "advertisers",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("name", String, nullable=False),
    Column("contact_email", String),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

# Campaigns
tables["campaigns"] = Table(
    "campaigns",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("advertiser_id", Integer, ForeignKey("advertisers.id"), nullable=False, index=True),
    Column("name", String, nullable=False, index=True),
    Column("start_date", DateTime(timezone=True), index=True),
    Column("end_date", DateTime(timezone=True), index=True),
    Column("budget", Float, default=0.0),
    Column("daily_budget", Float, default=0.0),
    Column("status", String, default="active", index=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), index=True),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now()),
    Index("ix_campaigns_advertiser_id_status", "advertiser_id", "status"),
    Index("ix_campaigns_status_dates", "status", "start_date", "end_date"),
)

# AdGroups
tables["ad_groups"] = Table(
    "ad_groups",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("campaign_id", Integer, ForeignKey("campaigns.id"), nullable=False, index=True),
    Column("name", String, nullable=False, index=True),
    Column("targeting_settings", JSON),
    Column("bid_strategy", String, default="manual"),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), index=True),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now()),
    Index("ix_ad_groups_campaign_id_bid_strategy", "campaign_id", "bid_strategy"),
)

# Ads
tables["ads"] = Table(
    "ads",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("advertiser_id", Integer, ForeignKey("advertisers.id"), nullable=False, index=True),
    Column("ad_group_id", Integer, ForeignKey("ad_groups.id"), index=True),
    Column("title", String, nullable=False),
    Column("description", Text),
    Column("url", String, nullable=False),
    Column("targeting_constraints", JSON),
    Column("ethical_tags", JSON),
    Column("quality_score", Float, default=0.5),
    Column("creative_format", String),
    Column("bid_amount", Float, default=0.0),
    Column("status", String, default="active", index=True),
    Column("approval_status", String, default="pending"),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), index=True),
    Index("ix_ads_advertiser_status", "advertiser_id", "status"),
    Index("ix_ads_approval_status", "approval_status", "status"),
)

# CreativeAssets
tables["creative_assets"] = Table(
    "creative_assets",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("ad_id", Integer, ForeignKey("ads.id"), nullable=False, index=True),
    Column("asset_type", String, nullable=False, index=True),
    Column("asset_url", String, nullable=False),
    Column("payload", JSON),
    Column("checksum", String),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), index=True),
    Index("ix_creative_assets_ad_id_asset_type", "ad_id", "asset_type"),
)

# AdMetrics
tables["ad_metrics"] = Table(
    "ad_metrics",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("ad_id", Integer, ForeignKey("ads.id"), nullable=False, index=True),
    Column("date", Date, nullable=False, index=True),
    Column("intent_goal", String),
    Column("intent_use_case", String),
    Column("impression_count", Integer, default=0),
    Column("click_count", Integer, default=0),
    Column("conversion_count", Integer, default=0),
    Column("ctr", Float),
    Column("cpm", Float),
    Column("cpc", Float),
    Column("roas", Float),
    Column("engagement_rate", Float),
    Column("spend", Float, default=0.0),
    Column("revenue", Float, default=0.0),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), index=True),
    Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
    Index("ix_ad_metrics_ad_id_date", "ad_id", "date"),
    Index("ix_ad_metrics_date_goal", "date", "intent_goal"),
)

# ClickTracking
tables["click_tracking"] = Table(
    "click_tracking",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("ad_id", Integer, ForeignKey("ads.id"), nullable=False, index=True),
    Column("session_id", String, index=True),
    Column("timestamp", DateTime(timezone=True), server_default=func.now(), index=True),
    Column("ip_hash", String, index=True),
    Column("user_agent_hash", String),
    Column("referring_url", String),
    Column("payload", JSON),
    Index("ix_click_tracking_ad_id_timestamp", "ad_id", "timestamp"),
    Index("ix_click_tracking_session_id_timestamp", "session_id", "timestamp"),
)

# ConversionTracking
tables["conversion_tracking"] = Table(
    "conversion_tracking",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("click_id", Integer, ForeignKey("click_tracking.id"), nullable=False, index=True),
    Column("conversion_type", String, index=True),
    Column("value", Float),
    Column("timestamp", DateTime(timezone=True), server_default=func.now(), index=True),
    Column("status", String, default="pending", index=True),
    Column("payload", JSON),
    Index("ix_conversion_tracking_click_id_timestamp", "click_id", "timestamp"),
    Index("ix_conversion_tracking_type_status", "conversion_type", "status"),
)

# FraudDetection
tables["fraud_detection"] = Table(
    "fraud_detection",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("event_id", Integer),
    Column("event_type", String, index=True),
    Column("reason", String),
    Column("severity", String, index=True),
    Column("review_status", String, default="pending", index=True),
    Column("ad_id", Integer, ForeignKey("ads.id"), nullable=True, index=True),
    Column("payload", JSON),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), index=True),
    Index("ix_fraud_detection_event_type_severity", "event_type", "severity"),
    Index("ix_fraud_detection_review_status_created", "review_status", "created_at"),
)

# ABTests
tables["ab_tests"] = Table(
    "ab_tests",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("name", String, nullable=False),
    Column("description", Text),
    Column("campaign_id", Integer, ForeignKey("campaigns.id"), index=True),
    Column("status", String, default="draft", index=True),
    Column("start_date", DateTime(timezone=True), index=True),
    Column("end_date", DateTime(timezone=True), index=True),
    Column("traffic_allocation", Float, default=1.0),
    Column("min_sample_size", Integer, default=1000),
    Column("confidence_level", Float, default=0.95),
    Column("primary_metric", String, default="ctr"),
    Column("winner_variant_id", Integer, nullable=True),
    Column("payload", JSON),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), index=True),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True),
    Index("ix_ab_tests_campaign_id_status", "campaign_id", "status"),
    Index("ix_ab_tests_status_dates", "status", "start_date", "end_date"),
)

# ABTestVariants
tables["ab_test_variants"] = Table(
    "ab_test_variants",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("test_id", Integer, ForeignKey("ab_tests.id"), nullable=False, index=True),
    Column("name", String, nullable=False, index=True),
    Column("ad_id", Integer, ForeignKey("ads.id"), index=True),
    Column("traffic_weight", Float, default=0.5),
    Column("is_control", Boolean, default=False),
    Column("impressions", Integer, default=0),
    Column("clicks", Integer, default=0),
    Column("conversions", Integer, default=0),
    Column("revenue", Float, default=0.0),
    Column("payload", JSON),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), index=True),
    Index("ix_ab_test_variants_test_id_ad_id", "test_id", "ad_id"),
    Index("ix_ab_test_variants_is_control", "is_control"),
)

# ABTestAssignments
tables["ab_test_assignments"] = Table(
    "ab_test_assignments",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("test_id", Integer, ForeignKey("ab_tests.id"), nullable=False, index=True),
    Column("variant_id", Integer, ForeignKey("ab_test_variants.id"), nullable=False),
    Column("user_hash", String, nullable=False, index=True),
    Column("assigned_at", DateTime(timezone=True), server_default=func.now(), index=True),
    Index("ix_ab_test_assignments_test_id_user_hash", "test_id", "user_hash"),
    Index("ix_ab_test_assignments_variant_id", "variant_id"),
)

# UserConsents
tables["user_consents"] = Table(
    "user_consents",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("user_id", String, nullable=False),
    Column("consent_type", String, nullable=False),
    Column("granted", Boolean, nullable=False),
    Column("consent_details", JSON),
    Column("granted_at", DateTime(timezone=True), server_default=func.now()),
    Column("expires_at", DateTime(timezone=True)),
    Column("withdrawn_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    Index("ix_user_consents_user_id_consent_type", "user_id", "consent_type"),
)

# AuditTrail
tables["audit_trails"] = Table(
    "audit_trails",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("user_id", String),
    Column("event_type", String, nullable=False),
    Column("resource_type", String),
    Column("resource_id", Integer),
    Column("action_description", Text),
    Column("ip_address", String),
    Column("user_agent", String),
    Column("payload", JSON),
    Column("timestamp", DateTime(timezone=True), server_default=func.now(), index=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Index("ix_audit_trails_event_type", "event_type"),
    Index("ix_audit_trails_resource_type_id", "resource_type", "resource_id"),
)

# Create all tables
engine = create_engine(DATABASE_URL, poolclass=NullPool, pool_pre_ping=True)
metadata.create_all(engine)

# Verify
inspector = inspect(engine)
tables_created = sorted(inspector.get_table_names())
print(f"   Tables created: {', '.join(tables_created)}")

# Verify creative_assets exists
if "creative_assets" in tables_created and "audit_trails" in tables_created:
    print("✅ Database initialized successfully!")
else:
    print(
        f"   ⚠️  WARNING: missing tables! creative_assets: {'creative_assets' in tables_created}, audit_trails: {'audit_trails' in tables_created}"
    )
    sys.exit(1)
