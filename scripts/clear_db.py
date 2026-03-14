#!/usr/bin/env python
"""Clear all data from database tables (PostgreSQL)"""

from sqlalchemy import text

from database import SessionLocal

db = SessionLocal()
try:
    # Delete all data from all tables (in order to respect foreign keys)
    # Using CASCADE to handle foreign key dependencies
    tables = [
        "ab_test_assignments",
        "ab_test_variants",
        "ab_tests",
        "fraud_detection",
        "conversion_tracking",
        "click_tracking",
        "ad_metrics",
        "creative_assets",
        "ads",
        "ad_groups",
        "campaigns",
        "advertisers",
        "audit_trails",
        "user_consents",
    ]

    for table in tables:
        try:
            db.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
            print(f"Cleared: {table}")
        except Exception as e:
            print(f"Skipping {table}: {e}")

    db.commit()
    print("\nAll data cleared successfully!")

finally:
    db.close()
