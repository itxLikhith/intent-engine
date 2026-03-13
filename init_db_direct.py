#!/usr/bin/env python
"""Initialize database tables directly without PgBouncer"""

import os
import sys

from sqlalchemy import text

# ---------------------------------------------------------
# 1. Force Direct Connection (Bypass PgBouncer)
# ---------------------------------------------------------
print("Configuring direct database connection...")
user = os.getenv("POSTGRES_USER", "intent_user")
password = os.getenv("POSTGRES_PASSWORD", "intent_secure_password_change_in_prod")
db = os.getenv("POSTGRES_DB", "intent_engine")
host = "postgres"  # Direct container name
port = "5432"  # Direct port

DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}"
os.environ["DATABASE_URL"] = DATABASE_URL

# ---------------------------------------------------------
# 2. Perform Operations
# ---------------------------------------------------------
try:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    # Create a fresh engine for schema operations
    print("1. Wiping database schema (Nuclear Option)...")
    clean_engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        isolation_level="AUTOCOMMIT",
    )
    with clean_engine.connect() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.commit()
    clean_engine.dispose()
    print("   Schema wiped successfully.")

    # Clear any cached SQLAlchemy metadata to avoid index conflicts
    import database

    database.Base.metadata.clear()

    # Import models AFTER schema is wiped to register them with Base
    # Import all SQLAlchemy ORM models explicitly so Base knows what tables to create
    from database import (
        Base,
    )

    # Create fresh engine for table creation with explicit metadata clearing
    print("2. Creating tables from Python models...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, poolclass=NullPool)
    print(f"   Connecting to: {engine.url}")
    Base.metadata.create_all(bind=engine)

    # Verify all tables were created
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"   Tables created: {', '.join(sorted(tables))}")

    # Verify creative_assets table exists
    if "creative_assets" not in tables:
        print("   ⚠️  WARNING: creative_assets table was not created!")
        sys.exit(1)

    print("✅ Database initialized successfully!")

except Exception as e:
    print(f"❌ Error initializing database: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
