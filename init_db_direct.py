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

os.environ["DATABASE_URL"] = f"postgresql://{user}:{password}@{host}:{port}/{db}"

# ---------------------------------------------------------
# 2. Perform Operations
# ---------------------------------------------------------
try:
    from database import Base, engine

    # Import models explicitly so Base knows what to create
    try:
        import models
    except ImportError:
        pass

    print(f"Connecting to: {engine.url}")

    # FORCE CLEAN: Drop the entire schema instead of individual tables.
    # This handles cases where migrations created tables that SQLAlchemy doesn't know about.
    print("1. Wiping database schema (Nuclear Option)...")
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))
        conn.commit()
        print("   Schema wiped successfully.")

    print("2. Creating tables from Python models...")
    Base.metadata.create_all(bind=engine)

    print("✅ Database initialized successfully!")

except Exception as e:
    print(f"❌ Error initializing database: {e}")
    sys.exit(1)
