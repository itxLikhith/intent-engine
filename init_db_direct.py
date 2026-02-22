#!/usr/bin/env python
"""Initialize database tables directly without PgBouncer"""

import os
import sys

# ---------------------------------------------------------
# CRITICAL FIX: Override connection to bypass PgBouncer
# ---------------------------------------------------------
# We must connect directly to 'postgres:5432' because PgBouncer (6543)
# in transaction mode causes issues with DDL (Drop/Create) sequences.
print("Configuring direct database connection for initialization...")

# Extract credentials or use defaults matching docker-compose
user = os.getenv("POSTGRES_USER", "intent_user")
password = os.getenv("POSTGRES_PASSWORD", "intent_secure_password_change_in_prod")
db = os.getenv("POSTGRES_DB", "intent_engine")
host = "postgres"  # The actual service name, NOT pgbouncer
port = "5432"  # The direct port, NOT 6543

# Force the environment variable BEFORE importing 'database'
os.environ["DATABASE_URL"] = f"postgresql://{user}:{password}@{host}:{port}/{db}"

# ---------------------------------------------------------

try:
    # Now import Base and engine. They will see the new DATABASE_URL
    from database import Base, engine

    # Import models explicitly to ensure they are registered with Base
    # (Adjust this import based on your project structure, e.g. 'from app import models')
    try:
        import models
    except ImportError:
        pass

    print(f"Connecting to: {engine.url}")

    print("1. Dropping all existing tables...")
    Base.metadata.drop_all(bind=engine)

    print("2. Creating all tables...")
    Base.metadata.create_all(bind=engine)

    print("✅ Database initialized successfully!")

except Exception as e:
    print(f"❌ Error initializing database: {e}")
    sys.exit(1)
