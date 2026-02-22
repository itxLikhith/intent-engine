#!/usr/bin/env python
"""Initialize database tables directly without PgBouncer"""

import os
import sys

# Ensure we use the direct Postgres port (5432) for DDL operations,
# not PgBouncer (6543) which doesn't support "PREPARE" statements well in some configs
if not os.getenv("DATABASE_URL"):
    host = os.getenv("POSTGRES_HOST", "postgres")
    user = os.getenv("POSTGRES_USER", "intent_user")
    password = os.getenv("POSTGRES_PASSWORD", "intent_secure_password_change_in_prod")
    db = os.getenv("POSTGRES_DB", "intent_engine")
    os.environ["DATABASE_URL"] = f"postgresql://{user}:{password}@{host}:5432/{db}"

from database import Base, engine

# IMPORTANT: Import models here so SQLAlchemy knows they exist before creating tables.
# Adjust 'models' to match your actual file name (e.g., 'app.models' or just 'models')
try:
    import models
except ImportError:
    # If models are defined inside database.py or imported there, this passes safely
    pass

print(f"Connecting to: {engine.url}")

try:
    # 1. DROP ALL TABLES
    # This fixes the "DuplicateTable" error by cleaning up whatever the 'migrations' container did.
    print("Dropping existing tables to ensure clean state...")
    Base.metadata.drop_all(bind=engine)

    # 2. CREATE ALL TABLES
    # This fixes the "UndefinedTable" error by creating tables exactly as defined in your code.
    print("Creating tables from SQLAlchemy models...")
    Base.metadata.create_all(bind=engine)

    print("Tables initialized successfully!")

except Exception as e:
    print(f"Error initializing database: {e}")
    sys.exit(1)
