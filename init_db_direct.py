#!/usr/bin/env python
"""Initialize database tables directly without PgBouncer"""

import os

if not os.getenv("DATABASE_URL"):
    host = os.getenv("POSTGRES_HOST", "postgres")
    user = os.getenv("POSTGRES_USER", "intent_user")
    password = os.getenv("POSTGRES_PASSWORD", "intent_secure_password_change_in_prod")
    db = os.getenv("POSTGRES_DB", "intent_engine")
    os.environ["DATABASE_URL"] = f"postgresql://{user}:{password}@{host}:5432/{db}"

from database import Base, engine

print(f"Connecting to: {engine.url}")
Base.metadata.create_all(bind=engine)
print("Tables created successfully!")
