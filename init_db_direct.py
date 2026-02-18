#!/usr/bin/env python
"""Initialize database tables directly without PgBouncer"""

import os
os.environ['DATABASE_URL'] = 'postgresql://intent_user:intent_secure_password_change_in_prod@postgres:5432/intent_engine'

from database import Base, engine

print(f"Connecting to: {engine.url}")
Base.metadata.create_all(bind=engine)
print("Tables created successfully!")
