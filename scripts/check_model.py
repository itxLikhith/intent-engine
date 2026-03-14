#!/usr/bin/env python
"""Check CreativeAsset model"""

import inspect

from database import CreativeAsset

print("CreativeAsset columns:")
for col in CreativeAsset.__table__.columns:
    print(f"  - {col.name}: {col.type}")

print("\nCreativeAsset __init__ signature:")
print(inspect.signature(CreativeAsset.__init__))
