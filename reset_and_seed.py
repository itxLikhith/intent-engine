#!/usr/bin/env python
"""Reset database and seed sample data"""

import subprocess
import sys

print("=" * 60)
print("Resetting database and seeding sample data")
print("=" * 60)

# Run init_db_direct.py to reset and create tables
print("\n[1/2] Initializing database...")
result = subprocess.run([sys.executable, "init_db_direct.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"Error initializing database: {result.stderr}")
    sys.exit(1)

# Run seed_sample_data.py
print("\n[2/2] Seeding sample data...")
result = subprocess.run([sys.executable, "seed_sample_data.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"Error seeding data: {result.stderr}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Database reset and seed completed successfully!")
print("=" * 60)
