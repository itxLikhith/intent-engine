#!/usr/bin/env python
"""Read seed_sample_data.py lines 250-265"""

with open("seed_sample_data.py") as f:
    lines = f.readlines()

for i, line in enumerate(lines[250:265], start=251):
    print(f"{i}: {line}", end="")
