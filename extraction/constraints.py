"""
Intent Engine - Constraint Parsing Logic

This module handles constraint parsing and validation logic.
"""

import re
from typing import List

from ..core.schema import Constraint, ConstraintType


class ConstraintExtractor:
    """
    Extracts constraints from user queries using rule-based regex patterns
    """

    def __init__(self):
        # Platform constraints
        self.platform_patterns = {
            r"\b(android|mobile|phone)\b": ("platform", "Android"),
            r"\b(ios|iphone|ipad|apple)\b": ("platform", "iOS"),
            r"\b(windows|pc|desktop)\b": ("platform", "Windows"),
            r"\b(mac|macos|macbook)\b": ("platform", "macOS"),
            r"\b(linux|ubuntu|debian|fedora)\b": ("platform", "Linux"),
            r"\b(web|browser|chrome|firefox|safari)\b": ("platform", "Web"),
        }

        # Provider exclusion constraints
        self.exclusion_patterns = {
            r"\b(no\s+(google|gmail|android))\b": ("provider", "Google"),
            r"\b(no\s+(microsoft|outlook|windows))\b": ("provider", "Microsoft"),
            r"\b(no\s+(apple|ios|iphone|ipad))\b": ("provider", "Apple"),
            r"\b(no\s+(big\s+tech|big\s+corporations?))\b": (
                "provider",
                ["Google", "Microsoft", "Apple", "Amazon", "Meta"],
            ),
            r"\b(not?\s+google|avoid\s+google)\b": ("provider", "Google"),
            r"\b(not?\s+microsoft|avoid\s+microsoft)\b": ("provider", "Microsoft"),
            r"\b(proprietary|closed\s+source)\b": ("license", "proprietary"),
            r"\b(open\s+source|oss|free\s+software)\b": ("license", "open-source"),
        }

        # Price constraints
        self.price_patterns = {
            r"\b(under|less than|below)\s*(\d+)\s*(rupees|rs|₹|dollars?|usd)\b": ("price", "<={value}"),
            r"\b(over|more than|above)\s*(\d+)\s*(rupees|rs|₹|dollars?|usd)\b": ("price", ">={value}"),
            r"\b(free|gratis|no cost|zero cost)\b": ("price", "0"),
            r"\b(budget|cheap|affordable|low cost)\b": ("price", "budget"),
        }

        # Feature constraints
        self.feature_patterns = {
            r"\b(end[-\s]*to[-\s]*end[-\s]*encrypt|e2e[-\s]*encrypt|end[-\s]*to[-\s]*end[-\s]*encrypted|e2e[-\s]*encrypted)\b": (
                "feature",
                "end-to-end_encryption",
            ),
            r"\b(end[-\s]*to[-\s]*end|e2e)\s+(encrypt|encryption)\b": ("feature", "end-to-end_encryption"),
            r"\b(encrypted|secure|private)\s+(email|mail)\b": ("feature", "encrypted_email"),
            r"\b(real[-\s]*time|instant)\s+(sync|collaboration)\b": ("feature", "real-time_collaboration"),
            r"\b(offline|local|on[-\s]*device)\s+(storage|sync)\b": ("feature", "offline_capability"),
            r"\b(ad[-\s]*free|no\s+ads|ad[-\s]*less)\b": ("feature", "ad-free"),
        }

    def extract_constraints(self, text: str) -> List[Constraint]:
        """
        Extract constraints from the input text using regex patterns
        """
        constraints = []
        seen_constraints = set()  # To avoid duplicates
        text_lower = text.lower()

        # Extract platform constraints
        for pattern, (dimension, value) in self.platform_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                constraint_key = (ConstraintType.INCLUSION.value, dimension, str(value))
                if constraint_key not in seen_constraints:
                    constraints.append(
                        Constraint(type=ConstraintType.INCLUSION, dimension=dimension, value=value, hardFilter=True)
                    )
                    seen_constraints.add(constraint_key)

        # Extract exclusion constraints
        for pattern, (dimension, value) in self.exclusion_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                constraint_key = (ConstraintType.EXCLUSION.value, dimension, str(value))
                if constraint_key not in seen_constraints:
                    constraints.append(
                        Constraint(type=ConstraintType.EXCLUSION, dimension=dimension, value=value, hardFilter=True)
                    )
                    seen_constraints.add(constraint_key)

        # Extract price constraints
        for pattern, (dimension, template) in self.price_patterns.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    value_str = match[1]  # The numeric value
                    actual_template = template.format(value=value_str)
                    constraint_key = (ConstraintType.RANGE.value, dimension, actual_template)
                    if constraint_key not in seen_constraints:
                        constraints.append(
                            Constraint(
                                type=ConstraintType.RANGE, dimension=dimension, value=actual_template, hardFilter=True
                            )
                        )
                        seen_constraints.add(constraint_key)

        # Extract feature constraints - need to handle overlaps carefully
        # Only allow one feature constraint per feature type to avoid duplicates
        feature_types_seen = set()
        for pattern, (dimension, value) in self.feature_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches and dimension not in feature_types_seen:
                constraint_key = (ConstraintType.INCLUSION.value, dimension, str(value))
                if constraint_key not in seen_constraints:
                    constraints.append(
                        Constraint(type=ConstraintType.INCLUSION, dimension=dimension, value=value, hardFilter=True)
                    )
                    seen_constraints.add(constraint_key)
                    feature_types_seen.add(dimension)

        return constraints

    def extract_negative_preferences(self, text: str) -> List[str]:
        """
        Extract negative preferences from the text
        """
        text_lower = text.lower()
        negative_prefs = []

        # Look for common negative preference patterns
        if re.search(r"\b(no\s+big\s+tech|no\s+big\s+corporations?)\b", text_lower):
            negative_prefs.append("no big tech")
        if re.search(r"\b(no\s+proprietary|no\s+closed\s+source|open\s+source)\b", text_lower):
            negative_prefs.append("no proprietary")
        if re.search(r"\b(privacy[-\s]*first|privacy\s+focused)\b", text_lower):
            negative_prefs.append("privacy-focused")

        return negative_prefs
