"""
Intent Engine - Core Utilities

This module contains shared utilities used across all components of the Intent Engine.
"""

import logging
import re
from datetime import UTC, datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_price_range(text: str) -> tuple | None:
    """
    Extract price range from text using regex patterns
    Returns (operator, value) tuple or None
    """
    # Pattern to match prices like "under 100", "less than 50", etc.
    patterns = [
        r"\b(under|less than|below)\s*(\d+)\s*(?:rupees|rs|₹|dollars?|usd)?\b",
        r"\b(over|more than|above)\s*(\d+)\s*(?:rupees|rs|₹|dollars?|usd)?\b",
        r"\b(above|greater than)\s*(\d+)\s*(?:rupees|rs|₹|dollars?|usd)?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            operator_word = match.group(1)
            value = int(match.group(2))

            # Map operator words to symbols
            if operator_word in ["under", "less than", "below"]:
                return ("<=", value)
            elif operator_word in ["over", "more than", "above", "greater than"]:
                return (">=", value)

    return None


def normalize_datetime(dt_str: str) -> datetime:
    """
    Normalize datetime string to a consistent format
    """
    try:
        # Handle ISO format with Z suffix
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"

        # Parse the datetime
        if "+" in dt_str or dt_str.count("-") > 2:  # Has timezone info
            return datetime.fromisoformat(dt_str)
        else:
            # Naive datetime, treat as UTC
            return datetime.fromisoformat(dt_str).replace(tzinfo=UTC)
    except ValueError:
        # If parsing fails, return current time
        return datetime.now(UTC)


def calculate_expiration_time(hours: int = 8) -> str:
    """
    Calculate expiration time (default 8 hours from now)
    """
    expires_at = (datetime.utcnow() + timedelta(hours=hours)).isoformat() + "Z"
    return expires_at


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks
    """
    # Remove potentially dangerous characters/sequences
    sanitized = re.sub(r'[<>"\';]', "", text)
    return sanitized.strip()


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format
    """
    return datetime.utcnow().isoformat() + "Z"


def fuzzy_match(text1: str, text2: str, threshold: float = 0.7) -> bool:
    """
    Perform fuzzy matching between two texts
    """
    # Simple token-based similarity
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if not union:
        return False

    similarity = len(intersection) / len(union)
    return similarity >= threshold
