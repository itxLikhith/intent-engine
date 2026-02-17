"""
SearXNG Integration Module

This module provides integration between the Intent Engine and SearXNG privacy search.
"""

from .client import SearXNGClient, SearXNGResult, SearXNGResponse, get_searxng_client

__all__ = [
    "SearXNGClient",
    "SearXNGResult",
    "SearXNGResponse",
    "get_searxng_client",
]
