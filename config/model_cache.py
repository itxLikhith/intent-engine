"""
Model Cache - Pre-loads and caches ML models for faster startup.

This module provides model initialization and caching to avoid cold starts.
"""

import logging

logger = logging.getLogger(__name__)


def initialize_models():
    """
    Pre-load and cache all ML models used by the Intent Engine.

    This should be called once at startup to avoid cold-start latency
    on the first request.
    """
    logger.info("Initializing ML models...")

    try:
        from core.embedding_service import get_embedding_service

        service = get_embedding_service()
        service.initialize(use_redis=False)
        logger.info("Embedding service initialized")
    except Exception as e:
        logger.warning(f"Could not initialize embedding service: {e}")

    try:
        from extraction.extractor import get_intent_extractor

        get_intent_extractor()
        logger.info("Intent extractor initialized")
    except Exception as e:
        logger.warning(f"Could not initialize intent extractor: {e}")

    try:
        from ranking.ranker import get_intent_ranker

        get_intent_ranker()
        logger.info("Intent ranker initialized")
    except Exception as e:
        logger.warning(f"Could not initialize intent ranker: {e}")

    logger.info("Model initialization complete")
