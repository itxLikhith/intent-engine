"""
Intent Engine - Custom Exception Classes

This module defines custom exception classes for consistent error handling
across the Intent Engine API.
"""

from typing import Any


class IntentEngineException(Exception):
    """Base exception for all Intent Engine errors"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


# Intent Extraction Exceptions
class IntentExtractionError(IntentEngineException):
    """Error during intent extraction"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=500, details=details)


class InvalidIntentError(IntentEngineException):
    """Invalid intent data provided"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=400, details=details)


class IntentExpiredError(IntentEngineException):
    """Intent has expired"""

    def __init__(self, message: str = "Intent has expired", details: dict[str, Any] | None = None):
        super().__init__(message, status_code=400, details=details)


# Ranking Exceptions
class RankingError(IntentEngineException):
    """Error during ranking process"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=500, details=details)


class InvalidRankingInputError(IntentEngineException):
    """Invalid input for ranking"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=400, details=details)


# Service Recommendation Exceptions
class ServiceRecommendationError(IntentEngineException):
    """Error during service recommendation"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=500, details=details)


# Ad Matching Exceptions
class AdMatchingError(IntentEngineException):
    """Error during ad matching"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=500, details=details)


class NoMatchingAdsError(IntentEngineException):
    """No ads matched the criteria"""

    def __init__(
        self,
        message: str = "No matching ads found",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, status_code=404, details=details)


class AdFairnessViolationError(IntentEngineException):
    """Ad violates fairness constraints"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=400, details=details)


# Database Exceptions
class DatabaseError(IntentEngineException):
    """Database operation failed"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=500, details=details)


class NotFoundError(IntentEngineException):
    """Requested resource not found"""

    def __init__(self, resource: str, resource_id: Any, details: dict[str, Any] | None = None):
        message = f"{resource} with ID {resource_id} not found"
        super().__init__(
            message,
            status_code=404,
            details=details or {"resource": resource, "id": resource_id},
        )


class ValidationError(IntentEngineException):
    """Data validation failed"""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if field:
            details["field"] = field
        super().__init__(message, status_code=422, details=details)


# External Service Exceptions
class ExternalServiceError(IntentEngineException):
    """External service (Redis, SearXNG) failed"""

    def __init__(self, service: str, message: str, details: dict[str, Any] | None = None):
        details = details or {}
        details["service"] = service
        super().__init__(message, status_code=503, details=details)


class ServiceUnavailableError(IntentEngineException):
    """Required external service is unavailable"""

    def __init__(self, service: str, message: str | None = None):
        message = message or f"Service {service} is unavailable"
        super().__init__(message, status_code=503, details={"service": service})


# Privacy & Compliance Exceptions
class PrivacyViolationError(IntentEngineException):
    """Privacy policy violation detected"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=400, details=details)


class ConsentRequiredError(IntentEngineException):
    """User consent required but not obtained"""

    def __init__(self, consent_type: str, details: dict[str, Any] | None = None):
        message = f"Consent required for: {consent_type}"
        super().__init__(message, status_code=403, details=details or {"consent_type": consent_type})


# Fraud Detection Exceptions
class FraudDetectedError(IntentEngineException):
    """Fraudulent activity detected"""

    def __init__(self, reason: str, severity: str = "high", details: dict[str, Any] | None = None):
        details = details or {}
        details["reason"] = reason
        details["severity"] = severity
        super().__init__(f"Fraud detected: {reason}", status_code=403, details=details)


# Campaign Management Exceptions
class CampaignError(IntentEngineException):
    """Campaign operation failed"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=500, details=details)


class BudgetExceededError(IntentEngineException):
    """Campaign budget exceeded"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=400, details=details)


# A/B Testing Exceptions
class ABTestError(IntentEngineException):
    """A/B test operation failed"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=500, details=details)


class ABTestNotFoundError(NotFoundError):
    """A/B test not found"""

    def __init__(self, test_id: Any):
        super().__init__("A/B test", test_id)


# Exception handler for FastAPI
def register_exception_handlers(app):
    """
    Register exception handlers with FastAPI app.

    Args:
        app: FastAPI application instance
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.exception_handler(IntentEngineException)
    async def intent_engine_exception_handler(request: Request, exc: IntentEngineException):
        """Handle all Intent Engine exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": exc.__class__.__name__,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "details": {},
                }
            },
        )
