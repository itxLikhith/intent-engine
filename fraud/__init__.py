"""
Fraud Detection Module

Provides comprehensive fraud detection for ad clicks, impressions, and conversions.
"""

from .detector import (
    FraudAnalysisResult,
    FraudDetector,
    FraudSeverity,
    FraudSignal,
    FraudType,
    get_fraud_detector,
)

__all__ = [
    "FraudDetector",
    "FraudAnalysisResult",
    "FraudSignal",
    "FraudSeverity",
    "FraudType",
    "get_fraud_detector",
]
