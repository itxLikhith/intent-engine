"""
Intent Engine - Privacy and Compliance Module

This module implements privacy and compliance features for the Intent Engine,
ensuring user data is handled appropriately and in compliance with regulations.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from ads.matcher import AdMetadata
from core.schema import UniversalIntent


class PrivacyComplianceEngine:
    """
    Engine to handle privacy and compliance checks
    """

    def __init__(self):
        # Define forbidden targeting dimensions that violate privacy
        self.forbidden_targeting_dimensions = {
            "age",
            "gender",
            "income",
            "race",
            "ethnicity",
            "political_affiliation",
            "religious_belief",
            "health_condition",
            "sexual_orientation",
            "precise_location",
            "behavior",
            "interests",
            "purchasing_history",
            "browsing_history",
            "device_fingerprint",
            "ip_address",
        }

    def validate_ad_targeting(self, ad: AdMetadata) -> dict[str, Any]:
        """
        Validate that an ad doesn't use forbidden targeting dimensions
        Returns a compliance report
        """
        violations = []

        # Check forbidden dimensions in targeting constraints
        if ad.targetingConstraints:
            for dimension, values in ad.targetingConstraints.items():
                if dimension.lower() in self.forbidden_targeting_dimensions:
                    violations.append(
                        {
                            "type": "forbidden_targeting_dimension",
                            "dimension": dimension,
                            "values": values,
                            "severity": "HIGH",
                        }
                    )

        # Check forbidden dimensions in forbiddenDimensions field
        for dimension in ad.forbiddenDimensions:
            if dimension.lower() in self.forbidden_targeting_dimensions:
                violations.append(
                    {"type": "forbidden_forbidden_dimension", "dimension": dimension, "severity": "MEDIUM"}
                )

        return {
            "is_compliant": len(violations) == 0,
            "violations": violations,
            "report_timestamp": datetime.utcnow().isoformat(),
        }

    def anonymize_user_data(self, intent: UniversalIntent) -> UniversalIntent:
        """
        Anonymize user data in the intent object to protect privacy
        Removes any potentially identifying information
        """
        # Create a copy of the intent to avoid modifying the original
        anonymized_intent = intent

        # Remove any session IDs or user identifiers from context
        if "sessionId" in anonymized_intent.context:
            del anonymized_intent.context["sessionId"]

        if "userId" in anonymized_intent.context:
            del anonymized_intent.context["userId"]

        # Clear session feedback which might contain user behavior
        anonymized_intent.sessionFeedback = None

        # Ensure expiresAt is set correctly (8 hours from creation)
        anonymized_intent.expiresAt = (datetime.utcnow() + timedelta(hours=8)).isoformat() + "Z"

        return anonymized_intent

    def check_intent_expiry(self, intent: UniversalIntent) -> bool:
        """
        Check if an intent object has expired based on its TTL
        """
        if not intent.expiresAt:
            return True  # Treat as expired if no expiry time

        try:
            # Parse the expiry time - it may already have timezone info
            expiry_str = intent.expiresAt
            if expiry_str.endswith("Z"):
                expiry_str = expiry_str[:-1] + "+00:00"
            expiry_time = datetime.fromisoformat(expiry_str)

            # Make sure both datetimes are timezone-aware or naive
            if expiry_time.tzinfo is not None:
                # If expiry_time is timezone-aware, make current time timezone-aware too

                current_time = datetime.now(UTC)
            else:
                # If expiry_time is naive, make current time naive too
                current_time = datetime.utcnow().replace(tzinfo=None)

            return current_time >= expiry_time
        except ValueError:
            # If we can't parse the expiry time, consider it expired
            return True


# Singleton instance
privacy_engine = PrivacyComplianceEngine()


def validate_advertiser_constraints(ad: AdMetadata) -> dict[str, Any]:
    """
    Public function to validate advertiser constraints for compliance
    """
    return privacy_engine.validate_ad_targeting(ad)


def anonymize_intent_data(intent: UniversalIntent) -> UniversalIntent:
    """
    Public function to anonymize intent data for privacy
    """
    return privacy_engine.anonymize_user_data(intent)


def is_intent_expired(intent: UniversalIntent) -> bool:
    """
    Public function to check if an intent has expired
    """
    return privacy_engine.check_intent_expiry(intent)
