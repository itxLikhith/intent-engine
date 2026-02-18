"""
Audit Trail Module

This module implements comprehensive logging of all operations for compliance and security.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import Session

# Define Base here to avoid circular import
from database import Base


class AuditEventType(Enum):
    """Types of auditable events"""

    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    AD_CREATION = "ad_creation"
    AD_UPDATE = "ad_update"
    AD_DELETION = "ad_deletion"
    CAMPAIGN_CREATION = "campaign_creation"
    CAMPAIGN_UPDATE = "campaign_update"
    CAMPAIGN_DELETION = "campaign_deletion"
    AD_VIEW = "ad_view"
    AD_CLICK = "ad_click"
    CONVERSION_TRACKING = "conversion_tracking"
    PRIVACY_SETTING_CHANGE = "privacy_setting_change"
    CONSENT_GIVEN = "consent_given"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    ADMIN_ACTION = "admin_action"
    SYSTEM_ERROR = "system_error"
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"


class AuditTrail(Base):
    """Audit trail records table"""

    __tablename__ = "audit_trails"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)  # User performing the action
    event_type = Column(String, nullable=False)  # Type of event
    resource_type = Column(String)  # Type of resource affected (ad, campaign, etc.)
    resource_id = Column(Integer)  # ID of the resource affected
    action_description = Column(Text)  # Description of the action taken
    ip_address = Column(String)  # IP address of the user
    user_agent = Column(String)  # User agent string
    payload = Column(JSON)  # Additional metadata about the event
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditTrailManager:
    """Manages comprehensive audit logging for compliance and security"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def log_event(
        self,
        user_id: str | None,
        event_type: AuditEventType,
        resource_type: str | None = None,
        resource_id: int | None = None,
        action_description: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        metadata: dict | None = None,
    ) -> AuditTrail:
        """Log an event to the audit trail"""
        audit_entry = AuditTrail(
            user_id=user_id,
            event_type=event_type.value,
            resource_type=resource_type,
            resource_id=resource_id,
            action_description=action_description,
            ip_address=ip_address,
            user_agent=user_agent,
            payload=metadata or {},
        )

        self.db.add(audit_entry)
        self.db.commit()
        self.db.refresh(audit_entry)

        return audit_entry

    def get_audit_events(
        self,
        user_id: str | None = None,
        event_type: AuditEventType | None = None,
        resource_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditTrail]:
        """Retrieve audit events with optional filters"""
        query = self.db.query(AuditTrail)

        if user_id:
            query = query.filter(AuditTrail.user_id == user_id)

        if event_type:
            query = query.filter(AuditTrail.event_type == event_type.value)

        if resource_type:
            query = query.filter(AuditTrail.resource_type == resource_type)

        if start_date:
            query = query.filter(AuditTrail.timestamp >= start_date)

        if end_date:
            query = query.filter(AuditTrail.timestamp <= end_date)

        events = query.order_by(AuditTrail.timestamp.desc()).offset(offset).limit(limit).all()
        return events

    def get_user_audit_events(self, user_id: str, limit: int = 50) -> list[AuditTrail]:
        """Get audit events for a specific user"""
        events = (
            self.db.query(AuditTrail)
            .filter(AuditTrail.user_id == user_id)
            .order_by(AuditTrail.timestamp.desc())
            .limit(limit)
            .all()
        )

        return events

    def get_recent_events(self, hours: int = 24) -> list[AuditTrail]:
        """Get recent audit events within the specified hours"""
        from datetime import timedelta

        start_time = datetime.utcnow() - timedelta(hours=hours)

        events = (
            self.db.query(AuditTrail)
            .filter(AuditTrail.timestamp >= start_time)
            .order_by(AuditTrail.timestamp.desc())
            .all()
        )

        return events

    def get_event_statistics(self) -> dict[str, Any]:
        """Get statistics about audit events"""
        from sqlalchemy import func

        total_events = self.db.query(AuditTrail).count()

        # Count by event type
        event_counts = (
            self.db.query(AuditTrail.event_type, func.count(AuditTrail.id).label("count"))
            .group_by(AuditTrail.event_type)
            .all()
        )

        # Count by day for the last week
        from datetime import timedelta

        week_ago = datetime.utcnow() - timedelta(days=7)
        daily_counts = (
            self.db.query(
                func.date(AuditTrail.timestamp).label("date"),
                func.count(AuditTrail.id).label("count"),
            )
            .filter(AuditTrail.timestamp >= week_ago)
            .group_by(func.date(AuditTrail.timestamp))
            .order_by("date")
            .all()
        )

        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_events": total_events,
            "events_by_type": {event_type: count for event_type, count in event_counts},
            "daily_counts": [{"date": str(date), "count": count} for date, count in daily_counts],
            "recent_activity": len(self.get_recent_events(hours=24)),
        }

        return stats

    def cleanup_old_events(self, days_to_keep: int = 90) -> int:
        """Remove audit events older than the specified number of days"""
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        deleted_count = self.db.query(AuditTrail).filter(AuditTrail.timestamp < cutoff_date).delete()

        self.db.commit()
        return deleted_count


def get_audit_trail_manager(db_session: Session) -> AuditTrailManager:
    """Factory function to get audit trail manager instance"""
    return AuditTrailManager(db_session)
