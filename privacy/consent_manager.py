"""
Consent Management Module

This module implements granular consent controls for user privacy preferences.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base

# Define Base here to avoid circular import
from database import Base




class ConsentType(Enum):
    """Types of consent that can be given"""
    ADVERTISING = "advertising"
    ANALYTICS = "analytics"
    PERSONALIZATION = "personalization"
    DATA_SHARING = "data_sharing"
    PROFILING = "profiling"


class ConsentStatus(Enum):
    """Status of consent"""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class UserConsent(Base):
    """User consent records table"""
    __tablename__ = "user_consents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)  # Could be hashed user ID or session ID
    consent_type = Column(String, nullable=False)  # Type of consent
    granted = Column(Boolean, nullable=False)  # True for granted, False for denied
    consent_details = Column(JSON)  # Additional details about the consent
    granted_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # When consent expires
    withdrawn_at = Column(DateTime)  # When consent was withdrawn
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ConsentManager:
    """Manages user consent for privacy preferences"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def record_consent(self, user_id: str, consent_type: ConsentType, granted: bool, 
                      consent_details: Optional[Dict] = None, expires_in_days: Optional[int] = None) -> UserConsent:
        """Record a user's consent decision"""
        from sqlalchemy.exc import IntegrityError
        
        # Calculate expiration date if specified
        expires_at = None
        if expires_in_days:
            from datetime import timedelta
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        consent_record = UserConsent(
            user_id=user_id,
            consent_type=consent_type.value,
            granted=granted,
            consent_details=consent_details or {},
            expires_at=expires_at
        )
        
        try:
            self.db.add(consent_record)
            self.db.commit()
            self.db.refresh(consent_record)
            return consent_record
        except IntegrityError:
            self.db.rollback()
            # Update existing record if it exists
            existing = self.db.query(UserConsent).filter(
                UserConsent.user_id == user_id,
                UserConsent.consent_type == consent_type.value
            ).first()
            
            if existing:
                existing.granted = granted
                existing.consent_details = consent_details or {}
                existing.expires_at = expires_at
                existing.updated_at = datetime.utcnow()
                if granted:
                    existing.withdrawn_at = None  # Reset withdrawn status if re-granted
                self.db.commit()
                self.db.refresh(existing)
                return existing
            else:
                raise
    
    def get_user_consent(self, user_id: str, consent_type: ConsentType) -> Optional[UserConsent]:
        """Get a user's consent for a specific type"""
        consent = self.db.query(UserConsent).filter(
            UserConsent.user_id == user_id,
            UserConsent.consent_type == consent_type.value
        ).first()
        
        # Check if consent has expired
        if consent and consent.expires_at and consent.expires_at < datetime.utcnow():
            consent.granted = False  # Mark as not granted if expired
            consent.consent_status = ConsentStatus.EXPIRED.value
            self.db.commit()
        
        return consent
    
    def get_all_user_consents(self, user_id: str) -> List[UserConsent]:
        """Get all consents for a user"""
        consents = self.db.query(UserConsent).filter(UserConsent.user_id == user_id).all()
        
        # Check for expired consents
        for consent in consents:
            if consent.expires_at and consent.expires_at < datetime.utcnow():
                consent.granted = False
                consent.consent_status = ConsentStatus.EXPIRED.value
        
        self.db.commit()
        return consents
    
    def withdraw_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Withdraw a user's consent"""
        consent = self.db.query(UserConsent).filter(
            UserConsent.user_id == user_id,
            UserConsent.consent_type == consent_type.value
        ).first()
        
        if consent:
            consent.granted = False
            consent.withdrawn_at = datetime.utcnow()
            consent.updated_at = datetime.utcnow()
            self.db.commit()
            return True
        
        return False
    
    def is_consent_granted(self, user_id: str, consent_type: ConsentType) -> bool:
        """Check if consent is granted for a specific type"""
        consent = self.get_user_consent(user_id, consent_type)
        
        if not consent:
            return False  # No consent recorded means not granted
        
        # Check if consent has expired
        if consent.expires_at and consent.expires_at < datetime.utcnow():
            consent.granted = False
            consent.consent_status = ConsentStatus.EXPIRED.value
            self.db.commit()
            return False
        
        # Check if consent was withdrawn
        if consent.withdrawn_at:
            return False
        
        return consent.granted
    
    def get_consent_summary(self) -> Dict[str, Any]:
        """Get a summary of all consents in the system"""
        total_consents = self.db.query(UserConsent).count()
        granted_consents = self.db.query(UserConsent).filter(UserConsent.granted == True).count()
        denied_consents = self.db.query(UserConsent).filter(UserConsent.granted == False).count()
        
        # Count by type
        consent_counts_by_type = {}
        for consent_type in ConsentType:
            count = self.db.query(UserConsent).filter(
                UserConsent.consent_type == consent_type.value
            ).count()
            consent_counts_by_type[consent_type.value] = count
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_consents": total_consents,
            "granted_consents": granted_consents,
            "denied_consents": denied_consents,
            "by_type": consent_counts_by_type,
            "overall_compliance_rate": granted_consents / max(total_consents, 1) * 100
        }
        
        return summary


def get_consent_manager(db_session: Session) -> ConsentManager:
    """Factory function to get consent manager instance"""
    return ConsentManager(db_session)