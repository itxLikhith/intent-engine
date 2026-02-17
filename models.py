"""
Intent Engine - Pydantic Models

This module defines Pydantic models for all entities and requests used in the API.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from core.schema import IntentExtractionRequest, UniversalIntent


# Database entity models
class AdvertiserBase(BaseModel):
    name: str
    contact_email: Optional[str] = None


class AdvertiserCreate(AdvertiserBase):
    pass


class Advertiser(AdvertiserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class CampaignBase(BaseModel):
    name: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    budget: float = 0.0
    daily_budget: float = 0.0
    status: str = "active"


class CampaignCreate(CampaignBase):
    advertiser_id: int


class CampaignUpdate(BaseModel):
    name: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    budget: Optional[float] = None
    daily_budget: Optional[float] = None
    status: Optional[str] = None


class Campaign(CampaignBase):
    id: int
    advertiser_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AdGroupBase(BaseModel):
    name: str
    targeting_settings: Optional[Dict[str, Any]] = None
    bid_strategy: str = "manual"


class AdGroupCreate(AdGroupBase):
    campaign_id: int


class AdGroupUpdate(BaseModel):
    name: Optional[str] = None
    targeting_settings: Optional[Dict[str, Any]] = None
    bid_strategy: Optional[str] = None


class AdGroup(AdGroupBase):
    id: int
    campaign_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AdBase(BaseModel):
    advertiser_id: int
    ad_group_id: Optional[int] = None
    title: str
    description: Optional[str] = None
    url: str
    targeting_constraints: Optional[Dict[str, Any]] = None  # e.g., [{"dimension": "device_type", "value": "mobile"}]
    ethical_tags: Optional[List[str]] = None  # e.g., ["privacy", "open_source"]
    quality_score: float = 0.5
    creative_format: Optional[str] = None  # Banner, native, video, etc.
    bid_amount: float = 0.0  # Current bid amount
    status: str = "active"  # active, paused, disapproved
    approval_status: str = "pending"  # pending, approved, rejected


class AdCreate(AdBase):
    pass


class AdUpdate(BaseModel):
    advertiser_id: Optional[int] = None
    ad_group_id: Optional[int] = None
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    targeting_constraints: Optional[Dict[str, Any]] = None
    ethical_tags: Optional[List[str]] = None
    quality_score: Optional[float] = None
    creative_format: Optional[str] = None
    bid_amount: Optional[float] = None
    status: Optional[str] = None
    approval_status: Optional[str] = None


class Ad(AdBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class CreativeAssetBase(BaseModel):
    asset_type: str  # image, video, text, html
    asset_url: str
    dimensions: Optional[Dict[str, Any]] = None  # {"width": 300, "height": 250}
    checksum: Optional[str] = None


class CreativeAssetCreate(CreativeAssetBase):
    ad_id: int


class CreativeAssetUpdate(BaseModel):
    asset_type: Optional[str] = None
    asset_url: Optional[str] = None
    dimensions: Optional[Dict[str, Any]] = None
    checksum: Optional[str] = None


class CreativeAsset(CreativeAssetBase):
    id: int
    ad_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class AdMetricBase(BaseModel):
    ad_id: int
    date: date
    intent_goal: Optional[str] = None  # e.g., "LEARN"
    intent_use_case: Optional[str] = None  # e.g., "learning"
    impression_count: int = 0
    click_count: int = 0
    conversion_count: int = 0
    ctr: Optional[float] = None  # NEW: Click-through rate
    cpm: Optional[float] = None  # NEW: Cost per thousand impressions
    cpc: Optional[float] = None  # NEW: Cost per click
    roas: Optional[float] = None  # NEW: Return on ad spend
    engagement_rate: Optional[float] = None  # NEW: Interaction rate
    expires_at: datetime  # 30 days from creation


class AdMetricCreate(AdMetricBase):
    pass


class AdMetric(AdMetricBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ClickTrackingBase(BaseModel):
    ad_id: int
    session_id: Optional[str] = None  # Anonymous session identifier
    ip_hash: Optional[str] = None  # Hashed IP for fraud detection
    user_agent_hash: Optional[str] = None  # Hashed user agent
    referring_url: Optional[str] = None  # Source of click


class ClickTrackingCreate(ClickTrackingBase):
    pass


class ClickTracking(ClickTrackingBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


class ConversionTrackingBase(BaseModel):
    click_id: int  # Foreign key to click tracking
    conversion_type: Optional[str] = None  # Purchase, signup, download, etc.
    value: Optional[float] = None  # Conversion value (if applicable)
    status: str = "pending"  # Verified, pending, rejected


class ConversionTrackingCreate(ConversionTrackingBase):
    pass


class ConversionTracking(ConversionTrackingBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


class FraudDetectionBase(BaseModel):
    event_id: Optional[int] = None  # ID of suspicious event
    event_type: Optional[str] = None  # Click, impression, conversion
    reason: Optional[str] = None  # Reason for flagging
    severity: Optional[str] = None  # Low, medium, high risk
    review_status: str = "pending"  # Pending, reviewed, action_taken
    ad_id: Optional[int] = None  # Optional link to ad


class FraudDetectionCreate(FraudDetectionBase):
    pass


class FraudDetection(FraudDetectionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Request models for API endpoints
class RankingRequest(BaseModel):
    intent: Dict[str, Any]  # UniversalIntent as dict (converted from JSON)
    candidates: List[Dict[str, Any]]  # SearchResult equivalent as dict
    options: Optional[Dict[str, Any]] = None


class ServiceRecommendationRequest(BaseModel):
    intent: Dict[str, Any]  # UniversalIntent as dict (converted from JSON)
    available_services: List[Dict[str, Any]]  # ServiceMetadata equivalent as dict
    options: Optional[Dict[str, Any]] = None


class AdMatchingRequest(BaseModel):
    intent: Dict[str, Any]  # UniversalIntent as dict (converted from JSON)
    ad_inventory: List[Dict[str, Any]]  # AdMetadata equivalent as dict
    config: Optional[Dict[str, Any]] = None


class AdMatchingWithCampaignRequest(AdMatchingRequest):
    campaign_context: Optional[Dict[str, Any]] = None  # Additional campaign-specific context


# Response models for API endpoints
class RankingResponse(BaseModel):
    ranked_results: List[Dict[str, Any]]  # RankedResult equivalent as dict


class ServiceRecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]  # ServiceRecommendation equivalent as dict


class AdMatchingResponse(BaseModel):
    matched_ads: List[Dict[str, Any]]  # MatchedAd equivalent as dict
    metrics: Dict[str, int]


# Reporting models
class CampaignPerformanceReport(BaseModel):
    campaign_id: int
    campaign_name: str
    impressions: int
    clicks: int
    conversions: int
    ctr: float
    cpc: float
    cost: float
    roas: float


# Additional models for the API
class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime


class StatusResponse(BaseModel):
    service: str
    version: str
    uptime: str
    status: str


# Models for Consent Management
class ConsentRecord(BaseModel):
    id: int
    user_id: str
    consent_type: str
    granted: bool
    consent_details: Optional[Dict[str, Any]] = None
    granted_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConsentRequest(BaseModel):
    user_id: str
    consent_type: str
    granted: bool
    consent_details: Optional[Dict[str, Any]] = None
    expires_in_days: Optional[int] = None


class ConsentSummary(BaseModel):
    timestamp: str
    total_consents: int
    granted_consents: int
    denied_consents: int
    by_type: Dict[str, int]
    overall_compliance_rate: float


# Models for Audit Trail
class AuditEvent(BaseModel):
    id: int
    user_id: Optional[str] = None
    event_type: str
    resource_type: Optional[str] = None
    resource_id: Optional[int] = None
    action_description: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class AuditEventRequest(BaseModel):
    user_id: Optional[str] = None
    event_type: str
    resource_type: Optional[str] = None
    resource_id: Optional[int] = None
    action_description: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AuditStats(BaseModel):
    timestamp: str
    total_events: int
    events_by_type: Dict[str, int]
    daily_counts: List[Dict[str, Any]]
    recent_activity: int


# Models for Privacy Controls
class DataRetentionPolicy(BaseModel):
    data_type: str
    retention_period: str
    deletion_date: str
    affected_records: int


class PrivacyComplianceReport(BaseModel):
    timestamp: str
    data_summary: Dict[str, int]
    compliance_status: Dict[str, Any]
    recommendations: List[str]


# A/B Testing Models
class ABTestBase(BaseModel):
    name: str
    description: Optional[str] = None
    campaign_id: int
    traffic_allocation: float = 1.0
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    primary_metric: str = "ctr"


class ABTestCreate(ABTestBase):
    pass


class ABTestUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    traffic_allocation: Optional[float] = None
    min_sample_size: Optional[int] = None
    confidence_level: Optional[float] = None
    primary_metric: Optional[str] = None
    status: Optional[str] = None


class ABTestResponse(ABTestBase):
    id: int
    status: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    winner_variant_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ABTestVariantBase(BaseModel):
    name: str
    ad_id: int
    traffic_weight: float = 0.5
    is_control: bool = False


class ABTestVariantCreate(ABTestVariantBase):
    test_id: int


class ABTestVariantResponse(ABTestVariantBase):
    id: int
    test_id: int
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    created_at: datetime

    class Config:
        from_attributes = True


class ABTestResultsResponse(BaseModel):
    test_id: int
    status: str
    total_impressions: int
    variants: List[Dict[str, Any]]
    is_significant: bool
    winner_variant_id: Optional[int] = None
    p_value: Optional[float] = None
    recommended_action: str


# Fraud Detection Response Models
class FraudAnalysisResponse(BaseModel):
    is_fraudulent: bool
    risk_score: float
    signals: List[Dict[str, Any]]
    recommended_action: str


class FraudScanSummary(BaseModel):
    total_analyzed: int
    fraudulent_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    fraud_types: Dict[str, int]
    new_fraud_reports: int


# URL Ranking Models
class URLRankingAPIRequest(BaseModel):
    query: str
    urls: List[str]
    intent: Optional[UniversalIntent] = None
    options: Optional[Dict[str, Any]] = None


class URLRankedResult(BaseModel):
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    privacy_score: float = 0.5
    tracker_count: int = 0
    encryption_enabled: bool = True
    content_type: Optional[str] = None
    is_open_source: bool = False
    is_non_profit: bool = False
    relevance_score: float = 0.0
    final_score: float = 0.0


class URLRankingAPIResponse(BaseModel):
    query: str
    ranked_urls: List[URLRankedResult]
    processing_time_ms: float
    total_urls: int
    filtered_count: int = 0


# Advanced Analytics Models
class AttributionResultResponse(BaseModel):
    conversion_id: int
    touchpoints: List[Dict[str, Any]]
    attribution_weights: Dict[str, float]
    total_value: float
    attributed_values: Dict[str, float]


class CampaignROIResponse(BaseModel):
    campaign_id: int
    campaign_name: str
    total_spend: float
    total_revenue: float
    roi: float
    roas: float
    cpa: float
    clv: float
    impressions: int
    clicks: int
    conversions: int
    ctr: float
    cvr: float


class TrendAnalysisResponse(BaseModel):
    metric_name: str
    current_value: float
    previous_value: float
    change_percent: float
    trend_direction: str
    data_points: List[Dict[str, Any]]
    forecast_next_period: Optional[float] = None


# Unified Search Models (SearXNG Integration)
class UnifiedSearchRequest(BaseModel):
    """Request model for unified privacy search with intent ranking."""

    query: str
    categories: Optional[List[str]] = None
    engines: Optional[List[str]] = None
    language: str = "en"
    safe_search: int = 0
    time_range: Optional[str] = None
    extract_intent: bool = True
    rank_results: bool = True
    weights: Optional[Dict[str, float]] = None
    min_privacy_score: Optional[float] = None
    exclude_big_tech: bool = False


class RankedSearchResult(BaseModel):
    """Ranked search result with intent alignment and privacy scoring."""

    url: str
    title: str
    content: str
    engine: str
    original_score: float
    ranked_score: float
    rank: int
    category: str
    thumbnail: Optional[str] = None
    published_date: Optional[str] = None
    intent_goal: Optional[str] = None
    match_reasons: List[str] = []
    privacy_score: Optional[float] = None
    ethical_alignment: Optional[float] = None


class ExtractedIntent(BaseModel):
    """Extracted intent from search query for API response."""

    goal: str
    constraints: List[Dict[str, Any]]
    use_cases: List[str]
    result_type: str
    complexity: str
    confidence: float


class UnifiedSearchResponse(BaseModel):
    """Response model for unified search endpoint."""

    query: str
    results: List[RankedSearchResult]
    total_results: int
    processing_time_ms: float
    extracted_intent: Optional[ExtractedIntent] = None
    engines_used: List[str] = []
    categories_searched: List[str] = []
    ranking_applied: bool = False
    results_ranked: int = 0
    privacy_enhanced: bool = True
    tracking_blocked: bool = True
