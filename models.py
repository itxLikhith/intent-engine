"""
Intent Engine - Pydantic Models

This module defines Pydantic models for all entities and requests used in the API.
"""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from core.schema import UniversalIntent


# Database entity models
class AdvertiserBase(BaseModel):
    name: str
    contact_email: str | None = None


class AdvertiserCreate(AdvertiserBase):
    pass


class Advertiser(AdvertiserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class CampaignBase(BaseModel):
    name: str
    start_date: datetime | None = None
    end_date: datetime | None = None
    budget: float = 0.0
    daily_budget: float = 0.0
    status: str = "active"


class CampaignCreate(CampaignBase):
    advertiser_id: int


class CampaignUpdate(BaseModel):
    name: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    budget: float | None = None
    daily_budget: float | None = None
    status: str | None = None


class Campaign(CampaignBase):
    id: int
    advertiser_id: int
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class AdGroupBase(BaseModel):
    name: str
    targeting_settings: dict[str, Any] | None = None
    bid_strategy: str = "manual"


class AdGroupCreate(AdGroupBase):
    campaign_id: int


class AdGroupUpdate(BaseModel):
    name: str | None = None
    targeting_settings: dict[str, Any] | None = None
    bid_strategy: str | None = None


class AdGroup(AdGroupBase):
    id: int
    campaign_id: int
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class AdBase(BaseModel):
    advertiser_id: int
    ad_group_id: int | None = None
    title: str
    description: str | None = None
    url: str
    targeting_constraints: dict[str, Any] | None = None  # e.g., [{"dimension": "device_type", "value": "mobile"}]
    ethical_tags: list[str] | None = None  # e.g., ["privacy", "open_source"]
    quality_score: float = 0.5
    creative_format: str | None = None  # Banner, native, video, etc.
    bid_amount: float = 0.0  # Current bid amount
    status: str = "active"  # active, paused, disapproved
    approval_status: str = "pending"  # pending, approved, rejected


class AdCreate(AdBase):
    pass


class AdUpdate(BaseModel):
    advertiser_id: int | None = None
    ad_group_id: int | None = None
    title: str | None = None
    description: str | None = None
    url: str | None = None
    targeting_constraints: dict[str, Any] | None = None
    ethical_tags: list[str] | None = None
    quality_score: float | None = None
    creative_format: str | None = None
    bid_amount: float | None = None
    status: str | None = None
    approval_status: str | None = None


class Ad(AdBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class CreativeAssetBase(BaseModel):
    asset_type: str  # image, video, text, html
    asset_url: str
    dimensions: dict[str, Any] | None = None  # {"width": 300, "height": 250}
    checksum: str | None = None


class CreativeAssetCreate(CreativeAssetBase):
    ad_id: int


class CreativeAssetUpdate(BaseModel):
    asset_type: str | None = None
    asset_url: str | None = None
    dimensions: dict[str, Any] | None = None
    checksum: str | None = None


class CreativeAsset(CreativeAssetBase):
    id: int
    ad_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class AdMetricBase(BaseModel):
    ad_id: int
    date: date
    intent_goal: str | None = None  # e.g., "LEARN"
    intent_use_case: str | None = None  # e.g., "learning"
    impression_count: int = 0
    click_count: int = 0
    conversion_count: int = 0
    ctr: float | None = None  # NEW: Click-through rate
    cpm: float | None = None  # NEW: Cost per thousand impressions
    cpc: float | None = None  # NEW: Cost per click
    roas: float | None = None  # NEW: Return on ad spend
    engagement_rate: float | None = None  # NEW: Interaction rate
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
    session_id: str | None = None  # Anonymous session identifier
    ip_hash: str | None = None  # Hashed IP for fraud detection
    user_agent_hash: str | None = None  # Hashed user agent
    referring_url: str | None = None  # Source of click


class ClickTrackingCreate(ClickTrackingBase):
    pass


class ClickTracking(ClickTrackingBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


class ConversionTrackingBase(BaseModel):
    click_id: int  # Foreign key to click tracking
    conversion_type: str | None = None  # Purchase, signup, download, etc.
    value: float | None = None  # Conversion value (if applicable)
    status: str = "pending"  # Verified, pending, rejected


class ConversionTrackingCreate(ConversionTrackingBase):
    pass


class ConversionTracking(ConversionTrackingBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


class FraudDetectionBase(BaseModel):
    event_id: int | None = None  # ID of suspicious event
    event_type: str | None = None  # Click, impression, conversion
    reason: str | None = None  # Reason for flagging
    severity: str | None = None  # Low, medium, high risk
    review_status: str = "pending"  # Pending, reviewed, action_taken
    ad_id: int | None = None  # Optional link to ad


class FraudDetectionCreate(FraudDetectionBase):
    pass


class FraudDetection(FraudDetectionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Request models for API endpoints
class IntentExtractionInput(BaseModel):
    """Validated input for intent extraction"""

    text: str = Field(..., min_length=1, max_length=2000, description="User query text")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")


class IntentExtractionContext(BaseModel):
    """Validated context for intent extraction"""

    session_id: str | None = Field(None, max_length=128, description="Session identifier")
    user_locale: str | None = Field(None, pattern=r"^[a-z]{2}(-[A-Z]{2})?$", description="User locale (e.g., en-US)")
    product_context: str | None = Field(None, max_length=50, description="Product context (search, docs, mail, etc.)")
    additional_context: dict[str, Any] | None = Field(None, description="Additional context data")


class IntentExtractionOptions(BaseModel):
    """Validated options for intent extraction"""

    extract_constraints: bool = Field(True, description="Whether to extract constraints")
    extract_ethical_signals: bool = Field(True, description="Whether to extract ethical signals")
    extract_temporal: bool = Field(True, description="Whether to extract temporal intent")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")


class IntentExtractionRequest(BaseModel):
    """
    Validated request model for /extract-intent endpoint.

    Ensures all input data is properly validated before processing.
    """

    product: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^(search|docs|mail|calendar|meet|forms|sites|diary)$",
        description="Product type (search, docs, mail, calendar, meet, forms, sites, diary)",
    )
    input: IntentExtractionInput = Field(..., description="User input data")
    context: IntentExtractionContext = Field(default_factory=IntentExtractionContext, description="Extraction context")
    options: IntentExtractionOptions | None = Field(None, description="Extraction options")

    class Config:
        json_schema_extra = {
            "example": {
                "product": "search",
                "input": {
                    "text": "best laptop for programming under 50000 rupees",
                    "metadata": {"source": "web"},
                },
                "context": {
                    "session_id": "test-123",
                    "user_locale": "en-US",
                    "product_context": "search",
                },
                "options": {
                    "extract_constraints": True,
                    "extract_ethical_signals": True,
                    "extract_temporal": True,
                    "confidence_threshold": 0.5,
                },
            }
        }


class RankingRequest(BaseModel):
    intent: dict[str, Any]  # UniversalIntent as dict (converted from JSON)
    candidates: list[dict[str, Any]]  # SearchResult equivalent as dict
    options: dict[str, Any] | None = None


class ServiceRecommendationRequest(BaseModel):
    intent: dict[str, Any]  # UniversalIntent as dict (converted from JSON)
    available_services: list[dict[str, Any]]  # ServiceMetadata equivalent as dict
    options: dict[str, Any] | None = None


class AdMatchingRequest(BaseModel):
    intent: dict[str, Any]  # UniversalIntent as dict (converted from JSON)
    ad_inventory: list[dict[str, Any]]  # AdMetadata equivalent as dict
    config: dict[str, Any] | None = None


class AdMatchingWithCampaignRequest(AdMatchingRequest):
    campaign_context: dict[str, Any] | None = None  # Additional campaign-specific context


# Response models for API endpoints
class RankingResponse(BaseModel):
    ranked_results: list[dict[str, Any]]  # RankedResult equivalent as dict


class ServiceRecommendationResponse(BaseModel):
    recommendations: list[dict[str, Any]]  # ServiceRecommendation equivalent as dict


class AdMatchingResponse(BaseModel):
    matched_ads: list[dict[str, Any]]  # MatchedAd equivalent as dict
    metrics: dict[str, int]


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
    checks: dict[str, bool] | None = None
    version: str | None = None


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
    consent_details: dict[str, Any] | None = None
    granted_at: datetime
    expires_at: datetime | None = None
    withdrawn_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConsentRequest(BaseModel):
    user_id: str
    consent_type: str
    granted: bool
    consent_details: dict[str, Any] | None = None
    expires_in_days: int | None = None


class ConsentSummary(BaseModel):
    timestamp: str
    total_consents: int
    granted_consents: int
    denied_consents: int
    by_type: dict[str, int]
    overall_compliance_rate: float


# Models for Audit Trail
class AuditEvent(BaseModel):
    id: int
    user_id: str | None = None
    event_type: str
    resource_type: str | None = None
    resource_id: int | None = None
    action_description: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    metadata: dict[str, Any] | None = None
    timestamp: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class AuditEventRequest(BaseModel):
    user_id: str | None = None
    event_type: str
    resource_type: str | None = None
    resource_id: int | None = None
    action_description: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    metadata: dict[str, Any] | None = None


class AuditStats(BaseModel):
    timestamp: str
    total_events: int
    events_by_type: dict[str, int]
    daily_counts: list[dict[str, Any]]
    recent_activity: int


# Models for Privacy Controls
class DataRetentionPolicy(BaseModel):
    data_type: str
    retention_period: str
    deletion_date: str
    affected_records: int


class PrivacyComplianceReport(BaseModel):
    timestamp: str
    data_summary: dict[str, int]
    compliance_status: dict[str, Any]
    recommendations: list[str]


# A/B Testing Models
class ABTestBase(BaseModel):
    name: str
    description: str | None = None
    campaign_id: int
    traffic_allocation: float = 1.0
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    primary_metric: str = "ctr"


class ABTestCreate(ABTestBase):
    pass


class ABTestUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    traffic_allocation: float | None = None
    min_sample_size: int | None = None
    confidence_level: float | None = None
    primary_metric: str | None = None
    status: str | None = None


class ABTestResponse(ABTestBase):
    id: int
    status: str
    start_date: datetime | None = None
    end_date: datetime | None = None
    winner_variant_id: int | None = None
    created_at: datetime
    updated_at: datetime | None = None

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
    variants: list[dict[str, Any]]
    is_significant: bool
    winner_variant_id: int | None = None
    p_value: float | None = None
    recommended_action: str


# Fraud Detection Response Models
class FraudAnalysisResponse(BaseModel):
    is_fraudulent: bool
    risk_score: float
    signals: list[dict[str, Any]]
    recommended_action: str


class FraudScanSummary(BaseModel):
    total_analyzed: int
    fraudulent_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    fraud_types: dict[str, int]
    new_fraud_reports: int


# URL Ranking Models
class URLRankingAPIRequest(BaseModel):
    query: str
    urls: list[str]
    intent: UniversalIntent | None = None
    options: dict[str, Any] | None = None


class URLRankedResult(BaseModel):
    url: str
    title: str | None = None
    description: str | None = None
    domain: str | None = None
    privacy_score: float = 0.5
    tracker_count: int = 0
    encryption_enabled: bool = True
    content_type: str | None = None
    is_open_source: bool = False
    is_non_profit: bool = False
    relevance_score: float = 0.0
    final_score: float = 0.0


class URLRankingAPIResponse(BaseModel):
    query: str
    ranked_urls: list[URLRankedResult]
    processing_time_ms: float
    total_urls: int
    filtered_count: int = 0


# Advanced Analytics Models
class AttributionResultResponse(BaseModel):
    conversion_id: int
    touchpoints: list[dict[str, Any]]
    attribution_weights: dict[str, float]
    total_value: float
    attributed_values: dict[str, float]


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
    data_points: list[dict[str, Any]]
    forecast_next_period: float | None = None


# Unified Search Models (SearXNG Integration)
class UnifiedSearchRequest(BaseModel):
    """Request model for unified privacy search with intent ranking."""

    query: str
    categories: list[str] | None = None
    engines: list[str] | None = None
    language: str = "en"
    safe_search: int = 0
    time_range: str | None = None
    extract_intent: bool = True
    rank_results: bool = True
    weights: dict[str, float] | None = None
    min_privacy_score: float | None = None
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
    thumbnail: str | None = None
    published_date: str | None = None
    intent_goal: str | None = None
    match_reasons: list[str] = []
    privacy_score: float | None = None
    ethical_alignment: float | None = None


class ExtractedIntent(BaseModel):
    """Extracted intent from search query for API response."""

    goal: str
    constraints: list[dict[str, Any]]
    use_cases: list[str]
    result_type: str
    complexity: str
    confidence: float


class UnifiedSearchResponse(BaseModel):
    """Response model for unified search endpoint."""

    query: str
    results: list[RankedSearchResult]
    total_results: int
    processing_time_ms: float
    extracted_intent: ExtractedIntent | None = None
    engines_used: list[str] = []
    categories_searched: list[str] = []
    ranking_applied: bool = False
    results_ranked: int = 0
    privacy_enhanced: bool = True
    tracking_blocked: bool = True
