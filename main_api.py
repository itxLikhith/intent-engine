"""
Intent Engine - Main API Service

This module implements the FastAPI service with all required endpoints for the Intent Engine.
"""

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy import func
from sqlalchemy.orm import Session

from ads.matcher import match_ads
from analytics.realtime import handle_analytics_websocket
from audit.audit_trail import AuditEventType, get_audit_trail_manager
from core.schema import UniversalIntent
from database import Ad as DbAd
from database import AdGroup as DbAdGroup
from database import AdMetric as DbAdMetric
from database import Advertiser as DbAdvertiser
from database import Base
from database import Campaign as DbCampaign
from database import ClickTracking as DbClickTracking
from database import ConversionTracking as DbConversionTracking
from database import CreativeAsset as DbCreativeAsset
from database import FraudDetection as DbFraudDetection
from database import db_manager, engine
from extraction.extractor import extract_intent
from models import (
    ABTestCreate,
    ABTestResponse,
    ABTestResultsResponse,
    ABTestVariantCreate,
    ABTestVariantResponse,
    Ad,
    AdCreate,
    AdGroup,
    AdGroupCreate,
    AdGroupUpdate,
    AdMatchingRequest,
    AdMatchingResponse,
    AdMatchingWithCampaignRequest,
    AdUpdate,
    Advertiser,
    AdvertiserCreate,
    AttributionResultResponse,
    AuditEvent,
    AuditStats,
    Campaign,
    CampaignCreate,
    CampaignPerformanceReport,
    CampaignROIResponse,
    CampaignUpdate,
    ClickTracking,
    ClickTrackingCreate,
    ConsentRecord,
    ConsentSummary,
    ConversionTracking,
    ConversionTrackingCreate,
    CreativeAsset,
    CreativeAssetCreate,
    CreativeAssetUpdate,
    DataRetentionPolicy,
    FraudAnalysisResponse,
    FraudDetection,
    FraudDetectionCreate,
    FraudScanSummary,
    HealthCheckResponse,
    PrivacyComplianceReport,
    RankingRequest,
    RankingResponse,
    ServiceRecommendationRequest,
    ServiceRecommendationResponse,
    StatusResponse,
    TrendAnalysisResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
    URLRankedResult,
    URLRankingAPIRequest,
    URLRankingAPIResponse,
)
from privacy.consent_manager import ConsentType, get_consent_manager
from privacy.enhanced_privacy import DataRetentionPeriod, get_enhanced_privacy_controls
from privacy_core import anonymize_intent_data, is_intent_expired, validate_advertiser_constraints
from ranking.optimized_ranker import rank_results
from searxng.unified_search import get_unified_search_service
from services.recommender import recommend_services


# Create the database dependency
def get_db():
    db = next(db_manager.get_db())
    try:
        yield db
    finally:
        db.close()


# Helper function to convert dict to UniversalIntent
def convert_dict_to_universal_intent(intent_dict: Dict[str, Any]) -> UniversalIntent:
    """
    Convert intent dictionary to UniversalIntent dataclass.
    Reusable helper for /rank-results and /match-ads endpoints.
    """
    from core.schema import (
        Complexity,
        Constraint,
        ConstraintType,
        DeclaredIntent,
        EthicalDimension,
        EthicalSignal,
        Frequency,
        InferredIntent,
        IntentGoal,
        Recency,
        ResultType,
        SessionFeedback,
        SkillLevel,
        TemporalHorizon,
        TemporalIntent,
        UniversalIntent,
        Urgency,
        UseCase,
    )

    # Convert constraints
    constraints = []
    for c in intent_dict.get("declared", {}).get("constraints", []):
        if isinstance(c, dict):
            constraint_type = c.get("type")
            if isinstance(constraint_type, str) and constraint_type in [ct.value for ct in ConstraintType]:
                constraint_type = ConstraintType(constraint_type)
            constraints.append(
                Constraint(
                    type=constraint_type,
                    dimension=c.get("dimension", "") or "",
                    value=c.get("value", "") or "",
                    hardFilter=c.get("hardFilter", True),
                )
            )

    # Convert declared intent
    declared_dict = intent_dict.get("declared", {}) or {}
    goal = declared_dict.get("goal")
    if isinstance(goal, str) and goal in [g.value for g in IntentGoal]:
        goal = IntentGoal(goal)
    urgency = declared_dict.get("urgency", "FLEXIBLE")
    if isinstance(urgency, str) and urgency in [u.value for u in Urgency]:
        urgency = Urgency(urgency)
    skill_level = declared_dict.get("skillLevel", "INTERMEDIATE")
    if isinstance(skill_level, str) and skill_level in [s.value for s in SkillLevel]:
        skill_level = SkillLevel(skill_level)

    declared = DeclaredIntent(
        query=declared_dict.get("query"),
        goal=goal,
        constraints=constraints,
        negativePreferences=declared_dict.get("negativePreferences", []) or [],
        urgency=urgency if isinstance(urgency, Urgency) else Urgency.FLEXIBLE,
        budget=declared_dict.get("budget"),
        skillLevel=skill_level if isinstance(skill_level, SkillLevel) else SkillLevel.INTERMEDIATE,
    )

    # Convert inferred intent
    inferred_dict = intent_dict.get("inferred", {}) or {}
    use_cases = []
    for uc in inferred_dict.get("useCases", []) or []:
        if isinstance(uc, str) and uc in [u.value for u in UseCase]:
            use_cases.append(UseCase(uc))

    result_type = inferred_dict.get("resultType")
    if isinstance(result_type, str) and result_type in [r.value for r in ResultType]:
        result_type = ResultType(result_type)

    complexity = inferred_dict.get("complexity", "MODERATE")
    if isinstance(complexity, str) and complexity in [c.value for c in Complexity]:
        complexity = Complexity(complexity)

    ethical_signals = []
    for es in inferred_dict.get("ethicalSignals", []) or []:
        if isinstance(es, dict):
            dimension = es.get("dimension")
            if isinstance(dimension, str) and dimension in [d.value for d in EthicalDimension]:
                dimension = EthicalDimension(dimension)
            ethical_signals.append(
                EthicalSignal(
                    dimension=dimension if isinstance(dimension, EthicalDimension) else None,
                    preference=es.get("preference", "") or "",
                )
            )

    # Handle temporal intent
    temporal_dict = inferred_dict.get("temporalIntent")
    temporal_intent = None
    if temporal_dict and isinstance(temporal_dict, dict):
        horizon = temporal_dict.get("horizon", "FLEXIBLE")
        if isinstance(horizon, str) and horizon in [h.value for h in TemporalHorizon]:
            horizon = TemporalHorizon(horizon)
        recency = temporal_dict.get("recency", "EVERGREEN")
        if isinstance(recency, str) and recency in [r.value for r in Recency]:
            recency = Recency(recency)
        frequency = temporal_dict.get("frequency", "FLEXIBLE")
        if isinstance(frequency, str) and frequency in [f.value for f in Frequency]:
            frequency = Frequency(frequency)
        temporal_intent = TemporalIntent(
            horizon=horizon if isinstance(horizon, TemporalHorizon) else TemporalHorizon.FLEXIBLE,
            recency=recency if isinstance(recency, Recency) else Recency.EVERGREEN,
            frequency=frequency if isinstance(frequency, Frequency) else Frequency.FLEXIBLE,
        )

    inferred = InferredIntent(
        useCases=use_cases,
        temporalIntent=temporal_intent,
        documentContext=inferred_dict.get("documentContext"),
        meetingContext=inferred_dict.get("meetingContext"),
        resultType=result_type if isinstance(result_type, ResultType) else None,
        complexity=complexity if isinstance(complexity, Complexity) else Complexity.MODERATE,
        ethicalSignals=ethical_signals,
    )

    # Convert session feedback
    feedback_dict = intent_dict.get("sessionFeedback", {}) or {}
    session_feedback = SessionFeedback(
        clicked=feedback_dict.get("clicked"),
        dwell=feedback_dict.get("dwell"),
        reformulated=feedback_dict.get("reformulated"),
        bounced=feedback_dict.get("bounced"),
    )

    # Create UniversalIntent
    intent = UniversalIntent(
        intentId=intent_dict.get("intentId", "") or "",
        context=intent_dict.get("context", {}) or {},
        declared=declared,
        inferred=inferred,
        sessionFeedback=session_feedback,
        expiresAt=intent_dict.get("expiresAt", "") or "",
    )

    return intent


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
extraction_latency = Histogram("intent_extraction_latency_seconds", "Time spent in intent extraction")
ranking_throughput = Counter("ranking_throughput_total", "Number of ranking requests processed")
ad_matching_success_rate = Counter("ad_matching_success_total", "Number of successful ad matches")
fairness_violations = Counter("fairness_violations_total", "Number of fairness violations detected")
active_sessions = Gauge("active_sessions", "Number of active user sessions")
extraction_requests = Counter("intent_extraction_requests_total", "Number of intent extraction requests")
ranking_requests_counter = Counter("ranking_requests_processed_total", "Number of ranking requests processed")
service_recommendation_requests = Counter(
    "service_recommendation_requests_total", "Number of service recommendation requests"
)
url_ranking_requests = Counter("url_ranking_requests_total", "Number of URL ranking requests")
unified_search_requests = Counter("unified_search_requests_total", "Number of unified search requests")


# Rate limiting configuration
def get_rate_limit_enabled() -> bool:
    return os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"


def get_rate_limit_default() -> str:
    return os.getenv("RATE_LIMIT_DEFAULT", "100/minute")


def get_rate_limit_strict() -> str:
    return os.getenv("RATE_LIMIT_STRICT", "10/minute")


# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address, default_limits=[get_rate_limit_default()], enabled=get_rate_limit_enabled()
)

# Prometheus metrics
app = FastAPI(
    title="Intent Engine API",
    description="Privacy-first, intent-driven search, service recommendation, and ad matching system",
    version="1.0.0",
)


# Configure CORS from environment variables
def get_cors_origins() -> List[str]:
    """
    Get CORS origins from environment variable.
    Returns a list of allowed origins.
    """
    origins_str = os.getenv("CORS_ORIGINS", "")
    if origins_str:
        # Parse comma-separated origins
        origins = [origin.strip() for origin in origins_str.split(",")]
        return [origin for origin in origins if origin]  # Filter empty strings
    # Default to localhost for development if no origins specified
    return ["http://localhost:3000", "http://localhost:8080"]


def get_cors_allow_methods() -> List[str]:
    """
    Get allowed CORS methods from environment variable.
    """
    methods_str = os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS")
    return [method.strip() for method in methods_str.split(",")]


def get_cors_allow_headers() -> List[str]:
    """
    Get allowed CORS headers from environment variable.
    """
    headers_str = os.getenv("CORS_ALLOW_HEADERS", "Authorization,Content-Type,X-Requested-With")
    return [header.strip() for header in headers_str.split(",")]


# Add CORS middleware with configurable origins
cors_enabled = os.getenv("ENABLE_CORS", "true").lower() == "true"
if cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
        allow_methods=get_cors_allow_methods(),
        allow_headers=get_cors_allow_headers(),
    )
    logger.info(f"CORS enabled for origins: {get_cors_origins()}")
else:
    logger.info("CORS middleware disabled")

# Add rate limiting middleware
if get_rate_limit_enabled():
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    logger.info(f"Rate limiting enabled: {get_rate_limit_default()}")
else:
    logger.info("Rate limiting disabled")


@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware to track request metrics"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Track response times
    if request.url.path.startswith("/extract-intent"):
        extraction_latency.observe(process_time)
    elif request.url.path.startswith("/rank-results"):
        ranking_requests_counter.inc()
    elif request.url.path.startswith("/match-ads"):
        ad_matching_success_rate.inc()

    return response


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("Starting up Intent Engine API...")

    # Import all modules that define database tables to ensure they're registered with Base.metadata
    # We need to explicitly access the model classes to ensure they're registered with Base.metadata

    # Log the tables that will be created
    logger.info(f"Registered tables: {list(Base.metadata.tables.keys())}")

    # Initialize database tables - use try/except to handle race conditions with multiple workers
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("Database tables created successfully")
    except Exception as e:
        # This can happen if multiple workers try to create tables simultaneously
        # or if tables already exist - this is safe to ignore
        logger.warning(f"Table creation warning (safe to ignore if tables exist): {str(e)}")

    # Pre-load models to avoid cold start
    from ads.matcher import get_ad_matcher
    from extraction.extractor import get_intent_extractor
    from ranking.ranker import get_intent_ranker
    from services.recommender import get_service_recommender

    get_intent_extractor()
    get_intent_ranker()
    get_service_recommender()
    get_ad_matcher()

    from ranking.url_ranker import get_url_ranker

    get_url_ranker()

    logger.info("Models loaded and API ready!")


@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(status="healthy", timestamp=datetime.now(timezone.utc))


@app.post("/extract-intent", response_model=Dict[str, Any])
async def extract_intent_endpoint(request: Dict[str, Any]):
    """
    Extract structured intent from user query
    """
    extraction_requests.inc()  # Increment counter

    time.time()

    try:
        # Convert dict to IntentExtractionRequest
        from core.schema import IntentExtractionRequest

        intent_request = IntentExtractionRequest(
            product=request.get("product", ""),
            input=request.get("input", {}),
            context=request.get("context", {}),
            options=request.get("options", {}),
        )

        # Call the extraction function
        response = extract_intent(intent_request)

        # Record extraction latency (already handled by middleware)
        # extraction_latency.observe(time.time() - start_time)

        return response.__dict__

    except Exception as e:
        logger.error(f"Error in intent extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=UnifiedSearchResponse)
async def unified_search_endpoint(request: UnifiedSearchRequest):
    """
    Unified privacy search with intent extraction and ranking.

    Combines SearXNG privacy-focused search with Intent Engine's
    intent extraction and ranking capabilities to provide:
    - Privacy-first search (no tracking via SearXNG)
    - Intent extraction from queries
    - Intent-aware result ranking
    - Privacy score calculation
    - Ethical alignment scoring
    """
    unified_search_requests.inc()  # Increment counter

    try:
        # Get unified search service
        search_service = get_unified_search_service()

        # Perform unified search
        response = await search_service.search(request)

        return response

    except Exception as e:
        import traceback

        logger.error(f"Error in unified search: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rank-results", response_model=RankingResponse)
async def rank_results_endpoint(request: RankingRequest):
    """
    Rank results based on user intent
    """
    ranking_requests_counter.inc()  # Increment counter

    try:
        # Call the ranking function
        from ranking.ranker import RankingRequest as InternalRankingRequest
        from ranking.ranker import SearchResult

        # Convert intent dict to UniversalIntent dataclass
        intent = convert_dict_to_universal_intent(request.intent)

        # Convert the request to internal format
        search_results = []
        for candidate in request.candidates:
            search_result = SearchResult(
                id=candidate.get("id", ""),
                title=candidate.get("title", ""),
                description=candidate.get("description", ""),
                platform=candidate.get("platform"),
                provider=candidate.get("provider"),
                license=candidate.get("license"),
                price=candidate.get("price"),
                tags=candidate.get("tags", []),
                qualityScore=candidate.get("qualityScore", 0.5),
                recency=candidate.get("recency"),
                complexity=candidate.get("complexity"),
                compatibility=candidate.get("compatibility", []),
                privacyRating=candidate.get("privacyRating"),
                opensource=candidate.get("opensource"),
            )
            search_results.append(search_result)

        internal_request = InternalRankingRequest(intent=intent, candidates=search_results, options=request.options)

        response = rank_results(internal_request)

        # Convert response to dict format
        ranked_results = []
        for ranked_result in response.rankedResults:
            ranked_results.append(
                {
                    "result": {
                        "id": ranked_result.result.id,
                        "title": ranked_result.result.title,
                        "description": ranked_result.result.description,
                        "platform": ranked_result.result.platform,
                        "provider": ranked_result.result.provider,
                        "license": ranked_result.result.license,
                        "price": ranked_result.result.price,
                        "tags": ranked_result.result.tags,
                        "qualityScore": ranked_result.result.qualityScore,
                        "recency": ranked_result.result.recency,
                        "complexity": ranked_result.result.complexity,
                        "compatibility": ranked_result.result.compatibility,
                        "privacyRating": ranked_result.result.privacyRating,
                        "opensource": ranked_result.result.opensource,
                    },
                    "alignmentScore": ranked_result.alignmentScore,
                    "matchReasons": ranked_result.matchReasons,
                }
            )

        return RankingResponse(ranked_results=ranked_results)

    except Exception as e:
        logger.error(f"Error in ranking results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== URL RANKING ENDPOINT =====


@app.post("/rank-urls", response_model=URLRankingAPIResponse)
async def rank_urls_endpoint(request: URLRankingAPIRequest):
    """
    Rank URLs for a privacy-focused search engine.

    Accepts a search query and a list of URLs (e.g. 25), then scores and ranks
    them based on query relevance, privacy compliance, ethical alignment, and
    content quality.  Processing is parallelised for efficiency.

    Options (all optional, passed in the `options` dict):
      - weights: custom scoring weights (relevance, privacy, quality, ethics)
      - min_privacy_score: minimum privacy score threshold (0-1)
      - exclude_big_tech: boolean to filter out big-tech domains
    """
    url_ranking_requests.inc()

    try:
        from ranking.url_ranker import URLRankingRequest as InternalURLRankingRequest
        from ranking.url_ranker import (
            rank_urls,
        )

        internal_request = InternalURLRankingRequest(
            query=request.query,
            urls=request.urls,
            intent=request.intent,
            options=request.options,
        )

        response = await rank_urls(internal_request)

        ranked_results = [
            URLRankedResult(
                url=r.url,
                title=r.title,
                description=r.description,
                domain=r.domain,
                privacy_score=r.privacy_score,
                tracker_count=r.tracker_count,
                encryption_enabled=r.encryption_enabled,
                content_type=r.content_type,
                is_open_source=r.is_open_source,
                is_non_profit=r.is_non_profit,
                relevance_score=r.relevance_score,
                final_score=r.final_score,
            )
            for r in response.ranked_urls
        ]

        return URLRankingAPIResponse(
            query=response.query,
            ranked_urls=ranked_results,
            processing_time_ms=response.processing_time_ms,
            total_urls=response.total_urls,
            filtered_count=response.filtered_count,
        )

    except Exception as e:
        logger.error(f"Error in URL ranking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend-services", response_model=ServiceRecommendationResponse)
async def recommend_services_endpoint(request: ServiceRecommendationRequest):
    """
    Recommend services based on user intent
    """
    service_recommendation_requests.inc()  # Increment counter

    try:
        # Call the service recommendation function
        from services.recommender import ServiceMetadata
        from services.recommender import ServiceRecommendationRequest as InternalServiceRequest

        # Convert the request to internal format
        services_metadata = []
        for service in request.available_services:
            service_metadata = ServiceMetadata(
                id=service.get("id", ""),
                name=service.get("name", ""),
                supportedGoals=service.get("supportedGoals", []),
                primaryUseCases=service.get("primaryUseCases", []),
                temporalPatterns=service.get("temporalPatterns", []),
                ethicalAlignment=service.get("ethicalAlignment", []),
                description=service.get("description"),
            )
            services_metadata.append(service_metadata)

        internal_request = InternalServiceRequest(
            intent=request.intent, availableServices=services_metadata, options=request.options
        )

        response = recommend_services(internal_request)

        # Convert response to dict format
        recommendations = []
        for recommendation in response.recommendations:
            recommendations.append(
                {
                    "service": {
                        "id": recommendation.service.id,
                        "name": recommendation.service.name,
                        "supportedGoals": recommendation.service.supportedGoals,
                        "primaryUseCases": recommendation.service.primaryUseCases,
                        "temporalPatterns": recommendation.service.temporalPatterns,
                        "ethicalAlignment": recommendation.service.ethicalAlignment,
                        "description": recommendation.service.description,
                    },
                    "serviceScore": recommendation.serviceScore,
                    "matchReasons": recommendation.matchReasons,
                }
            )

        return ServiceRecommendationResponse(recommendations=recommendations)

    except Exception as e:
        logger.error(f"Error in service recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match-ads", response_model=AdMatchingResponse)
async def match_ads_endpoint(request: AdMatchingRequest, background_tasks: BackgroundTasks):
    """
    Match ads to user intent with fairness validation
    """
    db = None
    try:
        # Convert intent dict to UniversalIntent dataclass
        intent = convert_dict_to_universal_intent(request.intent)

        # Check if intent has expired
        if is_intent_expired(intent):
            raise HTTPException(status_code=400, detail="Intent has expired")

        # Anonymize intent data for privacy
        anonymized_intent = anonymize_intent_data(intent)

        # Fetch ad inventory from database
        db = next(db_manager.get_db())

        # Get all ads from the database
        db_ads = db.query(DbAd).all()

        # Convert database ads to internal AdMetadata format
        from ads.matcher import AdMetadata

        ad_inventory = []
        for db_ad in db_ads:
            ad_metadata = AdMetadata(
                id=str(db_ad.id),
                title=db_ad.title,
                description=db_ad.description,
                targetingConstraints=db_ad.targeting_constraints or {},
                forbiddenDimensions=[],  # This would come from ad validation
                qualityScore=db_ad.quality_score,
                ethicalTags=db_ad.ethical_tags or [],
                advertiser=f"advertiser_{db_ad.advertiser_id}",
                creative_format=db_ad.creative_format,
            )

            # Validate ad for privacy compliance
            compliance_report = validate_advertiser_constraints(ad_metadata)
            if not compliance_report["is_compliant"]:
                # Log violations but still include ad (would be filtered in production)
                logger.warning(f"Ad {db_ad.id} has compliance violations: {compliance_report['violations']}")
                fairness_violations.inc(len(compliance_report["violations"]))

            ad_inventory.append(ad_metadata)

        # Add any ads from the request as well
        for req_ad in request.ad_inventory:
            ad_metadata = AdMetadata(
                id=req_ad.get("id", ""),
                title=req_ad.get("title", ""),
                description=req_ad.get("description", ""),
                targetingConstraints=req_ad.get("targetingConstraints", {}),
                forbiddenDimensions=req_ad.get("forbiddenDimensions", []),
                qualityScore=req_ad.get("qualityScore", 0.5),
                ethicalTags=req_ad.get("ethicalTags", []),
                advertiser=req_ad.get("advertiser"),
                category=req_ad.get("category"),
                creative_format=req_ad.get("creative_format"),
            )

            # Validate ad for privacy compliance
            compliance_report = validate_advertiser_constraints(ad_metadata)
            if not compliance_report["is_compliant"]:
                # Log violations but still include ad (would be filtered in production)
                logger.warning(
                    f"Ad {req_ad.get('id', 'unknown')} has compliance violations: {compliance_report['violations']}"
                )
                fairness_violations.inc(len(compliance_report["violations"]))

            ad_inventory.append(ad_metadata)

        # Prepare internal request
        from ads.matcher import AdMatchingRequest as InternalAdRequest

        internal_request = InternalAdRequest(intent=anonymized_intent, adInventory=ad_inventory, config=request.config)

        # Perform ad matching
        response = match_ads(internal_request)

        # Log metrics to database in background
        background_tasks.add_task(log_ad_metrics, anonymized_intent, response.matchedAds)

        # Convert response to dict format
        matched_ads = []
        for matched_ad in response.matchedAds:
            matched_ads.append(
                {
                    "ad": {
                        "id": matched_ad.ad.id,
                        "title": matched_ad.ad.title,
                        "description": matched_ad.ad.description,
                        "targetingConstraints": matched_ad.ad.targetingConstraints,
                        "forbiddenDimensions": matched_ad.ad.forbiddenDimensions,
                        "qualityScore": matched_ad.ad.qualityScore,
                        "ethicalTags": matched_ad.ad.ethicalTags,
                        "advertiser": matched_ad.ad.advertiser,
                        "category": matched_ad.ad.category,
                        "creative_format": matched_ad.ad.creative_format,
                    },
                    "adRelevanceScore": matched_ad.adRelevanceScore,
                    "matchReasons": matched_ad.matchReasons,
                }
            )

        return AdMatchingResponse(matched_ads=matched_ads, metrics=response.metrics)

    except HTTPException:
        # Re-raise HTTPException to preserve status code (400, 404, etc.)
        raise
    except Exception as e:
        logger.error(f"Error in ad matching: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e) or "Internal server error")
    finally:
        if db:
            db.close()


# Campaign Management Endpoints
@app.post("/campaigns", response_model=Campaign)
async def create_campaign(campaign: CampaignCreate, db: Session = Depends(get_db)):
    """
    Create new campaign
    """
    db_campaign = DbCampaign(
        advertiser_id=campaign.advertiser_id,
        name=campaign.name,
        start_date=campaign.start_date,
        end_date=campaign.end_date,
        budget=campaign.budget,
        daily_budget=campaign.daily_budget,
        status=campaign.status,
    )
    db.add(db_campaign)
    db.commit()
    db.refresh(db_campaign)
    return db_campaign


@app.get("/campaigns/{campaign_id}", response_model=Campaign)
async def get_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """
    Get campaign details
    """
    campaign = db.query(DbCampaign).filter(DbCampaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign


@app.put("/campaigns/{campaign_id}", response_model=Campaign)
async def update_campaign(campaign_id: int, campaign_update: CampaignUpdate, db: Session = Depends(get_db)):
    """
    Update campaign
    """
    campaign = db.query(DbCampaign).filter(DbCampaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    update_data = campaign_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(campaign, field, value)

    db.commit()
    db.refresh(campaign)
    return campaign


@app.delete("/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """
    Delete campaign
    """
    campaign = db.query(DbCampaign).filter(DbCampaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # First delete associated ad groups and ads
    ad_groups = db.query(DbAdGroup).filter(DbAdGroup.campaign_id == campaign_id).all()
    for ad_group in ad_groups:
        ads = db.query(DbAd).filter(DbAd.ad_group_id == ad_group.id).all()
        for ad in ads:
            # Delete creative assets first
            creative_assets = db.query(DbCreativeAsset).filter(DbCreativeAsset.ad_id == ad.id).all()
            for asset in creative_assets:
                db.delete(asset)
            # Then delete the ad
            db.delete(ad)
        # Then delete the ad group
        db.delete(ad_group)

    # Finally delete the campaign
    db.delete(campaign)
    db.commit()
    return {"message": "Campaign deleted successfully"}


@app.get("/campaigns", response_model=List[Campaign])
async def list_campaigns(
    advertiser_id: Optional[int] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    List campaigns with filters
    """
    query = db.query(DbCampaign)

    if advertiser_id:
        query = query.filter(DbCampaign.advertiser_id == advertiser_id)

    if status:
        query = query.filter(DbCampaign.status == status)

    campaigns = query.offset(skip).limit(limit).all()
    return campaigns


# Ad Group Management Endpoints
@app.post("/adgroups", response_model=AdGroup)
async def create_ad_group(ad_group: AdGroupCreate, db: Session = Depends(get_db)):
    """
    Create ad group
    """
    db_ad_group = DbAdGroup(
        campaign_id=ad_group.campaign_id,
        name=ad_group.name,
        targeting_settings=ad_group.targeting_settings,
        bid_strategy=ad_group.bid_strategy,
    )
    db.add(db_ad_group)
    db.commit()
    db.refresh(db_ad_group)
    return db_ad_group


@app.get("/adgroups/{ad_group_id}", response_model=AdGroup)
async def get_ad_group(ad_group_id: int, db: Session = Depends(get_db)):
    """
    Get ad group details
    """
    ad_group = db.query(DbAdGroup).filter(DbAdGroup.id == ad_group_id).first()
    if not ad_group:
        raise HTTPException(status_code=404, detail="Ad group not found")
    return ad_group


@app.put("/adgroups/{ad_group_id}", response_model=AdGroup)
async def update_ad_group(ad_group_id: int, ad_group_update: AdGroupUpdate, db: Session = Depends(get_db)):
    """
    Update ad group
    """
    ad_group = db.query(DbAdGroup).filter(DbAdGroup.id == ad_group_id).first()
    if not ad_group:
        raise HTTPException(status_code=404, detail="Ad group not found")

    update_data = ad_group_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(ad_group, field, value)

    db.commit()
    db.refresh(ad_group)
    return ad_group


@app.get("/adgroups", response_model=List[AdGroup])
async def list_ad_groups(
    campaign_id: Optional[int] = None, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    """
    List ad groups
    """
    query = db.query(DbAdGroup)

    if campaign_id:
        query = query.filter(DbAdGroup.campaign_id == campaign_id)

    ad_groups = query.offset(skip).limit(limit).all()
    return ad_groups


# Ads Management Endpoints
@app.post("/ads", response_model=Ad)
async def create_ad(ad: AdCreate, db: Session = Depends(get_db)):
    """
    Create a new ad
    """
    # Verify advertiser exists
    advertiser = db.query(DbAdvertiser).filter(DbAdvertiser.id == ad.advertiser_id).first()
    if not advertiser:
        raise HTTPException(status_code=404, detail="Advertiser not found")

    # Verify ad group exists if provided
    if ad.ad_group_id:
        ad_group = db.query(DbAdGroup).filter(DbAdGroup.id == ad.ad_group_id).first()
        if not ad_group:
            raise HTTPException(status_code=404, detail="Ad group not found")

    db_ad = DbAd(
        advertiser_id=ad.advertiser_id,
        ad_group_id=ad.ad_group_id,
        title=ad.title,
        description=ad.description,
        url=ad.url,
        targeting_constraints=ad.targeting_constraints,
        ethical_tags=ad.ethical_tags,
        quality_score=ad.quality_score,
        creative_format=ad.creative_format,
        bid_amount=ad.bid_amount,
        status=ad.status,
        approval_status=ad.approval_status,
    )
    db.add(db_ad)
    db.commit()
    db.refresh(db_ad)
    return db_ad


@app.get("/ads/{ad_id}", response_model=Ad)
async def get_ad(ad_id: int, db: Session = Depends(get_db)):
    """
    Get ad details
    """
    ad = db.query(DbAd).filter(DbAd.id == ad_id).first()
    if not ad:
        raise HTTPException(status_code=404, detail="Ad not found")
    return ad


@app.put("/ads/{ad_id}", response_model=Ad)
async def update_ad(ad_id: int, ad_update: AdUpdate, db: Session = Depends(get_db)):
    """
    Update an ad
    """
    ad = db.query(DbAd).filter(DbAd.id == ad_id).first()
    if not ad:
        raise HTTPException(status_code=404, detail="Ad not found")

    update_data = ad_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:
            setattr(ad, field, value)

    db.commit()
    db.refresh(ad)
    return ad


@app.delete("/ads/{ad_id}")
async def delete_ad(ad_id: int, db: Session = Depends(get_db)):
    """
    Delete an ad
    """
    ad = db.query(DbAd).filter(DbAd.id == ad_id).first()
    if not ad:
        raise HTTPException(status_code=404, detail="Ad not found")

    # Delete associated click tracking and conversions first
    clicks = db.query(DbClickTracking).filter(DbClickTracking.ad_id == ad_id).all()
    for click in clicks:
        db.query(DbConversionTracking).filter(DbConversionTracking.click_id == click.id).delete()
    db.query(DbClickTracking).filter(DbClickTracking.ad_id == ad_id).delete()

    # Delete associated fraud detection records
    db.query(DbFraudDetection).filter(DbFraudDetection.ad_id == ad_id).delete()

    # Delete associated creative assets
    db.query(DbCreativeAsset).filter(DbCreativeAsset.ad_id == ad_id).delete()

    # Delete associated metrics
    db.query(DbAdMetric).filter(DbAdMetric.ad_id == ad_id).delete()

    # Delete the ad
    db.delete(ad)
    db.commit()
    return {"message": "Ad deleted successfully"}


@app.get("/ads", response_model=List[Ad])
async def list_ads(
    advertiser_id: Optional[int] = None,
    ad_group_id: Optional[int] = None,
    status: Optional[str] = None,
    approval_status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    List ads with optional filters
    """
    query = db.query(DbAd)

    if advertiser_id:
        query = query.filter(DbAd.advertiser_id == advertiser_id)

    if ad_group_id:
        query = query.filter(DbAd.ad_group_id == ad_group_id)

    if status:
        query = query.filter(DbAd.status == status)

    if approval_status:
        query = query.filter(DbAd.approval_status == approval_status)

    ads = query.offset(skip).limit(limit).all()
    return ads


# Advertiser Management Endpoints
@app.post("/advertisers", response_model=Advertiser)
async def create_advertiser(advertiser: AdvertiserCreate, db: Session = Depends(get_db)):
    """
    Create a new advertiser
    """
    db_advertiser = DbAdvertiser(name=advertiser.name, contact_email=advertiser.contact_email)
    db.add(db_advertiser)
    db.commit()
    db.refresh(db_advertiser)
    return db_advertiser


@app.get("/advertisers/{advertiser_id}", response_model=Advertiser)
async def get_advertiser(advertiser_id: int, db: Session = Depends(get_db)):
    """
    Get advertiser details
    """
    advertiser = db.query(DbAdvertiser).filter(DbAdvertiser.id == advertiser_id).first()
    if not advertiser:
        raise HTTPException(status_code=404, detail="Advertiser not found")
    return advertiser


@app.get("/advertisers", response_model=List[Advertiser])
async def list_advertisers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    List all advertisers
    """
    advertisers = db.query(DbAdvertiser).offset(skip).limit(limit).all()
    return advertisers


# Creative Management Endpoints
@app.post("/creatives", response_model=CreativeAsset)
async def upload_creative_asset(creative: CreativeAssetCreate, db: Session = Depends(get_db)):
    """
    Upload creative assets
    """
    # Verify ad exists
    ad = db.query(DbAd).filter(DbAd.id == creative.ad_id).first()
    if not ad:
        raise HTTPException(status_code=404, detail="Ad not found")

    db_creative = DbCreativeAsset(
        ad_id=creative.ad_id,
        asset_type=creative.asset_type,
        asset_url=creative.asset_url,
        dimensions=creative.dimensions,
        checksum=creative.checksum,
    )
    db.add(db_creative)
    db.commit()
    db.refresh(db_creative)
    return db_creative


@app.get("/creatives/{creative_id}", response_model=CreativeAsset)
async def get_creative_asset(creative_id: int, db: Session = Depends(get_db)):
    """
    Get creative details
    """
    creative = db.query(DbCreativeAsset).filter(DbCreativeAsset.id == creative_id).first()
    if not creative:
        raise HTTPException(status_code=404, detail="Creative asset not found")
    return creative


@app.put("/creatives/{creative_id}", response_model=CreativeAsset)
async def update_creative_asset(creative_id: int, creative_update: CreativeAssetUpdate, db: Session = Depends(get_db)):
    """
    Update creative
    """
    creative = db.query(DbCreativeAsset).filter(DbCreativeAsset.id == creative_id).first()
    if not creative:
        raise HTTPException(status_code=404, detail="Creative asset not found")

    update_data = creative_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(creative, field, value)

    db.commit()
    db.refresh(creative)
    return creative


@app.delete("/creatives/{creative_id}")
async def delete_creative_asset(creative_id: int, db: Session = Depends(get_db)):
    """
    Delete creative
    """
    creative = db.query(DbCreativeAsset).filter(DbCreativeAsset.id == creative_id).first()
    if not creative:
        raise HTTPException(status_code=404, detail="Creative asset not found")

    db.delete(creative)
    db.commit()
    return {"message": "Creative asset deleted successfully"}


# Reporting & Analytics Endpoints
@app.get("/reports/campaign-performance", response_model=List[CampaignPerformanceReport])
async def get_campaign_performance(
    campaign_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
):
    """
    Campaign metrics
    """
    # Query to aggregate metrics by campaign
    query = (
        db.query(
            DbCampaign.id.label("campaign_id"),
            DbCampaign.name.label("campaign_name"),
            func.sum(DbAdMetric.impression_count).label("impressions"),
            func.sum(DbAdMetric.click_count).label("clicks"),
            func.sum(DbAdMetric.conversion_count).label("conversions"),
            func.avg(DbAdMetric.ctr).label("ctr"),
            func.avg(DbAdMetric.cpc).label("cpc"),
            func.sum(DbAdMetric.impression_count * DbAd.bid_amount).label("cost"),
            func.avg(DbAdMetric.roas).label("roas"),
        )
        .select_from(DbCampaign)
        .join(DbAdGroup, DbCampaign.id == DbAdGroup.campaign_id)
        .join(DbAd, DbAdGroup.id == DbAd.ad_group_id)
        .join(DbAdMetric, DbAd.id == DbAdMetric.ad_id)
    )

    if campaign_id:
        query = query.filter(DbCampaign.id == campaign_id)

    if start_date:
        query = query.filter(DbAdMetric.date >= start_date)

    if end_date:
        query = query.filter(DbAdMetric.date <= end_date)

    results = query.group_by(DbCampaign.id, DbCampaign.name).all()

    reports = []
    for result in results:
        reports.append(
            CampaignPerformanceReport(
                campaign_id=result.campaign_id,
                campaign_name=result.campaign_name,
                impressions=result.impressions or 0,
                clicks=result.clicks or 0,
                conversions=result.conversions or 0,
                ctr=result.ctr or 0.0,
                cpc=result.cpc or 0.0,
                cost=result.cost or 0.0,
                roas=result.roas or 0.0,
            )
        )

    return reports


# Enhanced Ad Matching Endpoint with Campaign Context
@app.post("/match-ads-advanced", response_model=AdMatchingResponse)
async def match_ads_advanced(
    request: AdMatchingWithCampaignRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """
    Advanced matching with campaign context
    """
    try:
        # Convert intent dict to UniversalIntent dataclass
        intent = convert_dict_to_universal_intent(request.intent)

        # Check if intent has expired
        if is_intent_expired(intent):
            raise HTTPException(status_code=400, detail="Intent has expired")

        # Anonymize intent data for privacy
        anonymized_intent = anonymize_intent_data(intent)

        # Fetch ad inventory from database with campaign context
        query = db.query(DbAd).join(DbAdGroup).join(DbCampaign)

        # Apply campaign filters based on context
        if request.campaign_context:
            if "campaign_ids" in request.campaign_context:
                query = query.filter(DbCampaign.id.in_(request.campaign_context["campaign_ids"]))

            if "budget_constraint" in request.campaign_context:
                # Only include campaigns with budget remaining
                # Calculate spent budget per campaign by joining with AdMetrics
                try:
                    # Get all active campaigns
                    active_campaigns = db.query(DbCampaign).filter(DbCampaign.status == "active").all()

                    # If there are no active campaigns, skip the budget constraint
                    if not active_campaigns:
                        pass  # Skip budget constraint if no active campaigns
                    else:
                        eligible_campaign_ids = []
                        for campaign in active_campaigns:
                            try:
                                # Calculate total spent for this campaign using a subquery approach
                                campaign_spent = (
                                    db.query(func.coalesce(func.sum(DbAdMetric.impression_count * DbAd.bid_amount), 0))
                                    .select_from(DbAdMetric)
                                    .join(DbAd, DbAdMetric.ad_id == DbAd.id)
                                    .join(DbAdGroup, DbAd.ad_group_id == DbAdGroup.id)
                                    .filter(DbAdGroup.campaign_id == campaign.id)
                                    .scalar()
                                    or 0
                                )

                                # If campaign has remaining budget, add to eligible list
                                if campaign_spent < campaign.budget:
                                    eligible_campaign_ids.append(campaign.id)
                            except Exception as inner_e:
                                # If there's an issue calculating spend for a specific campaign, log it and continue
                                logger.warning(f"Could not calculate spend for campaign {campaign.id}: {str(inner_e)}")
                                # Add the campaign anyway to avoid excluding it due to calculation error
                                eligible_campaign_ids.append(campaign.id)

                        # If there are eligible campaigns, filter the main query
                        if eligible_campaign_ids:
                            query = query.filter(DbCampaign.id.in_(eligible_campaign_ids))
                        else:
                            # If no campaigns have budget left, we might want to return no results
                            # Or we could skip the budget constraint in this case
                            # For now, let's skip the constraint to avoid returning no results
                            pass

                except Exception as e:
                    # If there's an issue with the budget constraint query, log it and continue without the constraint
                    logger.error(f"Budget constraint query failed: {str(e)}")
                    # Continue without applying the budget constraint
                    pass

        # Get eligible ads from active campaigns
        db_ads = query.filter(
            DbCampaign.status == "active", DbAd.status == "active", DbAd.approval_status == "approved"
        ).all()

        # Convert database ads to internal AdMetadata format
        from ads.matcher import AdMetadata

        ad_inventory = []
        for db_ad in db_ads:
            ad_metadata = AdMetadata(
                id=str(db_ad.id),
                title=db_ad.title,
                description=db_ad.description,
                targetingConstraints=db_ad.targeting_constraints or {},
                forbiddenDimensions=[],  # This would come from ad validation
                qualityScore=db_ad.quality_score,
                ethicalTags=db_ad.ethical_tags or [],
                advertiser=f"advertiser_{db_ad.advertiser_id}",
                creative_format=db_ad.creative_format,
            )

            # Validate ad for privacy compliance
            compliance_report = validate_advertiser_constraints(ad_metadata)
            if not compliance_report["is_compliant"]:
                # Log violations but still include ad (would be filtered in production)
                logger.warning(f"Ad {db_ad.id} has compliance violations: {compliance_report['violations']}")
                fairness_violations.inc(len(compliance_report["violations"]))

            ad_inventory.append(ad_metadata)

        # Add any ads from the request as well
        for req_ad in request.ad_inventory:
            ad_metadata = AdMetadata(
                id=req_ad.get("id", ""),
                title=req_ad.get("title", ""),
                description=req_ad.get("description", ""),
                targetingConstraints=req_ad.get("targetingConstraints", {}),
                forbiddenDimensions=req_ad.get("forbiddenDimensions", []),
                qualityScore=req_ad.get("qualityScore", 0.5),
                ethicalTags=req_ad.get("ethicalTags", []),
                advertiser=req_ad.get("advertiser"),
                creative_format=req_ad.get("creative_format"),
            )

            # Validate ad for privacy compliance
            compliance_report = validate_advertiser_constraints(ad_metadata)
            if not compliance_report["is_compliant"]:
                # Log violations but still include ad (would be filtered in production)
                logger.warning(
                    f"Ad {req_ad.get('id', 'unknown')} has compliance violations: {compliance_report['violations']}"
                )
                fairness_violations.inc(len(compliance_report["violations"]))

            ad_inventory.append(ad_metadata)

        # Prepare internal request
        from ads.matcher import AdMatchingRequest as InternalAdRequest

        internal_request = InternalAdRequest(intent=anonymized_intent, adInventory=ad_inventory, config=request.config)

        # Perform ad matching
        response = match_ads(internal_request)

        # Log metrics to database in background
        background_tasks.add_task(log_ad_metrics, anonymized_intent, response.matchedAds)

        # Convert response to dict format
        matched_ads = []
        for matched_ad in response.matchedAds:
            matched_ads.append(
                {
                    "ad": {
                        "id": matched_ad.ad.id,
                        "title": matched_ad.ad.title,
                        "description": matched_ad.ad.description,
                        "targetingConstraints": matched_ad.ad.targetingConstraints,
                        "forbiddenDimensions": matched_ad.ad.forbiddenDimensions,
                        "qualityScore": matched_ad.ad.qualityScore,
                        "ethicalTags": matched_ad.ad.ethicalTags,
                        "advertiser": matched_ad.ad.advertiser,
                        "creative_format": matched_ad.ad.creative_format,
                    },
                    "adRelevanceScore": matched_ad.adRelevanceScore,
                    "matchReasons": matched_ad.matchReasons,
                }
            )

        return AdMatchingResponse(matched_ads=matched_ads, metrics=response.metrics)

    except Exception as e:
        logger.error(f"Error in advanced ad matching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Click Tracking Endpoints
@app.post("/click-tracking", response_model=ClickTracking)
async def track_click(click_data: ClickTrackingCreate, db: Session = Depends(get_db)):
    """
    Record an ad click
    """
    # Verify ad exists
    ad = db.query(DbAd).filter(DbAd.id == click_data.ad_id).first()
    if not ad:
        raise HTTPException(status_code=404, detail="Ad not found")

    db_click = DbClickTracking(
        ad_id=click_data.ad_id,
        session_id=click_data.session_id,
        ip_hash=click_data.ip_hash,
        user_agent_hash=click_data.user_agent_hash,
        referring_url=click_data.referring_url,
    )
    db.add(db_click)
    db.commit()
    db.refresh(db_click)
    return db_click


@app.get("/click-tracking/{click_id}", response_model=ClickTracking)
async def get_click(click_id: int, db: Session = Depends(get_db)):
    """
    Get click details
    """
    click = db.query(DbClickTracking).filter(DbClickTracking.id == click_id).first()
    if not click:
        raise HTTPException(status_code=404, detail="Click not found")
    return click


@app.get("/click-tracking", response_model=List[ClickTracking])
async def list_clicks(
    ad_id: Optional[int] = None,
    session_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    List clicks with optional filters
    """
    query = db.query(DbClickTracking)

    if ad_id:
        query = query.filter(DbClickTracking.ad_id == ad_id)

    if session_id:
        query = query.filter(DbClickTracking.session_id == session_id)

    clicks = query.offset(skip).limit(limit).all()
    return clicks


# Conversion Tracking Endpoints
@app.post("/conversion-tracking", response_model=ConversionTracking)
async def track_conversion(conversion_data: ConversionTrackingCreate, db: Session = Depends(get_db)):
    """
    Record a conversion
    """
    # Verify click exists
    click = db.query(DbClickTracking).filter(DbClickTracking.id == conversion_data.click_id).first()
    if not click:
        raise HTTPException(status_code=404, detail="Click not found")

    db_conversion = DbConversionTracking(
        click_id=conversion_data.click_id,
        conversion_type=conversion_data.conversion_type,
        value=conversion_data.value,
        status=conversion_data.status,
    )
    db.add(db_conversion)
    db.commit()
    db.refresh(db_conversion)
    return db_conversion


@app.get("/conversion-tracking/{conversion_id}", response_model=ConversionTracking)
async def get_conversion(conversion_id: int, db: Session = Depends(get_db)):
    """
    Get conversion details
    """
    conversion = db.query(DbConversionTracking).filter(DbConversionTracking.id == conversion_id).first()
    if not conversion:
        raise HTTPException(status_code=404, detail="Conversion not found")
    return conversion


@app.get("/conversion-tracking", response_model=List[ConversionTracking])
async def list_conversions(
    click_id: Optional[int] = None,
    conversion_type: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    List conversions with optional filters
    """
    query = db.query(DbConversionTracking)

    if click_id:
        query = query.filter(DbConversionTracking.click_id == click_id)

    if conversion_type:
        query = query.filter(DbConversionTracking.conversion_type == conversion_type)

    if status:
        query = query.filter(DbConversionTracking.status == status)

    conversions = query.offset(skip).limit(limit).all()
    return conversions


# Fraud Detection Endpoints
@app.post("/fraud-detection", response_model=FraudDetection)
async def report_fraud(fraud_data: FraudDetectionCreate, db: Session = Depends(get_db)):
    """
    Report a potential fraud event
    """
    # Verify ad exists if provided
    if fraud_data.ad_id:
        ad = db.query(DbAd).filter(DbAd.id == fraud_data.ad_id).first()
        if not ad:
            raise HTTPException(status_code=404, detail="Ad not found")

    db_fraud = DbFraudDetection(
        event_id=fraud_data.event_id,
        event_type=fraud_data.event_type,
        reason=fraud_data.reason,
        severity=fraud_data.severity,
        review_status=fraud_data.review_status,
        ad_id=fraud_data.ad_id,
    )
    db.add(db_fraud)
    db.commit()
    db.refresh(db_fraud)
    return db_fraud


@app.get("/fraud-detection/{fraud_id}", response_model=FraudDetection)
async def get_fraud_report(fraud_id: int, db: Session = Depends(get_db)):
    """
    Get fraud report details
    """
    fraud_report = db.query(DbFraudDetection).filter(DbFraudDetection.id == fraud_id).first()
    if not fraud_report:
        raise HTTPException(status_code=404, detail="Fraud report not found")
    return fraud_report


@app.put("/fraud-detection/{fraud_id}", response_model=FraudDetection)
async def update_fraud_report(fraud_id: int, fraud_update: FraudDetectionCreate, db: Session = Depends(get_db)):
    """
    Update fraud report status
    """
    fraud_report = db.query(DbFraudDetection).filter(DbFraudDetection.id == fraud_id).first()
    if not fraud_report:
        raise HTTPException(status_code=404, detail="Fraud report not found")

    update_data = fraud_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(fraud_report, field, value)

    db.commit()
    db.refresh(fraud_report)
    return fraud_report


@app.get("/fraud-detection", response_model=List[FraudDetection])
async def list_fraud_reports(
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    review_status: Optional[str] = None,
    ad_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    List fraud reports with optional filters
    """
    query = db.query(DbFraudDetection)

    if event_type:
        query = query.filter(DbFraudDetection.event_type == event_type)

    if severity:
        query = query.filter(DbFraudDetection.severity == severity)

    if review_status:
        query = query.filter(DbFraudDetection.review_status == review_status)

    if ad_id:
        query = query.filter(DbFraudDetection.ad_id == ad_id)

    fraud_reports = query.offset(skip).limit(limit).all()
    return fraud_reports


def log_ad_metrics(intent: UniversalIntent, matched_ads: List[Any]):
    """
    Log aggregated metrics to ad_metrics table with TTL = 30 days
    This runs in the background to not block the main request
    """
    try:
        db = next(db_manager.get_db())

        for matched_ad in matched_ads:
            # Create a metric record for each matched ad
            metric = DbAdMetric(
                ad_id=int(matched_ad.ad.id) if matched_ad.ad.id.isdigit() else 0,  # Simplified for demo
                date=datetime.utcnow().date(),
                intent_goal=intent.declared.goal.value if intent.declared.goal else None,
                intent_use_case=intent.inferred.useCases[0].value if intent.inferred.useCases else None,
                impression_count=1,  # For now, count each match as an impression
                click_count=0,  # Would be incremented separately
                conversion_count=0,  # Would be incremented separately
                expires_at=datetime.utcnow() + timedelta(days=30),
            )

            db.add(metric)

        db.commit()
        logger.info(f"Logged metrics for {len(matched_ads)} ads")

    except Exception as e:
        logger.error(f"Error logging ad metrics: {str(e)}")
        db.rollback()
    finally:
        db.close()


@app.get("/metrics")
def get_metrics():
    """Endpoint to expose Prometheus metrics"""
    return Response(generate_latest(), media_type="text/plain")


# Real-time Analytics Endpoints
@app.websocket("/ws/analytics")
async def websocket_analytics(websocket: WebSocket):
    """WebSocket endpoint for real-time analytics"""
    db = next(db_manager.get_db())
    try:
        await handle_analytics_websocket(websocket, db)
    except Exception as e:
        logger.error(f"WebSocket analytics error: {str(e)}")
    finally:
        db.close()


# Enhanced Privacy Controls Endpoints
@app.post("/privacy-controls/apply-retention-policy", response_model=DataRetentionPolicy)
async def apply_data_retention_policy(
    data_type: str, retention_period: DataRetentionPeriod, db: Session = Depends(get_db)
):
    """Apply data retention policy"""
    privacy_controls = get_enhanced_privacy_controls(db)
    result = privacy_controls.apply_data_retention_policy(data_type, retention_period)

    # Log the action
    audit_manager = get_audit_trail_manager(db)
    audit_manager.log_event(
        user_id="system",
        event_type=AuditEventType.PRIVACY_SETTING_CHANGE,
        resource_type="privacy_policy",
        action_description=f"Applied retention policy for {data_type} with period {retention_period.value}",
        metadata=result,
    )

    return result


@app.get("/privacy-controls/compliance-report", response_model=PrivacyComplianceReport)
async def get_privacy_compliance_report(db: Session = Depends(get_db)):
    """Get privacy compliance report"""
    privacy_controls = get_enhanced_privacy_controls(db)
    report = privacy_controls.get_privacy_compliance_report()

    # Log the action
    audit_manager = get_audit_trail_manager(db)
    audit_manager.log_event(
        user_id="system",
        event_type=AuditEventType.ADMIN_ACTION,
        resource_type="privacy_report",
        action_description="Generated privacy compliance report",
        metadata=report,
    )

    return report


# Consent Management Endpoints
@app.post("/consent/record", response_model=ConsentRecord)
async def record_user_consent(
    user_id: str,
    consent_type: ConsentType,
    granted: bool,
    consent_details: Optional[Dict[str, Any]] = None,
    expires_in_days: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Record user consent"""
    consent_manager = get_consent_manager(db)
    consent_record = consent_manager.record_consent(user_id, consent_type, granted, consent_details, expires_in_days)

    # Log the action
    audit_manager = get_audit_trail_manager(db)
    audit_manager.log_event(
        user_id=user_id,
        event_type=AuditEventType.CONSENT_GIVEN if granted else AuditEventType.CONSENT_WITHDRAWN,
        resource_type="consent",
        resource_id=consent_record.id,
        action_description=f"User {'granted' if granted else 'denied'} consent for {consent_type.value}",
        metadata={"consent_type": consent_type.value, "granted": granted},
    )

    return consent_record


@app.get("/consent/{user_id}/{consent_type}", response_model=Optional[ConsentRecord])
async def get_user_consent(user_id: str, consent_type: ConsentType, db: Session = Depends(get_db)):
    """Get user consent for specific type"""
    consent_manager = get_consent_manager(db)
    consent = consent_manager.get_user_consent(user_id, consent_type)

    # Log the action
    audit_manager = get_audit_trail_manager(db)
    audit_manager.log_event(
        user_id=user_id,
        event_type=AuditEventType.ADMIN_ACTION,
        resource_type="consent",
        action_description=f"Retrieved consent for {consent_type.value}",
        metadata={"consent_type": consent_type.value},
    )

    return consent


@app.post("/consent/withdraw/{user_id}/{consent_type}")
async def withdraw_user_consent(user_id: str, consent_type: ConsentType, db: Session = Depends(get_db)):
    """Withdraw user consent"""
    consent_manager = get_consent_manager(db)
    success = consent_manager.withdraw_consent(user_id, consent_type)

    if success:
        # Log the action
        audit_manager = get_audit_trail_manager(db)
        audit_manager.log_event(
            user_id=user_id,
            event_type=AuditEventType.CONSENT_WITHDRAWN,
            resource_type="consent",
            action_description=f"Withdrew consent for {consent_type.value}",
            metadata={"consent_type": consent_type.value},
        )

    return {"success": success}


@app.get("/consent-summary", response_model=ConsentSummary)
async def get_consent_summary(db: Session = Depends(get_db)):
    """Get consent summary for the entire system"""
    consent_manager = get_consent_manager(db)
    summary = consent_manager.get_consent_summary()

    # Log the action
    audit_manager = get_audit_trail_manager(db)
    audit_manager.log_event(
        user_id="system",
        event_type=AuditEventType.ADMIN_ACTION,
        resource_type="consent_summary",
        action_description="Generated consent summary report",
        metadata=summary,
    )

    return summary


# Audit Trail Endpoints
@app.post("/audit-log", response_model=AuditEvent)
async def log_audit_event(
    user_id: Optional[str] = None,
    event_type: AuditEventType = AuditEventType.ADMIN_ACTION,
    resource_type: Optional[str] = None,
    resource_id: Optional[int] = None,
    action_description: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
):
    """Manually log an audit event"""
    audit_manager = get_audit_trail_manager(db)
    audit_entry = audit_manager.log_event(
        user_id, event_type, resource_type, resource_id, action_description, ip_address, user_agent, metadata
    )

    return audit_entry


@app.get("/audit-events", response_model=List[AuditEvent])
async def get_audit_events(
    user_id: Optional[str] = None,
    event_type: Optional[str] = None,
    resource_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Get audit events with optional filters"""
    audit_manager = get_audit_trail_manager(db)

    # Convert event_type string to enum if provided
    event_type_enum = None
    if event_type:
        try:
            event_type_enum = AuditEventType(event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")

    # Convert end_date string to datetime if provided
    end_date_dt = None
    if end_date:
        try:
            from datetime import datetime

            end_date_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {end_date}")

    events = audit_manager.get_audit_events(
        user_id, event_type_enum, resource_type, start_date, end_date_dt, limit, offset
    )

    return events


@app.get("/audit-stats", response_model=AuditStats)
async def get_audit_statistics(db: Session = Depends(get_db)):
    """Get audit event statistics"""
    audit_manager = get_audit_trail_manager(db)
    stats = audit_manager.get_event_statistics()

    return stats


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get service status"""
    return StatusResponse(
        service="Intent Engine API",
        version="1.0.0",
        uptime="N/A",  # Would track actual uptime in a real implementation
        status="running",
    )


# ===== FRAUD DETECTION ENDPOINTS =====


@app.post("/fraud/analyze-click", response_model=FraudAnalysisResponse)
async def analyze_click_fraud(click_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Analyze a click for potential fraud"""
    from fraud.detector import get_fraud_detector

    detector = get_fraud_detector(db)
    result = detector.analyze_click(click_data)

    return FraudAnalysisResponse(
        is_fraudulent=result.is_fraudulent,
        risk_score=result.risk_score,
        signals=[
            {
                "type": s.fraud_type.value,
                "severity": s.severity.value,
                "confidence": s.confidence,
                "reason": s.reason,
                "metadata": s.metadata,
            }
            for s in result.signals
        ],
        recommended_action=result.recommended_action,
    )


@app.post("/fraud/analyze-conversion", response_model=FraudAnalysisResponse)
async def analyze_conversion_fraud(conversion_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Analyze a conversion for potential fraud"""
    from fraud.detector import get_fraud_detector

    detector = get_fraud_detector(db)
    result = detector.analyze_conversion(conversion_data)

    return FraudAnalysisResponse(
        is_fraudulent=result.is_fraudulent,
        risk_score=result.risk_score,
        signals=[
            {
                "type": s.fraud_type.value,
                "severity": s.severity.value,
                "confidence": s.confidence,
                "reason": s.reason,
                "metadata": s.metadata,
            }
            for s in result.signals
        ],
        recommended_action=result.recommended_action,
    )


@app.post("/fraud/run-scan", response_model=FraudScanSummary)
async def run_fraud_scan(hours: int = 24, db: Session = Depends(get_db)):
    """Run batch fraud analysis on recent events"""
    from fraud.detector import get_fraud_detector

    detector = get_fraud_detector(db)
    summary = detector.run_batch_analysis(hours)

    return FraudScanSummary(**summary)


# ===== A/B TESTING ENDPOINTS =====


@app.post("/ab-tests", response_model=ABTestResponse)
async def create_ab_test(test_data: ABTestCreate, db: Session = Depends(get_db)):
    """Create a new A/B test"""
    from abtesting.service import get_ab_test_service

    service = get_ab_test_service(db)
    test = service.create_test(
        name=test_data.name,
        campaign_id=test_data.campaign_id,
        description=test_data.description or "",
        traffic_allocation=test_data.traffic_allocation,
        min_sample_size=test_data.min_sample_size,
        confidence_level=test_data.confidence_level,
        primary_metric=test_data.primary_metric,
    )

    return test


@app.get("/ab-tests", response_model=List[ABTestResponse])
async def list_ab_tests(campaign_id: Optional[int] = None, status: Optional[str] = None, db: Session = Depends(get_db)):
    """List all A/B tests"""
    from abtesting.service import get_ab_test_service

    service = get_ab_test_service(db)
    tests = service.get_all_tests(campaign_id, status)

    return tests


@app.get("/ab-tests/{test_id}", response_model=ABTestResponse)
async def get_ab_test(test_id: int, db: Session = Depends(get_db)):
    """Get a specific A/B test"""
    from abtesting.service import get_ab_test_service

    service = get_ab_test_service(db)
    test = service.get_test(test_id)

    if not test:
        raise HTTPException(status_code=404, detail="A/B test not found")

    return test


@app.post("/ab-tests/{test_id}/start", response_model=ABTestResponse)
async def start_ab_test(test_id: int, db: Session = Depends(get_db)):
    """Start an A/B test"""
    from abtesting.service import get_ab_test_service

    service = get_ab_test_service(db)
    try:
        test = service.start_test(test_id)
        return test
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ab-tests/{test_id}/pause", response_model=ABTestResponse)
async def pause_ab_test(test_id: int, db: Session = Depends(get_db)):
    """Pause an A/B test"""
    from abtesting.service import get_ab_test_service

    service = get_ab_test_service(db)
    try:
        test = service.pause_test(test_id)
        return test
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ab-tests/{test_id}/complete", response_model=ABTestResponse)
async def complete_ab_test(test_id: int, winner_variant_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Complete an A/B test"""
    from abtesting.service import get_ab_test_service

    service = get_ab_test_service(db)
    try:
        test = service.complete_test(test_id, winner_variant_id)
        return test
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ab-tests/{test_id}/results", response_model=ABTestResultsResponse)
async def get_ab_test_results(test_id: int, db: Session = Depends(get_db)):
    """Get A/B test results with statistical analysis"""
    from abtesting.service import get_ab_test_service

    service = get_ab_test_service(db)
    try:
        results = service.get_test_results(test_id)
        return ABTestResultsResponse(
            test_id=results.test_id,
            status=results.status,
            total_impressions=results.total_impressions,
            variants=[
                {
                    "variant_id": v.variant_id,
                    "name": v.name,
                    "impressions": v.impressions,
                    "clicks": v.clicks,
                    "conversions": v.conversions,
                    "ctr": v.ctr,
                    "conversion_rate": v.conversion_rate,
                    "confidence_interval": v.confidence_interval,
                    "is_winner": v.is_winner,
                    "lift_vs_control": v.lift_vs_control,
                }
                for v in results.variants
            ],
            is_significant=results.is_significant,
            winner_variant_id=results.winner_variant_id,
            p_value=results.p_value,
            recommended_action=results.recommended_action,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/ab-tests/{test_id}")
async def delete_ab_test(test_id: int, db: Session = Depends(get_db)):
    """Delete an A/B test"""
    from abtesting.service import get_ab_test_service

    service = get_ab_test_service(db)
    success = service.delete_test(test_id)

    if not success:
        raise HTTPException(status_code=404, detail="A/B test not found")

    return {"message": "A/B test deleted successfully"}


@app.post("/ab-tests/{test_id}/variants", response_model=ABTestVariantResponse)
async def add_ab_test_variant(test_id: int, variant_data: ABTestVariantCreate, db: Session = Depends(get_db)):
    """Add a variant to an A/B test"""
    from abtesting.service import get_ab_test_service

    service = get_ab_test_service(db)
    variant = service.add_variant(
        test_id=test_id,
        name=variant_data.name,
        ad_id=variant_data.ad_id,
        traffic_weight=variant_data.traffic_weight,
        is_control=variant_data.is_control,
    )

    return variant


@app.post("/ab-tests/{test_id}/assign")
async def assign_ab_test_variant(test_id: int, user_identifier: str, db: Session = Depends(get_db)):
    """Assign a user to an A/B test variant"""
    from abtesting.service import get_ab_test_service

    service = get_ab_test_service(db)
    variant = service.assign_variant(test_id, user_identifier)

    if not variant:
        return {"variant_id": None, "message": "User not included in test"}

    return {"variant_id": variant.id, "variant_name": variant.name, "ad_id": variant.ad_id}


# ===== ADVANCED ANALYTICS ENDPOINTS =====


@app.get("/analytics/attribution/{conversion_id}", response_model=AttributionResultResponse)
async def get_conversion_attribution(conversion_id: int, model: str = "last_touch", db: Session = Depends(get_db)):
    """Get attribution analysis for a conversion"""
    from analytics.advanced import AttributionModel, get_advanced_analytics

    analytics = get_advanced_analytics(db)

    try:
        attr_model = AttributionModel(model)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid attribution model: {model}")

    try:
        result = analytics.attribute_conversion(conversion_id, attr_model)
        return AttributionResultResponse(
            conversion_id=result.conversion_id,
            touchpoints=result.touchpoints,
            attribution_weights={str(k): v for k, v in result.attribution_weights.items()},
            total_value=result.total_value,
            attributed_values={str(k): v for k, v in result.attributed_values.items()},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/analytics/campaign-roi/{campaign_id}", response_model=CampaignROIResponse)
async def get_campaign_roi(
    campaign_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
):
    """Get ROI metrics for a campaign"""
    from analytics.advanced import get_advanced_analytics

    analytics = get_advanced_analytics(db)

    try:
        result = analytics.calculate_campaign_roi(campaign_id, start_date, end_date)
        return CampaignROIResponse(
            campaign_id=result.campaign_id,
            campaign_name=result.campaign_name,
            total_spend=result.total_spend,
            total_revenue=result.total_revenue,
            roi=result.roi,
            roas=result.roas,
            cpa=result.cpa,
            clv=result.clv,
            impressions=result.impressions,
            clicks=result.clicks,
            conversions=result.conversions,
            ctr=result.ctr,
            cvr=result.cvr,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/analytics/trends/{metric_name}", response_model=TrendAnalysisResponse)
async def get_trend_analysis(
    metric_name: str, campaign_id: Optional[int] = None, days: int = 30, db: Session = Depends(get_db)
):
    """Get trend analysis for a metric"""
    from analytics.advanced import get_advanced_analytics

    analytics = get_advanced_analytics(db)
    result = analytics.analyze_trend(metric_name, campaign_id, days)

    return TrendAnalysisResponse(
        metric_name=result.metric_name,
        current_value=result.current_value,
        previous_value=result.previous_value,
        change_percent=result.change_percent,
        trend_direction=result.trend_direction,
        data_points=result.data_points,
        forecast_next_period=result.forecast_next_period,
    )


@app.get("/analytics/top-ads")
async def get_top_performing_ads(
    campaign_id: Optional[int] = None, metric: str = "ctr", limit: int = 10, db: Session = Depends(get_db)
):
    """Get top performing ads by metric"""
    from analytics.advanced import get_advanced_analytics

    analytics = get_advanced_analytics(db)
    results = analytics.get_top_performing_ads(campaign_id, metric, limit)

    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
