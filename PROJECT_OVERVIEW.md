# Intent Engine Advertising System - Project Overview

**Version:** 2.0.0
**Last Updated:** February 17, 2026
**Repository:** intent-ads

---

## Executive Summary

The Intent Engine Advertising System is a **privacy-first, intent-driven advertising platform** that provides ethical ad matching without discriminatory targeting or user tracking. The system processes user queries to extract structured intent and uses this intent for search ranking, service recommendation, and ad matching.

### Core Principles

1. **Intent-First**: All decisions derive from structured intent, not user identity
2. **Privacy Native**: No persistent tracking; intent signals decay on session boundary
3. **Open Architecture**: Intent schema is composable and extensible
4. **Non-Discriminatory**: Matching algorithms never use sensitive attributes
5. **Transparent**: Intent extraction rules are inspectable

### Key Features

- ✅ **Intent Extraction** - Converts free-form queries to structured intent
- ✅ **Privacy-Focused Search** - SearXNG integration with intent-aware ranking
- ✅ **URL Ranking** - Privacy-compliant URL scoring and ranking
- ✅ **Service Recommendation** - Routes users to appropriate services
- ✅ **Ethical Ad Matching** - Fair ad matching with fairness validation
- ✅ **Campaign Management** - Full advertising campaign lifecycle
- ✅ **Real-time Analytics** - Live metrics with WebSocket broadcasting
- ✅ **Fraud Detection** - Comprehensive fraud detection for clicks, impressions, conversions
- ✅ **A/B Testing** - Experiment management with statistical significance
- ✅ **Privacy Compliance** - GDPR-ready with consent management and audit trails

---

## Quick Reference

### Start the System

```bash
# Using Docker (Recommended)
docker-compose up -d

# Local Development
pip install -r requirements.txt
python -m uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
```

### First API Call

```bash
curl -X POST http://localhost:8000/extract-intent \
  -H "Content-Type: application/json" \
  -d '{
    "product": "search",
    "input": {"text": "best laptop for programming under 50000 rupees"},
    "context": {"sessionId": "test-123"}
  }'
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## System Architecture

### Four-Phase Processing Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌──────────┐
│  1. INTENT      │     │  2. CONSTRAINT   │     │  3. SERVICE  │     │  4. AD   │
│  EXTRACTION     │────▶│  & RANKING       │────▶│  REC         │────▶│  MATCH   │
└─────────────────┘     └──────────────────┘     └──────────────┘     └──────────┘
```

### Component Overview

| Module | Directory | Purpose |
|--------|-----------|---------|
| **Core Schema** | `core/` | Universal intent data models and enums |
| **Intent Extraction** | `extraction/` | Parse queries into structured intent |
| **Ranking** | `ranking/` | Constraint satisfaction and result ranking |
| **Services** | `services/` | Service recommendation and routing |
| **Ads** | `ads/` | Ethical ad matching with fairness validation |
| **Privacy** | `privacy/` | Consent management and privacy controls |
| **Audit** | `audit/` | Audit trail and compliance logging |
| **Analytics** | `analytics/` | Real-time metrics and advanced analytics |
| **Fraud** | `fraud/` | Fraud detection and prevention |
| **A/B Testing** | `abtesting/` | Experiment management and statistical analysis |
| **SearXNG** | `searxng/` | Privacy-focused search integration |
| **Config** | `config/` | Query caching and Redis caching |

---

## API Endpoints Reference

### Core Intent Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/status` | Service status with metrics |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/extract-intent` | Extract structured intent from query |
| `POST` | `/search` | Unified privacy search with intent |
| `POST` | `/rank-results` | Rank results based on intent |
| `POST` | `/rank-urls` | Privacy-focused URL ranking |
| `POST` | `/recommend-services` | Recommend services based on intent |
| `POST` | `/match-ads` | Ethical ad matching |
| `POST` | `/match-ads-advanced` | Advanced matching with campaign context |

### Campaign Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/campaigns` | Create new campaign |
| `GET` | `/campaigns` | List campaigns (paginated) |
| `GET` | `/campaigns/{id}` | Get campaign details |
| `PUT` | `/campaigns/{id}` | Update campaign |
| `DELETE` | `/campaigns/{id}` | Delete campaign |
| `POST` | `/adgroups` | Create ad group |
| `GET` | `/adgroups` | List ad groups |
| `GET` | `/adgroups/{id}` | Get ad group details |
| `PUT` | `/adgroups/{id}` | Update ad group |
| `POST` | `/ads` | Create ad |
| `GET` | `/ads` | List ads |
| `GET` | `/ads/{id}` | Get ad details |
| `PUT` | `/ads/{id}` | Update ad |
| `DELETE` | `/ads/{id}` | Delete ad |
| `POST` | `/advertisers` | Create advertiser |
| `GET` | `/advertisers` | List advertisers |
| `GET` | `/advertisers/{id}` | Get advertiser details |

### Tracking & Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/click-tracking` | Record ad click |
| `POST` | `/conversion-tracking` | Record conversion |
| `POST` | `/fraud-detection` | Report fraud event |
| `GET` | `/analytics/attribution/{id}` | Get attribution data |
| `GET` | `/analytics/campaign-roi/{id}` | Get campaign ROI |
| `GET` | `/analytics/trends/{metric}` | Get trend analysis |
| `GET` | `/analytics/top-ads` | Get top performing ads |
| `GET` | `/reports/campaign-performance` | Campaign performance reports |

### Privacy & Compliance

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/consent/record` | Record user consent |
| `GET` | `/consent/{user_id}/{type}` | Get consent status |
| `POST` | `/consent/withdraw/{user_id}/{type}` | Withdraw consent |
| `GET` | `/consent-summary` | System-wide consent summary |
| `GET` | `/audit-events` | Get audit events |
| `GET` | `/audit-stats` | Get audit statistics |
| `GET` | `/privacy-controls/compliance-report` | Compliance report |
| `POST` | `/privacy-controls/apply-retention-policy` | Apply retention policy |

### A/B Testing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/abtests` | Create A/B test |
| `GET` | `/abtests` | List A/B tests |
| `GET` | `/abtests/{id}` | Get A/B test details |
| `PUT` | `/abtests/{id}` | Update A/B test |
| `DELETE` | `/abtests/{id}` | Delete A/B test |
| `GET` | `/abtests/{id}/results` | Get A/B test results |
| `GET` | `/abtests/{id}/variants` | Get A/B test variants |

### Creative Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/creatives` | Upload creative assets |
| `GET` | `/creatives/{id}` | Get creative details |
| `PUT` | `/creatives/{id}` | Update creative |
| `DELETE` | `/creatives/{id}` | Delete creative |

---

## Technology Stack

### Backend

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | FastAPI 0.104+ | High-performance API |
| **Database ORM** | SQLAlchemy 2.0+ | Database abstraction |
| **Database** | SQLite (dev) / PostgreSQL (prod) | Data persistence |
| **ML/NLP** | sentence-transformers, transformers | Intent extraction |
| **Deep Learning** | PyTorch 2.1+ | Neural network operations |
| **Validation** | Pydantic 2.5+ | Data validation |
| **Caching** | Redis (optional) | Performance optimization |
| **Monitoring** | Prometheus | Metrics collection |
| **Task Scheduling** | APScheduler | Background tasks |

### Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Containerization** | Docker | Application packaging |
| **Orchestration** | Docker Compose | Multi-container management |
| **Search Backend** | SearXNG | Privacy-focused search |
| **Admin UI** | pgAdmin (optional) | Database management |

---

## Performance Characteristics

| Metric | Target | Notes |
|--------|--------|-------|
| **Warm-up Time** | <100ms | After initial model load |
| **Processing Time** | <50ms | Per query after warm-up |
| **Memory Footprint** | <500MB | RAM usage |
| **Concurrent Requests** | 1000+/sec | With proper scaling |
| **Database Queries** | <10ms | With connection pooling |
| **Cache Hit Rate** | >80% | With Redis enabled |

---

## Development Workflow

### Project Structure

```
intent-engine/
├── core/                    # Shared schema and utilities
│   ├── schema.py            # UniversalIntent class + enums
│   └── utils.py             # Shared helpers
├── extraction/              # Intent extraction
│   ├── extractor.py         # Main extraction logic
│   └── constraints.py       # Constraint parsing
├── ranking/                 # Result ranking
│   ├── ranker.py            # Main ranking logic
│   ├── optimized_ranker.py  # Optimized implementation
│   ├── url_ranker.py        # URL ranking
│   └── scoring.py           # Scoring functions
├── services/                # Service recommendation
│   └── recommender.py       # Recommendation logic
├── ads/                     # Ad matching
│   └── matcher.py           # Ad matching logic
├── privacy/                 # Privacy compliance
│   ├── consent_manager.py   # Consent management
│   └── enhanced_privacy.py  # Privacy controls
├── audit/                   # Audit trail
│   └── audit_trail.py       # Audit logging
├── analytics/               # Analytics
│   ├── realtime.py          # Real-time metrics
│   └── advanced.py          # Advanced analytics
├── fraud/                   # Fraud detection
│   └── detector.py          # Fraud detection logic
├── abtesting/               # A/B testing
│   └── service.py           # A/B test management
├── searxng/                 # SearXNG integration
│   ├── client.py            # SearXNG client
│   └── unified_search.py    # Unified search
├── config/                  # Configuration
│   ├── query_cache.py       # Query caching
│   └── redis_cache.py       # Redis caching
├── load_testing/            # Load testing
│   ├── locustfile.py        # Locust tests
│   └── stress_test.py       # Stress tests
├── perf_tests/              # Performance tests
├── tests/                   # Unit tests
├── demos/                   # Demo scripts
├── scripts/                 # Utility scripts
├── main_api.py              # FastAPI application
├── database.py              # Database models
├── models.py                # Pydantic models
├── privacy_core.py          # Privacy validation
├── requirements.txt         # Dependencies
├── Dockerfile               # Docker config
├── docker-compose.yml       # Docker Compose
└── README.md                # Documentation
```

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific modules
python -m pytest tests/test_extraction.py -v
python -m pytest tests/test_ranking.py -v
python -m pytest tests/test_ads.py -v
python -m pytest tests/test_advertising_api.py -v
python -m pytest tests/test_url_ranking.py -v
python -m pytest tests/comprehensive_test.py -v

# Load testing
cd load_testing
locust -f locustfile.py

# Stress testing
python stress_test_all.py
```

### Code Style

```bash
# Format code
black .

# Type checking
mypy .

# Linting
flake8 .
```

---

## Deployment Guide

### Docker Deployment

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale intent-engine=3

# Stop
docker-compose down
```

### Environment Configuration

```bash
# Copy example environment
cp .env.example .env

# Edit .env with your settings
```

### Production Checklist

- [ ] Use PostgreSQL instead of SQLite
- [ ] Enable Redis caching
- [ ] Configure CORS origins
- [ ] Set up SSL/TLS
- [ ] Configure logging
- [ ] Set up monitoring
- [ ] Enable automated backups
- [ ] Review security settings

---

## Privacy & Compliance Features

### Data Protection

- **No User Tracking**: No persistent profiles or behavioral tracking
- **Local Processing**: All intent extraction on-device
- **Data Minimization**: Only necessary data processed
- **Automatic Cleanup**: Session data deleted after 8 hours
- **Differential Privacy**: Applied to sensitive metrics

### Compliance

- **GDPR Ready**: Privacy-by-design architecture
- **Consent Management**: Granular consent controls
- **Right to Deletion**: Automated data deletion
- **Data Portability**: Export capabilities
- **Audit Trails**: Comprehensive logging

---

## Troubleshooting

### Common Issues

**Issue**: Models not loading  
**Solution**: Ensure 4GB+ RAM available; check model cache directory

**Issue**: Database connection errors  
**Solution**: Verify DATABASE_URL; check PostgreSQL is running

**Issue**: High latency  
**Solution**: Enable Redis caching; check model warm-up

**Issue**: Docker build fails  
**Solution**: Increase Docker memory limit; check network connectivity

### Getting Help

1. Check documentation in `COMPREHENSIVE_GUIDE.md`
2. Review whitepaper `Intent-Engine-Whitepaper.md`
3. Check technical reference `Intent-Engine-Tech-Reference.md`
4. Open GitHub issue

---

## License

MIT License - See LICENSE file for details.

## Support

- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: See docs in this directory
- **Email**: Contact maintainers for support
