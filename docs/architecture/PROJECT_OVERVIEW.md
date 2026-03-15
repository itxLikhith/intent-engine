# Intent Engine - Project Overview

**Version:** 0.3.0
**Last Updated:** March 15, 2026
**Repository:** [intent-engine](https://github.com/itxLikhith/intent-engine)
**Docker Image:** `anony45/intent-engine-api:latest`

---

## Executive Summary

The Intent Engine is a **privacy-first, intent-driven system** for search, service recommendation, and ad matching. It processes user queries to extract structured intent while respecting privacy and ethical considerations, without discriminatory targeting or user tracking.

### Core Principles

1. **Intent-First**: All decisions derive from structured intent, not user identity
2. **Privacy Native**: No persistent tracking; intent signals decay on session boundary (8-hour TTL)
3. **Open Architecture**: Intent schema is composable and extensible
4. **Non-Discriminatory**: Matching algorithms never use sensitive attributes
5. **Transparent**: Intent extraction rules are inspectable and rule-based

### Key Features

- ✅ **Intent Extraction** - Converts free-form queries to structured intent (NLP + rule-based)
- ✅ **Privacy-Focused Search** - SearXNG integration with intent-aware ranking
- ✅ **Federated Search** - Query router with Go Crawler + SearXNG backends
- ✅ **Result Aggregation** - Deduplication and score normalization across backends
- ✅ **URL Ranking** - Privacy-compliant URL scoring and ranking
- ✅ **Advanced Constraint Handling** - Supports range (`0-500`), comparison (`<=500`), min/max formats
- ✅ **Web Intent Extraction** - Automatic intent tagging for crawled web content
- ✅ **Vector Search** - Qdrant integration for semantic search (optional)
- ✅ **Event Streaming** - Kafka/Redpanda integration for real-time analytics (optional)
- ✅ **Distributed Tracing** - OpenTelemetry + Jaeger for observability (optional)
- ✅ **Service Recommendation** - Routes users to appropriate services based on intent
- ✅ **Ethical Ad Matching** - Fair ad matching with fairness validation
- ✅ **Campaign Management** - Full advertising campaign lifecycle with budget tracking
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

# Wait for initialization (~45 seconds)
sleep 45

# Verify installation
curl http://localhost:8000/
```

### First API Call

```bash
# Extract intent from a query
curl -X POST http://localhost:8000/extract-intent \
  -H "Content-Type: application/json" \
  -d '{
    "product": "search",
    "input": {"text": "best laptop for programming under 50000 rupees"},
    "context": {"sessionId": "test-123", "userLocale": "en-US"}
  }' | jq
```

### Run Demos

```bash
# Run all demos
python main.py demo

# Run specific demo
python main.py demo-search
```

---

## System Architecture

### Four Main Phases

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  1. Intent      │────▶│  2. Constraint       │────▶│  3. Service      │────▶│  4. Ad       │
│  Extraction     │     │  Satisfaction &      │     │  Recommendation  │     │  Matching    │
│                 │     │  Ranking             │     │                  │     │              │
└─────────────────┘     └──────────────────────┘     └──────────────────┘     └──────────────┘
```

### Component Flow (Enhanced with Query Router)

```
User Query → Intent Extraction → UniversalIntent Object
                                      │
                                      ▼
                              Query Router (Intent-Based)
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
            Go Crawler (60%)    SearXNG (40%)    Custom Index
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                              Result Aggregator
                              (Deduplication)
                                      │
                                      ▼
                              Intent-Based Ranking
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
            Search Results    Service Options    Ad Inventory
                    │                 │                 │
                    ▼                 ▼                 ▼
            Rank Results     Recommend        Match Ads
            (Constraint      Service          (Fairness
             Satisfaction)   (Intent Match)    Validation)
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                              Privacy-Preserving
                              Response to User
```

---

## Technology Stack

### Backend
- **FastAPI** - Modern async web framework
- **SQLAlchemy 2.0** - Async ORM for database operations
- **Pydantic v2** - Data validation and settings management
- **PostgreSQL** - Primary database with PgBouncer connection pooling
- **Redis/Valkey** - Caching and session management

### Machine Learning
- **PyTorch** - Deep learning framework
- **Sentence Transformers** - Semantic similarity calculations
- **Transformers** - NLP model inference
- **NumPy** - Numerical computations

### Infrastructure
- **Docker & Docker Compose** - Containerization and orchestration
- **SearXNG** - Privacy-focused metasearch engine
- **Go Crawler** - High-performance web crawler and indexer
- **Qdrant** - Vector database for semantic search (optional)
- **Redpanda/Kafka** - Message broker for event streaming (optional)
- **Jaeger** - Distributed tracing (optional)
- **Prometheus** - Metrics collection
- **Grafana** - Dashboards and visualization
- **ARQ** - Background task queue

### Development Tools
- **Ruff** - Fast Python linter and formatter
- **Pytest** - Testing framework
- **Pre-commit** - Git hooks management
- **Commitizen** - Conventional commits

---

## Performance Metrics

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| **Warm-up Time** | <100ms | ~50ms | After initial model load |
| **Processing Time** | <50ms | ~30ms | Per query after warm-up |
| **Memory Footprint** | <500MB | ~450MB | RAM usage |
| **Concurrent Requests** | 1000+/sec | 200-300/sec | With Redis caching |
| **Query Router Overhead** | <20ms | ~15ms | P50 latency impact |
| **Result Deduplication** | 15-25% | 18% | Duplicate removal rate |
| **Cache Hit Rate** | >80% | 70-80% | With Redis enabled |

### Load Test Results (March 2026)

| Concurrent Users | Throughput | Success Rate | Mean Latency | P95 Latency |
|------------------|------------|--------------|--------------|-------------|
| **1** | 243 req/s | 100% | 16ms | 20ms |
| **5** | 328 req/s | 100% | 53ms | 71ms |
| **10** | 369 req/s | 100% | 84ms | 123ms |
| **20** | 372 req/s | 100% | 183ms | 243ms |
| **50** | 646 req/s | 72% | 205ms | 336ms |

**Optimal Operating Point:** 10-20 concurrent users with 100% success rate

### Load Capacity

| Concurrent Users | Throughput | Success Rate | Latency (Mean) |
|------------------|------------|--------------|----------------|
| 1-10 | 200-370 req/s | 100% | 15-85ms |
| 20 | 370 req/s | 100% | 180ms |
| 50 | 650 req/s | 72% | 200ms |

---

## Privacy & Ethics Features

### Data Protection
- **No User Tracking**: No persistent profiles or behavioral tracking
- **Local Processing**: All intent extraction happens on-device
- **Data Minimization**: Only processes what's necessary for current session
- **Automatic Cleanup**: Session data auto-deletes after 8 hours
- **Differential Privacy**: Techniques applied to sensitive metrics

### Compliance Features
- **GDPR Ready**: Privacy-by-design architecture
- **Granular Consent**: Fine-grained user consent management
- **Right to Deletion**: Automated data deletion
- **Data Portability**: Export capabilities
- **Audit Trails**: Comprehensive logging for compliance

### Fair Ad Matching
The system enforces strict fairness rules:
- ❌ **Forbidden Dimensions**: Age, gender, income, race, ethnicity
- ❌ **Protected Attributes**: Political affiliation, religious belief
- ❌ **Sensitive Data**: Health condition, sexual orientation
- ❌ **Discriminatory Targeting**: Location, behavior, interests, purchasing history

---

## Project Structure

```
intent-engine/
├── abtesting/              # A/B testing module
├── ads/                    # Ad matching module
├── analytics/              # Real-time analytics + Kafka events
├── audit/                  # Audit trail logging
├── bin/                    # Executable scripts
├── config/                 # Configuration modules + tracing
├── core/                   # Shared schema and utilities
│   ├── schema.py           # UniversalIntent class + enums
│   ├── embedding_service.py # Shared embedding service
│   ├── exceptions.py       # Custom exceptions
│   ├── utils.py            # Shared helpers
│   └── vector_store.py     # Qdrant vector DB integration
├── extraction/             # Intent extraction
│   ├── extractor.py        # Main extraction logic
│   ├── constraints.py      # Constraint parsing
│   └── web_extractor.py    # Web content intent extraction
├── fraud/                  # Fraud detection
├── go-crawler/             # Go-based web crawler
├── privacy/                # Privacy compliance
│   ├── consent_manager.py  # Consent management
│   └── enhanced_privacy.py # Privacy controls
├── ranking/                # Ranking module
│   ├── ranker.py           # Main ranking logic
│   ├── url_ranker.py       # URL ranking
│   ├── optimized_ranker.py # Optimized ranking
│   └── optimized_url_ranker.py # Optimized URL ranking
├── searxng/                # SearXNG integration
│   ├── client.py           # SearXNG client
│   ├── query_router.py     # Intent-based query routing
│   ├── result_aggregator.py # Result deduplication
│   ├── topic_expander.py   # Topic discovery
│   ├── scheduled_seed_discovery.py # Seed URL discovery
│   └── unified_search.py   # Unified search service
├── services/               # Service recommendation
├── scripts/                # Utility scripts
├── tests/                  # Unit + integration tests
├── perf_tests/             # Performance tests
├── load_testing/           # Locust load tests
├── migrations/             # Database migrations
├── docs/                   # Documentation
├── demos/                  # Demo scripts
├── grafana/                # Grafana dashboards
├── main.py                 # CLI entry point
├── main_api.py             # FastAPI server
├── worker.py               # ARQ worker
├── database.py             # Database models
├── models.py               # Pydantic models
├── go_search_client.py     # Go crawler API client
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Container image
├── pyproject.toml          # Python project config
└── requirements.txt        # Python dependencies
```

---

## API Endpoints

### Core Intent Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/extract-intent` | Extract structured intent from query |
| `POST` | `/search` | Unified privacy search with intent ranking (Query Router enabled) |
| `POST` | `/rank-results` | Rank results based on user intent |
| `POST` | `/rank-urls` | Privacy-focused URL ranking |
| `POST` | `/recommend-services` | Recommend services based on intent |
| `POST` | `/match-ads` | Match ads with fairness validation |
| `POST` | `/match-ads-advanced` | Advanced matching with campaign context |

### Campaign Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/campaigns` | Create campaign |
| `GET` | `/campaigns/{id}` | Get campaign details |
| `PUT` | `/campaigns/{id}` | Update campaign |
| `DELETE` | `/campaigns/{id}` | Delete campaign |
| `GET` | `/campaigns` | List campaigns with filters |
| `POST` | `/adgroups` | Create ad group |
| `GET` | `/adgroups/{id}` | Get ad group details |
| `PUT` | `/adgroups/{id}` | Update ad group |
| `POST` | `/ads` | Create ad |
| `GET` | `/ads/{id}` | Get ad details |
| `PUT` | `/ads/{id}` | Update ad |
| `DELETE` | `/ads/{id}` | Delete ad |

### Creative & Tracking

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/creatives` | Upload creative assets |
| `POST` | `/click-tracking` | Record ad clicks |
| `POST` | `/conversion-tracking` | Record conversions |
| `POST` | `/fraud-detection` | Report potential fraud events |

### Analytics & Reporting

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/reports/campaign-performance` | Campaign performance reports |
| `GET` | `/analytics/attribution/{id}` | Get conversion attribution |
| `GET` | `/analytics/campaign-roi/{id}` | Get campaign ROI metrics |
| `GET` | `/analytics/trends/{metric}` | Get trend analysis |
| `GET` | `/analytics/top-ads` | Get top performing ads |

### Privacy & Compliance

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/consent/record` | Record user consent |
| `GET` | `/consent/{user_id}/{type}` | Get user consent |
| `POST` | `/consent/withdraw/{user_id}/{type}` | Withdraw consent |
| `GET` | `/consent-summary` | Get system-wide consent summary |
| `GET` | `/audit-events` | Get audit events with filters |
| `GET` | `/audit-stats` | Get audit statistics |
| `POST` | `/privacy-controls/apply-retention-policy` | Apply data retention policies |
| `GET` | `/privacy-controls/compliance-report` | Get privacy compliance report |

### A/B Testing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/abtests` | Create A/B test |
| `GET` | `/abtests/{id}` | Get A/B test details |
| `PUT` | `/abtests/{id}` | Update A/B test |
| `DELETE` | `/abtests/{id}` | Delete A/B test |
| `GET` | `/abtests` | List A/B tests |
| `GET` | `/abtests/{id}/results` | Get A/B test results |
| `GET` | `/abtests/{id}/variants` | Get A/B test variants |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Comprehensive health check |
| `GET` | `/status` | Service status |
| `GET` | `/metrics` | Prometheus metrics |
| `WS` | `/analytics/ws` | Real-time analytics WebSocket |
| `GET` | `/docs` | Interactive API documentation (Swagger) |
| `GET` | `/redoc` | API documentation (ReDoc) |

---

## Documentation

### Getting Started
| Document | Description |
|----------|-------------|
| **[README.md](../README.md)** | Main README with quick start guide |
| **[INDEX.md](../INDEX.md)** | Complete documentation index |
| **[Quick Start](../getting-started/QUICKSTART.md)** | Get started in 5 minutes |
| **[Production Setup](../getting-started/README_PRODUCTION.md)** | Production deployment guide |

### Architecture & Design
| Document | Description |
|----------|-------------|
| **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** | This file - Quick reference and overview |
| **[ARCHITECTURE_BLUEPRINT.md](../ARCHITECTURE_BLUEPRINT.md)** | Unified architecture blueprint |
| **[IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md)** | Step-by-step implementation guide |
| **[PHASE1_README.md](../PHASE1_README.md)** | Phase 1 implementation summary |
| **[Intent-Engine-Whitepaper.md](Intent-Engine-Whitepaper.md)** | Technical whitepaper |
| **[Intent-Engine-Tech-Reference.md](Intent-Engine-Tech-Reference.md)** | Developer reference |
| **[Intent-Engine-Visual-Guide.md](Intent-Engine-Visual-Guide.md)** | Visual diagrams |

### Go Crawler & Indexer
| Document | Description |
|----------|-------------|
| **[Go Crawler README](../go-crawler/README.md)** | Go crawler overview |
| **[Setup Guide](../go-crawler/GO_CRAWLER_SETUP_GUIDE.md)** | Complete setup guide |
| **[Quick Start](../go-crawler/QUICKSTART.md)** | Go crawler quick start |

### Deployment & Operations
| Document | Description |
|----------|-------------|
| **[Deployment Checklist](../deployment/DEPLOYMENT_CHECKLIST.md)** | Production checklist |
| **[Performance Optimization](../deployment/PERFORMANCE_OPTIMIZATION_PLAN.md)** | Optimization guide |
| **[CI/CD Improvements](../deployment/CI_IMPROVEMENTS.md)** | Continuous integration |

### Testing & Performance
| Document | Description |
|----------|-------------|
| **[Testing Guide](../testing/TESTING_GUIDE.md)** | Testing strategies |
| **[Testing Plan](../testing/TESTING_PLAN.md)** | Test planning |
| **[Stress Test Analysis](../testing/STRESS_TEST_ANALYSIS.md)** | Performance analysis |

---

## Key Architectural Components

### Query Router (Phase 1)

The Query Router intelligently routes queries to optimal search backends based on extracted intent:

```python
# Intent-based routing examples:
# Troubleshooting → SearXNG (community discussions)
# Comparison → Go Crawler (60%) + SearXNG (40%)
# Privacy-focused → Go Crawler (curated index)
# Breaking news → SearXNG (real-time news engines)
# Learning → Hybrid (50/50)
```

**Benefits:**
- Better result quality through backend specialization
- Parallel execution for comprehensive coverage
- Automatic fallback on backend failures
- 18% duplicate removal through aggregation

### Result Aggregator

Merges and deduplicates results from multiple backends:

- **URL Normalization**: Removes tracking parameters (UTM, gclid, etc.)
- **Deduplication**: Groups identical URLs from different backends
- **Score Normalization**: Normalizes scores across backend scoring systems
- **Source Attribution**: Tracks which backends returned each result

### Web Intent Extractor

Automatically extracts intent metadata from crawled web pages:

- Primary goal detection (learn, compare, troubleshoot, etc.)
- Skill level identification (beginner, intermediate, advanced)
- Topic extraction using TF-based keyword analysis
- Confidence scoring for intent classification

### Vector Store (Optional)

Qdrant integration for semantic search capabilities:

- Stores intent embeddings for crawled URLs
- Enables similarity search by intent
- Supports hybrid search (keyword + semantic)

### Event Streaming (Optional)

Kafka/Redpanda integration for real-time analytics:

- Intent extraction events
- Search execution events
- Click tracking events
- Conversion events

### Distributed Tracing (Optional)

OpenTelemetry + Jaeger for observability:

- End-to-end request tracing
- Performance bottleneck identification
- Error debugging across services

## Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Database
DATABASE_URL=postgresql://intent_user:password@localhost:5432/intent_engine

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here

# SearXNG
SEARXNG_BASE_URL=http://localhost:8080

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
pytest --cov=. --cov-report=html tests/

# Run load tests
cd load_testing
locust -f locustfile.py

# Run performance tests
python -m pytest perf_tests/ -v
```

---

## Deployment

### Production Checklist

**Already Configured:**
- [x] PostgreSQL with connection pooling (PgBouncer)
- [x] Redis caching enabled
- [x] CORS configuration
- [x] Prometheus + Grafana monitoring
- [x] Rate limiting
- [x] Health checks
- [x] SearXNG integration

**Required for Production:**
- [ ] Change default PostgreSQL password
- [ ] Set secure SECRET_KEY
- [ ] Update CORS_ORIGINS to actual domains
- [ ] Enable SSL/TLS
- [ ] Configure Redis authentication
- [ ] Set up automated backups
- [ ] Configure log aggregation
- [ ] Set up alerting rules
- [ ] Review rate limits
- [ ] Enable firewall rules

---

## Support

- **GitHub Issues**: [Open an issue](https://github.com/itxLikhith/intent-engine/issues)
- **Documentation**: Review the [docs](../docs/) directory
- **Email**: Contact the maintainers

---

## License

This project is licensed under the **Intent Engine Community License (IECL) v1.0** - see the [LICENSE](../LICENSE) file for details.

**Key Points:**
- ✅ Free for Non-Commercial Purposes (personal, educational, academic, internal evaluation)
- ❌ Commercial use requires separate Commercial License
- 📧 Contact: anony45.omnipresent@proton.me for Commercial Licensing

---

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [sentence-transformers](https://www.sbert.net/)
- Implements privacy-first design principles
- Incorporates ethical AI practices
