# Intent Engine - Project Overview

**Version:** 0.2.0  
**Last Updated:** March 14, 2026  
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
- ✅ **URL Ranking** - Privacy-compliant URL scoring and ranking
- ✅ **Advanced Constraint Handling** - Supports range (`0-500`), comparison (`<=500`), min/max formats
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

### Component Flow

```
User Query → Intent Extraction → UniversalIntent Object
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
| **Database Queries** | <10ms | ~5ms | With connection pooling |
| **Cache Hit Rate** | >80% | 70-80% | With Redis enabled |

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
├── analytics/              # Real-time analytics
├── audit/                  # Audit trail logging
├── config/                 # Configuration modules
├── core/                   # Shared schema and utilities
│   ├── schema.py           # UniversalIntent class + enums
│   ├── embedding_service.py # Shared embedding service
│   └── utils.py            # Shared helpers
├── extraction/             # Intent extraction
│   ├── extractor.py        # Main extraction logic
│   └── constraints.py      # Constraint parsing
├── fraud/                  # Fraud detection
├── privacy/                # Privacy compliance
│   ├── consent_manager.py  # Consent management
│   └── enhanced_privacy.py # Privacy controls
├── ranking/                # Ranking module
│   ├── ranker.py           # Main ranking logic
│   └── url_ranker.py       # URL ranking
├── searxng/                # SearXNG integration
│   ├── client.py           # SearXNG client
│   └── unified_search.py   # Unified search service
├── services/               # Service recommendation
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── main.py                 # CLI entry point
├── main_api.py             # FastAPI server
├── worker.py               # ARQ worker
├── database.py             # Database models
├── models.py               # Pydantic models
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
| `POST` | `/search` | Unified privacy search with intent ranking |
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
| `GET` | `/audit-events` | Get audit events with filters |
| `GET` | `/audit-stats` | Get audit statistics |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Comprehensive health check |
| `GET` | `/status` | Service status |
| `GET` | `/metrics` | Prometheus metrics |
| `WS` | `/analytics/ws` | Real-time analytics WebSocket |

---

## Documentation

| Document | Description |
|----------|-------------|
| **[README.md](../README.md)** | Main README with quick start guide |
| **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** | This file - Quick reference and overview |
| **[COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md)** | Complete usage guide with examples |
| **[Intent-Engine-Whitepaper.md](Intent-Engine-Whitepaper.md)** | Technical whitepaper and architecture |
| **[Intent-Engine-Tech-Reference.md](Intent-Engine-Tech-Reference.md)** | Developer reference documentation |
| **[Intent-Engine-Visual-Guide.md](Intent-Engine-Visual-Guide.md)** | Visual diagrams and illustrations |
| **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** | Detailed project structure |
| **[PERFORMANCE_OPTIMIZATION_PLAN.md](PERFORMANCE_OPTIMIZATION_PLAN.md)** | Performance optimization strategies |
| **[TESTING_GUIDE.md](TESTING_GUIDE.md)** | Testing strategies and procedures |

---

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

MIT License - see the [LICENSE](../LICENSE) file for details.

---

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [sentence-transformers](https://www.sbert.net/)
- Implements privacy-first design principles
- Incorporates ethical AI practices
