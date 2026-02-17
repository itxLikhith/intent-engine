# Intent Engine - Privacy-First Intent-Driven System

> **A privacy-first, intent-driven advertising platform** that combines search, service recommendation, and ad matching capabilities without discriminatory targeting or user tracking.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## Overview

The Intent Engine is a privacy-first, intent-driven system for search, service recommendation, and ad matching. It processes user queries to extract structured intent while respecting privacy and ethical considerations. The system is organized into a modular architecture with four main phases:

1. **Intent Extraction** - Converts free-form queries into structured intent objects
2. **Constraint Satisfaction & Ranking** - Filters and ranks results based on user intent
3. **Service Recommendation** - Routes users to the most appropriate service
4. **Ad Matching** - Matches ads without discriminatory targeting

### Additional Capabilities
- **Privacy & Compliance** - GDPR-ready with consent management, data retention policies, and audit trails
- **Real-time Analytics** - Live metrics collection and WebSocket broadcasting for dashboards
- **Fraud Detection** - Comprehensive fraud detection for clicks, impressions, and conversions
- **A/B Testing** - Experiment management with statistical significance calculation
- **Campaign Management** - Full advertising campaign lifecycle with budget tracking
- **SearXNG Integration** - Privacy-focused search backend with unified search API

## Architecture

The project follows a clean, modular structure:
```
intent_engine/
├── docs/                   # Documentation files
├── core/                   # Shared schema and utilities
│   ├── schema.py           # UniversalIntent class + enums
│   └── utils.py            # Shared helpers (caching, logging)
├── extraction/             # Intent extraction module
│   ├── extractor.py        # Main IntentExtraction logic
│   └── constraints.py      # Constraint parsing logic
├── ranking/                # Ranking module
│   ├── ranker.py           # Main Ranking logic
│   ├── optimized_ranker.py # Optimized ranking implementation
│   ├── url_ranker.py       # URL ranking implementation
│   ├── optimized_url_ranker.py # Optimized URL ranking implementation
│   └── scoring.py          # Alignment/quality/ethical scoring
├── services/               # Service recommendation module
│   └── recommender.py      # Service recommendation logic
├── ads/                    # Ad matching module
│   └── matcher.py          # Ad matching logic
├── privacy/                # Privacy compliance module
│   ├── consent_manager.py  # Consent management
│   └── enhanced_privacy.py # Privacy controls and retention
├── audit/                  # Audit trail module
│   └── audit_trail.py      # Audit logging for compliance
├── analytics/              # Real-time analytics module
│   ├── realtime.py         # Real-time metrics and WebSocket
│   └── advanced.py         # Advanced analytics (ROI, attribution)
├── fraud/                  # Fraud detection module
│   └── detector.py         # Fraud detection algorithms
├── abtesting/              # A/B testing module
│   └── service.py          # A/B test management
├── searxng/                # SearXNG integration
│   ├── client.py           # SearXNG client
│   ├── settings.yml        # SearXNG configuration
│   └── unified_search.py   # Unified search with intent
├── config/                 # Configuration
│   ├── query_cache.py      # Query caching
│   └── redis_cache.py      # Redis caching
├── load_testing/           # Load testing
│   ├── locustfile.py       # Locust load tests
│   └── stress_test.py      # Stress testing
├── perf_tests/             # Performance tests
├── tests/                  # Unit tests
├── demos/                  # Demo scripts
├── scripts/                # Utility scripts
├── main.py                 # Entry point for CLI or server
├── main_api.py             # FastAPI server implementation
├── models.py               # Pydantic models for API
├── database.py             # Database models and connection
├── privacy_core.py         # Privacy validation
├── requirements.txt        # Python dependencies
├── Dockerfile              # Full Docker image
├── docker-compose.yml      # Docker Compose configuration
├── README.md               # This file
└── benchmark.py            # Benchmarking utilities
```

## Key Components

### Core Schema (`core/schema.py`)
Defines the `UniversalIntent` dataclass and associated enums that form the foundation of the system:
- `IntentGoal` - Purpose of the query (LEARN, COMPARE, TROUBLESHOOT, etc.)
- `UseCase` - Specific use cases (LEARNING, TROUBLESHOOTING, etc.)
- `ConstraintType` - Types of constraints (INCLUSION, EXCLUSION, RANGE)
- `EthicalDimension` - Privacy, sustainability, ethics considerations
- `UniversalIntent` - Main intent object with declared and inferred components

### Intent Extraction (`extraction/extractor.py`)
Converts free-form queries into structured intent objects using:
- Rule-based constraint extraction with regex patterns
- Goal classification based on keyword matching
- Skill level detection
- Semantic inference for use cases and ethical signals

### Ranking (`ranking/ranker.py`)
Implements constraint satisfaction and intent-aligned ranking:
- Filters results based on user constraints
- Computes alignment scores between results and user intent
- Applies weighted scoring for semantic similarity, ethical alignment, etc.

### Service Recommendation (`services/recommender.py`)
Routes users to the most appropriate service based on intent matching:
- Matches intent goals to service capabilities
- Considers use cases, temporal patterns, and ethical alignment

### Ad Matching (`ads/matcher.py`)
Performs ethical ad matching without tracking:
- Validates ads against fairness rules (no discriminatory targeting)
- Filters ads based on user constraints
- Scores ads for relevance and ethical alignment

## Quick Start

### Prerequisites

- **Docker** and **Docker Compose** (recommended)
- **Python 3.11+** (for local development)
- **4GB+ RAM** (for ML models)
- **2GB disk space**

### Installation with Docker (Recommended)

```bash
# Clone the repository
git clone git@github.com-work:itxLikhith/intent-engine.git
cd intent-engine

# Start all services
docker-compose up -d

# Wait for initialization (~45 seconds)
sleep 45

# Verify installation
curl http://localhost:8000/
```

### Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python -m uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload

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

### Running Demos

```bash
# Run all demos
python main.py demo

# Run specific demos
python main.py demo-search      # Intent extraction demo
python main.py demo-service     # Service recommendation demo
python main.py demo-ad          # Ad matching demo
python main.py demo-ranking     # Ranking demo
```

### Programmatic Usage
```python
from extraction.extractor import extract_intent, IntentExtractionRequest

# Create extraction request
request = IntentExtractionRequest(
    product='search',
    input={'text': 'How to set up E2E encrypted email on Android, no big tech'},
    context={'sessionId': 'session_123', 'userLocale': 'en-US'}
)

# Extract intent
response = extract_intent(request)
intent = response.intent

print(f"Goal: {intent.declared.goal}")
print(f"Constraints: {intent.declared.constraints}")
print(f"Use Cases: {[uc.value for uc in intent.inferred.useCases]}")
```

## Key Features

### Privacy & Ethics
- ✅ **No user tracking** - No persistent profiles or behavioral tracking
- ✅ **Local processing** - All intent extraction happens on-device
- ✅ **Ethical ad matching** - No discriminatory targeting based on protected attributes
- ✅ **Data minimization** - Only processes what's necessary for current session
- ✅ **Automatic cleanup** - Session data auto-deletes after 8 hours
- ✅ **Differential privacy** - Techniques applied to sensitive metrics
- ✅ **Granular consent controls** - Fine-grained user consent management

### Core Capabilities
- 🔍 **Intent Extraction** - Converts free-form queries to structured intent
- 📊 **Constraint Satisfaction** - Filters results based on user constraints
- 🎯 **Service Recommendation** - Routes users to appropriate services
- 📢 **Ad Matching** - Ethical ad matching with fairness validation
- 📈 **Real-time Analytics** - Performance metrics and reporting
- 🛡️ **Fraud Detection** - Built-in fraud prevention
- 🧪 **A/B Testing** - Experiment management
- 🔎 **SearXNG Integration** - Privacy-focused search

## Dependencies
Key dependencies defined in `requirements.txt`:
- `transformers` and `sentence-transformers` for NLP
- `torch` for neural network operations
- `numpy` for numerical computations
- `pytest` for testing
- `fastapi` and `uvicorn` for serving
- `sqlalchemy` for database operations
- `prometheus-client` for metrics

## Testing

The system includes comprehensive testing covering all modules:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_extraction.py -v
python -m pytest tests/test_ranking.py -v
python -m pytest tests/test_services.py -v
python -m pytest tests/test_ads.py -v
python -m pytest tests/test_advertising_api.py -v
python -m pytest tests/comprehensive_test.py -v

# Run load tests
cd load_testing
locust -f locustfile.py

# Run performance tests
python -m pytest perf_tests/ -v
```

## Performance
- **Warm-up time**: <100ms after initial load
- **Processing time**: <50ms per query after warm-up
- **Memory footprint**: <500MB RAM
- **CPU optimized**: Runs efficiently on standard hardware

## Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **Warm-up Time** | <100ms | After initial model load |
| **Processing Time** | <50ms | Per query after warm-up |
| **Memory Footprint** | <500MB | RAM usage |
| **Concurrent Requests** | 1000+/sec | With proper scaling |
| **Database Queries** | <10ms | With connection pooling |
| **Cache Hit Rate** | >80% | With Redis enabled |

## Privacy & Ethics Features

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

## Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Database
DATABASE_URL=sqlite:///./intent_engine.db
# For production: postgresql+asyncpg://user:pass@localhost:5432/intent_engine

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# SearXNG
SEARXNG_URL=http://localhost:8080

# Logging
LOG_LEVEL=INFO
```

## API Endpoints
- `GET /` - Health check
- `GET /status` - Service status
- `GET /metrics` - Prometheus metrics
- `POST /extract-intent` - Extract structured intent from user query
- `POST /search` - Unified privacy search with intent extraction and ranking
- `POST /rank-results` - Rank results based on user intent
- `POST /rank-urls` - Rank URLs for privacy-focused search
- `POST /recommend-services` - Recommend services based on user intent
- `POST /match-ads` - Match ads to user intent with fairness validation
- `POST /match-ads-advanced` - Advanced matching with campaign context

### Campaign Management
- `POST /campaigns` - Create new campaign
- `GET /campaigns/{id}` - Get campaign details
- `PUT /campaigns/{id}` - Update campaign
- `DELETE /campaigns/{id}` - Delete campaign
- `GET /campaigns` - List campaigns with filters
- `POST /adgroups` - Create ad group
- `GET /adgroups/{id}` - Get ad group details
- `PUT /adgroups/{id}` - Update ad group
- `GET /adgroups` - List ad groups
- `POST /ads` - Create ad
- `GET /ads/{id}` - Get ad details
- `PUT /ads/{id}` - Update ad
- `DELETE /ads/{id}` - Delete ad
- `GET /ads` - List ads with filters
- `POST /advertisers` - Create advertiser
- `GET /advertisers/{id}` - Get advertiser details
- `GET /advertisers` - List advertisers

### Creative & Tracking
- `POST /creatives` - Upload creative assets
- `GET /creatives/{id}` - Get creative details
- `PUT /creatives/{id}` - Update creative
- `DELETE /creatives/{id}` - Delete creative
- `POST /click-tracking` - Record ad clicks
- `POST /conversion-tracking` - Record conversions
- `POST /fraud-detection` - Report potential fraud events

### Analytics & Reporting
- `GET /reports/campaign-performance` - Campaign performance reports
- `GET /analytics/attribution/{conversion_id}` - Get conversion attribution
- `GET /analytics/campaign-roi/{campaign_id}` - Get campaign ROI metrics
- `GET /analytics/trends/{metric_name}` - Get trend analysis
- `GET /analytics/top-ads` - Get top performing ads

### Privacy & Compliance
- `POST /consent/record` - Record user consent
- `GET /consent/{user_id}/{consent_type}` - Get user consent
- `POST /consent/withdraw/{user_id}/{consent_type}` - Withdraw consent
- `GET /consent-summary` - Get system-wide consent summary
- `POST /privacy-controls/apply-retention-policy` - Apply data retention policies
- `GET /privacy-controls/compliance-report` - Get privacy compliance report
- `GET /audit-events` - Get audit events with filters
- `GET /audit-stats` - Get audit statistics

### A/B Testing
- `POST /abtests` - Create A/B test
- `GET /abtests/{id}` - Get A/B test details
- `PUT /abtests/{id}` - Update A/B test
- `DELETE /abtests/{id}` - Delete A/B test
- `GET /abtests` - List A/B tests
- `GET /abtests/{id}/results` - Get A/B test results
- `GET /abtests/{id}/variants` - Get A/B test variants

## Testing
The system includes comprehensive testing covering all modules:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_extraction.py -v
python -m pytest tests/test_ranking.py -v
python -m pytest tests/test_services.py -v
python -m pytest tests/test_ads.py -v
python -m pytest tests/test_advertising_api.py -v
python -m pytest tests/comprehensive_test.py -v
python -m pytest tests/test_url_ranking.py -v

# Run load tests
cd load_testing
locust -f locustfile.py

# Run stress tests
python stress_test_all.py
```

### Test Coverage
- ✅ Unit tests for core modules (extraction, ranking, services, ads)
- ✅ Integration tests for API endpoints
- ✅ Performance tests for key operations
- ✅ Privacy compliance validation
- ✅ Fraud detection mechanism tests
- ✅ Load testing with Locust
- ✅ Stress testing for concurrent requests
- ✅ URL ranking tests

## Deployment

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

### Production Checklist

- [ ] Use PostgreSQL instead of SQLite
- [ ] Enable Redis caching
- [ ] Configure CORS origins
- [ ] Set up SSL/TLS
- [ ] Configure logging
- [ ] Set up monitoring
- [ ] Enable automated backups
- [ ] Review security settings

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

Comprehensive documentation is available in the `docs` directory:

| Document | Description |
|----------|-------------|
| **[README.md](README.md)** | Main README with quick start guide |
| **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** | Quick reference and project overview |
| **[COMPREHENSIVE_GUIDE.md](docs/COMPREHENSIVE_GUIDE.md)** | Complete usage guide with examples |
| **[Intent-Engine-Whitepaper.md](docs/Intent-Engine-Whitepaper.md)** | Technical whitepaper and architecture |
| **[Intent-Engine-Tech-Reference.md](docs/Intent-Engine-Tech-Reference.md)** | Developer reference documentation |
| **[Intent-Engine-Visual-Guide.md](docs/Intent-Engine-Visual-Guide.md)** | Visual diagrams and illustrations |
| **[PARENT_DIRECTORY_GUIDE.md](docs/PARENT_DIRECTORY_GUIDE.md)** | Parent directory information |

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

- **GitHub Issues**: [Open an issue](https://github.com/itxLikhith/intent-engine/issues) for bug reports and feature requests
- **Documentation**: Review the documentation files listed above
- **Email**: Contact the maintainers for direct support

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for high-performance API development
- Uses [sentence-transformers](https://www.sbert.net/) for semantic similarity
- Implements privacy-first design principles throughout
- Incorporates ethical AI practices