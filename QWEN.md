# Intent Engine - Project Context

## Project Overview

The **Intent Engine** is a privacy-first, intent-driven system for search, service recommendation, and ad matching. It processes user queries to extract structured intent while respecting privacy and ethical considerations, without discriminatory targeting or user tracking.

**Version:** 0.3.0  
**Repository:** https://github.com/itxLikhith/intent-engine  
**Docker Image:** `anony45/intent-engine-api:latest`

### Core Principles

1. **Intent-First**: All decisions derive from structured intent, not user identity
2. **Privacy Native**: No persistent tracking; intent signals decay on session boundary (8-hour TTL)
3. **Open Architecture**: Intent schema is composable and extensible
4. **Non-Discriminatory**: Matching algorithms never use sensitive attributes
5. **Transparent**: Intent extraction rules are inspectable and rule-based

### Key Features

- **Intent Extraction** - Converts free-form queries to structured intent (NLP + rule-based)
- **Privacy-Focused Search** - SearXNG integration with intent-aware ranking
- **Federated Search** - Query router with Go Crawler + SearXNG backends
- **Result Aggregation** - Deduplication and score normalization across backends
- **URL Ranking** - Privacy-compliant URL scoring and ranking
- **Advanced Constraint Handling** - Supports range (`0-500`), comparison (`<=500`), min/max formats
- **Web Intent Extraction** - Automatic intent tagging for crawled web content
- **Vector Search** - Qdrant integration for semantic search (optional)
- **Event Streaming** - Kafka/Redpanda integration for real-time analytics (optional)
- **Distributed Tracing** - OpenTelemetry + Jaeger for observability (optional)
- **Service Recommendation** - Routes users to appropriate services based on intent
- **Ethical Ad Matching** - Fair ad matching with fairness validation
- **Campaign Management** - Full advertising campaign lifecycle with budget tracking
- **Real-time Analytics** - Live metrics with WebSocket broadcasting
- **Fraud Detection** - Comprehensive fraud detection for clicks, impressions, conversions
- **A/B Testing** - Experiment management with statistical significance
- **Privacy Compliance** - GDPR-ready with consent management and audit trails

## Technology Stack

### Backend
- **Python 3.11+** - Primary language
- **FastAPI 0.104+** - Modern async web framework
- **SQLAlchemy 2.0** - Async ORM for database operations
- **Pydantic v2** - Data validation and settings management
- **PostgreSQL 15+** - Primary database with PgBouncer connection pooling
- **Redis/Valkey 8+** - Caching and session management

### Machine Learning
- **PyTorch** - Deep learning framework
- **Sentence Transformers** - Semantic similarity calculations
- **Transformers** - NLP model inference
- **NumPy** - Numerical computations

### Infrastructure
- **Docker & Docker Compose** - Containerization and orchestration
- **SearXNG** - Privacy-focused metasearch engine
- **Go Crawler** - High-performance web crawler and indexer (Go-based)
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
- **Commitizen** - Conventional commits enforcement

## Project Structure

```
intent-engine/
├── .github/                # GitHub Actions workflows (CI/CD, auto-version)
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
├── main_api.py             # FastAPI server (2700+ lines)
├── worker.py               # ARQ worker
├── database.py             # Database models
├── models.py               # Pydantic models
├── go_search_client.py     # Go crawler API client
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Container image
├── pyproject.toml          # Python project config (PEP 621)
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── Makefile                # Common development tasks
└── .pre-commit-config.yaml # Pre-commit hooks
```

## Building and Running

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **PostgreSQL 15+** (or use Docker)
- **Redis/Valkey 8+** (or use Docker)
- **4GB+ RAM** (for ML models)
- **2GB disk space**

### Quick Start (Docker - Recommended)

```bash
# Clone repository
git clone git@github.com-work:itxLikhith/intent-engine.git
cd intent-engine

# Start all services
docker-compose up -d

# Wait for initialization (~45 seconds)
sleep 45

# Verify installation
curl http://localhost:8000/

# Test search endpoint
curl http://localhost:8000/search -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"best laptop for programming"}'
```

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
make install        # Production dependencies
make dev            # Development dependencies + pre-commit hooks

# Run database migrations
python scripts/init_db_standalone.py

# Start API server
python -m uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload

# Or use Makefile
make docker-run     # Start all services via Docker
```

### Using Makefile (Common Tasks)

```bash
make install        # Install production dependencies
make dev            # Install dev dependencies + pre-commit
make test           # Run all tests
make test-cov       # Run tests with coverage
make lint           # Run linters
make format         # Format code
make check          # Run all checks
make security       # Run security scans
make docker-build   # Build Docker image
make docker-run     # Start all services
make docker-stop    # Stop all services
make migrations     # Run database migrations
make seed           # Seed database with sample data
make push (p)       # Auto commit + rebase + push
make quickpush (q)  # Auto commit + push (skip checks)
make fixpush (f)    # Auto-fix lint + commit + push
make clean          # Remove build artifacts
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=. --cov-report=html tests/

# Run specific test suite
pytest tests/test_extraction.py -v

# Run load tests
cd load_testing
locust -f locustfile.py

# Run performance tests
pytest perf_tests/ -v
```

### API Endpoints

**Core Intent:**
- `POST /extract-intent` - Extract structured intent from query
- `POST /search` - Unified privacy search with intent ranking
- `POST /rank-results` - Rank results based on user intent
- `POST /rank-urls` - Privacy-focused URL ranking
- `POST /recommend-services` - Recommend services based on intent
- `POST /match-ads` - Match ads with fairness validation

**Campaign Management:**
- `POST /campaigns`, `GET /campaigns/{id}`, `PUT /campaigns/{id}`, `DELETE /campaigns/{id}`
- `POST /adgroups`, `GET /adgroups/{id}`, `PUT /adgroups/{id}`
- `POST /ads`, `GET /ads/{id}`, `PUT /ads/{id}`, `DELETE /ads/{id}`

**Analytics & Reporting:**
- `GET /reports/campaign-performance`
- `GET /analytics/attribution/{id}`
- `GET /analytics/campaign-roi/{id}`
- `GET /analytics/trends/{metric}`

**Privacy & Compliance:**
- `POST /consent/record`, `GET /consent/{user_id}/{type}`
- `GET /audit-events`, `GET /audit-stats`

**System:**
- `GET /`, `GET /health`, `GET /status`, `GET /metrics`
- `GET /docs` (Swagger), `GET /redoc` (ReDoc)

## Development Conventions

### Commit Messages (Conventional Commits)

The project enforces **Conventional Commits** via pre-commit hooks:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat:` - New feature (triggers MINOR version bump)
- `fix:` - Bug fix (triggers PATCH version bump)
- `docs:` - Documentation only
- `style:` - Formatting, no code change
- `refactor:` - Code restructuring
- `perf:` - Performance improvement
- `test:` - Adding tests
- `chore:` - Maintenance tasks
- `ci:` - CI configuration

**Examples:**
```bash
feat: add query router for federated search
fix(ranking): resolve constraint satisfaction issue
docs: update API documentation
feat(search)!: change ranking algorithm  # Breaking change
```

### Code Quality

**Linting & Formatting:**
```bash
# Using Ruff (configured in pyproject.toml)
ruff check . --fix
ruff format .

# Line length: 120 characters
# Target version: Python 3.11
```

**Pre-commit Hooks:**
- Commitizen (commit message validation)
- Ruff (linting + formatting)
- Merge conflict checks
- File format validation (JSON, YAML, TOML)
- Security scans (Bandit, Safety)
- SQL linting (migrations)

### Testing Practices

- **Unit Tests:** `tests/` directory
- **Performance Tests:** `perf_tests/` directory
- **Load Testing:** `load_testing/` with Locust
- **Integration Tests:** Docker-based tests
- **Coverage Target:** >80% (configured in pytest)

```bash
# Run tests with coverage
pytest --cov=. --cov-report=html tests/

# Run in parallel
pytest -n auto --cov=. tests/
```

### Versioning

**Automatic Versioning:**
- Triggered on successful CI/CD pipeline completion
- Uses Commitizen for version bumps
- Tag format: `vX.Y.Z`
- Changelog auto-generated

**Manual Version Bump:**
```bash
python scripts/bump_version.py --patch  # or --minor, --major
```

### Environment Variables

Create `.env` from `.env.example`:

```bash
# Database
DATABASE_URL=postgresql://intent_user:password@localhost:5432/intent_engine

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=<generate-secure-random-string>

# SearXNG
SEARXNG_BASE_URL=http://localhost:8080

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

## Performance Benchmarks

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| Warm-up Time | <100ms | ~50ms | After initial model load |
| Processing Time | <50ms | ~30ms | Per query after warm-up |
| Memory Footprint | <500MB | ~450MB | RAM usage |
| Concurrent Requests | 1000+/sec | 200-300/sec | With Redis caching |
| Query Router Overhead | <20ms | ~15ms | P50 latency impact |
| Result Deduplication | 15-25% | 18% | Duplicate removal rate |
| Cache Hit Rate | >80% | 70-80% | With Redis enabled |

## Key Architectural Components

### Query Router (Phase 1)
Intent-based backend selection:
- Troubleshooting → SearXNG (community discussions)
- Comparison → Go Crawler (60%) + SearXNG (40%)
- Privacy-focused → Go Crawler (curated index)
- Breaking news → SearXNG (real-time news engines)
- Learning → Hybrid (50/50)

### Result Aggregator
- URL normalization (removes tracking parameters)
- Deduplication across backends
- Score normalization
- Source attribution

### Web Intent Extractor
- Primary goal detection from web content
- Skill level identification
- Topic extraction (TF-based)
- Confidence scoring

### Vector Store (Optional)
- Qdrant integration
- Intent embedding storage
- Semantic search capabilities

### Event Streaming (Optional)
- Kafka/Redpanda integration
- Intent extraction events
- Search execution events
- Click/conversion tracking

### Distributed Tracing (Optional)
- OpenTelemetry + Jaeger
- End-to-end request tracing
- Performance bottleneck identification

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Getting Started:** `docs/getting-started/`
- **Architecture:** `docs/architecture/`
- **Deployment:** `docs/deployment/`
- **Go Crawler:** `docs/go-crawler/`
- **Reference:** `docs/reference/`
- **Testing:** `docs/testing/`

**Key Documents:**
- [INDEX.md](INDEX.md) - Complete documentation index
- [README.md](README.md) - Main README
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [ARCHITECTURE_BLUEPRINT.md](ARCHITECTURE_BLUEPRINT.md) - Architecture design
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Implementation guide
- [docs/reference/VERSIONING_AND_RELEASES.md](docs/reference/VERSIONING_AND_RELEASES.md) - Versioning guide

## Support

- **GitHub Issues:** https://github.com/itxLikhith/intent-engine/issues
- **Email:** likhith.anony45@gmail.com
- **API Docs:** http://localhost:8000/docs (when running)
- **Grafana:** http://localhost:3000 (when running)

## License

**Intent Engine Community License (IECL) v1.0**
- ✅ Free for Non-Commercial Purposes
- ❌ Commercial use requires separate license
- 📧 Contact: anony45.omnipresent@proton.me for licensing
