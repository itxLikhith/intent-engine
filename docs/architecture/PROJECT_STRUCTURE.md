# Project Structure - Intent Engine

This document describes the organized project structure for the Intent Engine.

## Directory Layout

```
intent-engine/
├── .github/                    # GitHub Actions workflows and templates
├── .qwen/                      # Qwen Code configuration
├── abtesting/                  # A/B testing module
│   └── service.py
├── ads/                        # Ad matching module
│   └── matcher.py
├── analytics/                  # Real-time analytics module
│   ├── realtime.py
│   └── advanced.py
├── audit/                      # Audit trail module
│   └── audit_trail.py
├── bin/                        # Executable scripts (entry points)
├── config/                     # Configuration modules
│   ├── query_cache.py
│   └── redis_cache.py
├── core/                       # Core shared components
│   ├── schema.py               # UniversalIntent schema and enums
│   └── utils.py                # Shared utilities
├── data/                       # Local data directory (git-ignored)
├── demos/                      # Demo scripts for each module
│   ├── demo_search.py
│   ├── demo_service.py
│   ├── demo_ad.py
│   ├── demo_ranking.py
│   └── demo_searxng_ranking.py
├── docs/                       # Documentation
│   ├── README.md               # This file
│   ├── COMPREHENSIVE_GUIDE.md
│   ├── PROJECT_OVERVIEW.md
│   ├── Intent-Engine-Whitepaper.md
│   ├── Intent-Engine-Tech-Reference.md
│   ├── Intent-Engine-Visual-Guide.md
│   ├── PARENT_DIRECTORY_GUIDE.md
│   ├── STRESS_TEST_ANALYSIS.md
│   ├── TESTING_GUIDE.md
│   ├── CONFIGURATION_CHANGES.md
│   ├── CI_IMPROVEMENTS.md
│   ├── DEPLOYMENT_SUMMARY.md
│   ├── TESTING_PLAN.md
│   └── VERSIONING.md
├── extraction/                 # Intent extraction module
│   ├── extractor.py
│   └── constraints.py
├── fraud/                      # Fraud detection module
│   └── detector.py
├── grafana/                    # Grafana dashboards and provisioning
│   ├── dashboards/
│   └── provisioning/
├── load_testing/               # Load testing with Locust
│   └── locustfile.py
├── migrations/                 # SQL database migrations
│   ├── 001_create_missing_tables.sql
│   ├── 002_fix_click_tracking.sql
│   ├── 003_fix_conversion_fraud.sql
│   └── 004_fix_ab_tests.sql
├── pgbouncer/                  # PgBouncer configuration (git-ignored)
│   ├── pgbouncer.ini
│   └── userlist.txt
├── perf_tests/                 # Performance tests
│   ├── perf_test_ad.py
│   ├── perf_test_extractor.py
│   ├── perf_test_ranker.py
│   └── perf_test_service.py
├── privacy/                    # Privacy compliance module
│   ├── consent_manager.py
│   └── enhanced_privacy.py
├── ranking/                    # Ranking module
│   ├── ranker.py
│   ├── optimized_ranker.py
│   ├── url_ranker.py
│   ├── optimized_url_ranker.py
│   └── scoring.py
├── scripts/                    # Utility and maintenance scripts
│   ├── utils/                  # Utility sub-modules
│   ├── __init__.py
│   ├── autopush.py             # Auto-commit and push
│   ├── benchmark.py            # Benchmarking
│   ├── bump_version.py         # Version bumping
│   ├── check_db_schema.py      # Check DB schema
│   ├── check_db_status.py      # Check DB status
│   ├── check_model.py          # Check model status
│   ├── clear_db.py             # Clear database
│   ├── commit-gen.py           # Generate commit messages
│   ├── commit.py               # Commit helper
│   ├── init_db_direct.py       # Initialize DB directly
│   ├── init_db_standalone.py   # Initialize DB standalone
│   ├── init_sample_data.py     # Initialize sample data
│   ├── install_hooks.py        # Install Git hooks
│   ├── read_seed.py            # Read seed data
│   ├── reset_and_seed.py       # Reset and seed DB
│   ├── run_migrations.sh       # Run migrations
│   ├── seed_data.py            # Seed database
│   ├── seed_sample_data.py     # Seed sample data
│   ├── setup_git_hooks.py      # Setup Git hooks
│   ├── stress_test_all.py      # Stress tests
│   ├── test_api_comprehensive.sh
│   ├── test_api_docker.sh
│   └── verify_api_routes.py    # Verify API routes
├── searxng/                    # SearXNG integration
│   ├── client.py
│   ├── settings.yml
│   └── unified_search.py
├── services/                   # Service recommendation module
│   └── recommender.py
├── tests/                      # Unit and integration tests
│   └── __init__.py
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── .gitmessage                 # Git commit message template
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── CONTRIBUTING.md             # Contribution guidelines
├── docker-compose.searxng.yml  # SearXNG Docker Compose
├── docker-compose.yml          # Main Docker Compose configuration
├── Dockerfile                  # Docker image definition
├── LICENSE                     # Intent Engine Community License
├── main.py                     # CLI entry point
├── main_api.py                 # FastAPI server entry point
├── Makefile                    # Makefile for common tasks
├── models.py                   # Pydantic models for API
├── database.py                 # Database models and connection
├── privacy_core.py             # Privacy validation core
├── prometheus.yml              # Prometheus configuration
├── pyproject.toml              # Python project metadata (PEP 621)
├── README.md                   # Main README
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
└── worker.py                   # ARQ worker entry point
```

## Module Descriptions

### Core Modules

| Module | Description |
|--------|-------------|
| `core/` | Shared schemas, enums, and utilities used across the project |
| `extraction/` | Intent extraction from user queries using NLP |
| `ranking/` | Constraint satisfaction and intent-aligned ranking |
| `services/` | Service recommendation based on intent matching |
| `ads/` | Ethical ad matching with fairness validation |

### Privacy & Compliance

| Module | Description |
|--------|-------------|
| `privacy/` | Consent management and privacy controls |
| `audit/` | Audit trail logging for compliance |

### Analytics & Monitoring

| Module | Description |
|--------|-------------|
| `analytics/` | Real-time metrics and WebSocket broadcasting |
| `fraud/` | Fraud detection for clicks, impressions, conversions |
| `abtesting/` | A/B test management with statistical analysis |

### Integrations

| Module | Description |
|--------|-------------|
| `searxng/` | Privacy-focused search backend integration |

### Infrastructure

| Directory | Description |
|-----------|-------------|
| `config/` | Configuration modules for caching and settings |
| `migrations/` | SQL database migrations |
| `pgbouncer/` | PgBouncer connection pooler configuration |
| `grafana/` | Grafana dashboards and provisioning |
| `prometheus.yml` | Prometheus monitoring configuration |

### Testing

| Directory | Description |
|-----------|-------------|
| `tests/` | Unit and integration tests |
| `perf_tests/` | Performance benchmarking tests |
| `load_testing/` | Load testing with Locust |
| `demos/` | Demo scripts for each module |

### Scripts

| Directory | Description |
|-----------|-------------|
| `scripts/` | Utility scripts for development and maintenance |
| `bin/` | Executable entry points |

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Python project metadata, dependencies, and tool configuration |
| `requirements.txt` | Production Python dependencies |
| `requirements-dev.txt` | Development Python dependencies |
| `.env.example` | Environment variables template |
| `docker-compose.yml` | Docker Compose configuration for all services |
| `Dockerfile` | Docker image definition |
| `Makefile` | Common development tasks |
| `.pre-commit-config.yaml` | Pre-commit hooks configuration |
| `.gitmessage` | Git commit message template |

## Getting Started

### Local Development

```bash
# Clone the repository
git clone git@github.com-work:itxLikhith/intent-engine.git
cd intent-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Copy environment file
cp .env.example .env

# Start services with Docker
docker-compose up -d

# Run the API locally
python main_api.py
```

### Running Scripts

```bash
# Run database initialization
python scripts/init_db_standalone.py

# Seed sample data
python scripts/seed_sample_data.py

# Run benchmarks
python scripts/benchmark.py

# Run stress tests
python scripts/stress_test_all.py
```

### Using Makefile

```bash
# Install dependencies
make install

# Run tests
make test

# Format code
make format

# Build and run Docker
make docker-run

# View logs
make docker-logs
```

## Architecture Overview

The Intent Engine follows a modular architecture with four main phases:

1. **Intent Extraction** - Converts free-form queries into structured intent objects
2. **Constraint Satisfaction & Ranking** - Filters and ranks results based on user intent
3. **Service Recommendation** - Routes users to the most appropriate service
4. **Ad Matching** - Matches ads without discriminatory targeting

Additional capabilities include:
- Privacy & Compliance (GDPR-ready)
- Real-time Analytics
- Fraud Detection
- A/B Testing
- Campaign Management
- SearXNG Integration

## Conventions

### Python Code
- Follow PEP 8 style guidelines
- Use type hints where possible
- Maximum line length: 120 characters
- Use double quotes for strings

### Git Commits
- Follow Conventional Commits specification
- Use commitizen for commit message validation
- Format: `type(scope): description`

### Testing
- Unit tests in `tests/` directory
- Performance tests in `perf_tests/`
- Load tests in `load_testing/`
- Name test files: `test_*.py` or `*_test.py`

### Documentation
- All modules should have docstrings
- Public functions should have docstrings
- Keep documentation in `docs/` directory
