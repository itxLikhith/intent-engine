# Intent Engine - Documentation Index

> **Organized Documentation Structure** - Find what you need quickly

**Last Updated:** March 15, 2026 | **Version:** v2.0.0 - Self-Improving Search Loop

---

## 🎯 What's New in v2.0

### Self-Improving Search Loop
- **Every search makes the system smarter!**
- +634,000 URLs added from just 3 searches
- Redis caching (11x faster)
- Prometheus + Grafana monitoring
- Qdrant vector search integration

**New Documentation:**
- [Integration Guide](docs/INTEGRATION_GUIDE.md) - Complete v2.0 integration guide
- [Self-Improving Loop](docs/architecture/SELF_IMPROVING_LOOP.md) - Architecture deep dive

---

## 📚 Documentation Structure

```
docs/
├── README.md                  # Documentation overview
├── ORGANIZATION.md            # File organization guide
├── INTEGRATION_GUIDE.md       # v2.0 Integration Guide
│
├── getting-started/           # Quick start guides
├── deployment/                # Production deployment
├── architecture/              # System design
├── go-crawler/                # Go crawler docs
├── reference/                 # Technical reference
└── testing/                   # Testing guides
```

**Root Files (Essential Only):**
- `README.md` - Main project README
- `INDEX.md` - Documentation index
- `CHANGELOG_v2.md` - Current version changelog
- `V2_SUMMARY.md` - v2.0 release summary
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - License file

**Note:** Historical docs moved to `docs/` subfolders. See `docs/ORGANIZATION.md` for details.

---

## 🚀 Quick Start

**New to Intent Engine? Start here:**

| Document | Description | Time |
|----------|-------------|------|
| [README.md](README.md) | Main README with overview | 3 min |
| [Quick Start](docs/getting-started/QUICKSTART.md) | Complete installation guide | 5 min |
| [Production Setup](docs/getting-started/README_PRODUCTION.md) | Production-focused setup | 3 min |
| [Full Production Guide](docs/getting-started/README_PRODUCTION_FULL.md) | Complete deployment guide | 10 min |
| [Quick Reference](docs/getting-started/QUICK_REFERENCE.md) | Command cheat sheet | 1 min |

### Start the Search Engine (60 seconds)

**Linux/Mac:**
```bash
./scripts/production_start.sh start
sleep 60
curl http://localhost:8000/search -X POST -H "Content-Type: application/json" -d '{"query":"best laptop for programming"}'
```

**Windows PowerShell:**
```powershell
.\scripts\production_start.ps1 start
Start-Sleep -Seconds 60
curl http://localhost:8000/search -Method POST -ContentType "application/json" -Body '{"query":"best laptop for programming"}'
```

**Verify it's working:**
```bash
curl http://localhost:8000/health
docker exec intent-redis valkey-cli ZCARD crawl_queue  # Check crawl queue (1M+ URLs!)
```

---

## 📖 Documentation by Category

### Getting Started (`docs/getting-started/`)
- **[QUICKSTART.md](docs/getting-started/QUICKSTART.md)** - Complete quick start with troubleshooting
- **[README_PRODUCTION.md](docs/getting-started/README_PRODUCTION.md)** - Production README
- **[README_PRODUCTION_FULL.md](docs/getting-started/README_PRODUCTION_FULL.md)** - Full production deployment guide
- **[QUICK_REFERENCE.md](docs/getting-started/QUICK_REFERENCE.md)** - Quick reference card

### Deployment & Operations (`docs/deployment/`)
- **[DEPLOYMENT_CHECKLIST.md](docs/deployment/DEPLOYMENT_CHECKLIST.md)** - Production deployment checklist
- **[PRODUCTION_SETUP_SUMMARY.md](docs/deployment/PRODUCTION_SETUP_SUMMARY.md)** - What was created and how to use it
- **[PRODUCTION_SETUP_COMPLETE.md](docs/deployment/PRODUCTION_SETUP_COMPLETE.md)** - Latest setup summary & bug fixes
- **[PERFORMANCE_OPTIMIZATION_PLAN.md](docs/deployment/PERFORMANCE_OPTIMIZATION_PLAN.md)** - Performance optimization guide
- **[CI_IMPROVEMENTS.md](docs/deployment/CI_IMPROVEMENTS.md)** - CI/CD improvements
- **[RELEASE_AUTOMATION.md](docs/deployment/RELEASE_AUTOMATION.md)** - Release automation

### Architecture & Design (`docs/architecture/`)
- **[PROJECT_OVERVIEW.md](docs/architecture/PROJECT_OVERVIEW.md)** - Project overview
- **[PROJECT_STRUCTURE.md](docs/architecture/PROJECT_STRUCTURE.md)** - Project structure guide
- **[Intent-Engine-Whitepaper.md](docs/architecture/Intent-Engine-Whitepaper.md)** - Technical whitepaper
- **[SELF_IMPROVING_LOOP.md](docs/architecture/SELF_IMPROVING_LOOP.md)** - 🆕 Self-improving search loop architecture (NEW!)

### Go Crawler & Indexer (`docs/go-crawler/`)
- **[README.md](docs/go-crawler/README.md)** - Go crawler overview
- **[README_PRODUCTION.md](docs/go-crawler/README_PRODUCTION.md)** - Production setup
- **[QUICKSTART.md](docs/go-crawler/QUICKSTART.md)** - Go crawler quick start
- **[GO_CRAWLER_SETUP_GUIDE.md](docs/go-crawler/GO_CRAWLER_SETUP_GUIDE.md)** - Complete setup guide
- **[GO_CRAWLER_INDEXER_PLAN.md](docs/go-crawler/GO_CRAWLER_INDEXER_PLAN.md)** - Architecture plan
- **[INTENT_INDEXER_README.md](docs/go-crawler/INTENT_INDEXER_README.md)** - Indexer documentation
- **[PHASE_1_IMPLEMENTATION.md](docs/go-crawler/PHASE_1_IMPLEMENTATION.md)** - Implementation details
- **[IMPLEMENTATION_SUMMARY.md](docs/go-crawler/IMPLEMENTATION_SUMMARY.md)** - Implementation summary
- **[GO_CRAWLER_PRODUCTION_SUMMARY.md](docs/go-crawler/GO_CRAWLER_PRODUCTION_SUMMARY.md)** - Production summary
- **[FILE_REFERENCE.md](docs/go-crawler/FILE_REFERENCE.md)** - File reference
- **[BUG_FIXES.md](docs/go-crawler/BUG_FIXES.md)** - Bug fixes
- **[CRAWLER_TEST_RESULTS.md](docs/go-crawler/CRAWLER_TEST_RESULTS.md)** - Test results
- **[DOCKER_TEST_RESULTS.md](docs/go-crawler/DOCKER_TEST_RESULTS.md)** - Docker test results

### Technical Reference (`docs/reference/`)
- **[Intent-Engine-Tech-Reference.md](docs/reference/Intent-Engine-Tech-Reference.md)** - Technical reference
- **[Intent-Engine-Visual-Guide.md](docs/reference/Intent-Engine-Visual-Guide.md)** - Visual guide
- **[COMPREHENSIVE_GUIDE.md](docs/reference/COMPREHENSIVE_GUIDE.md)** - Comprehensive guide
- **[CONFIGURATION_CHANGES.md](docs/reference/CONFIGURATION_CHANGES.md)** - Configuration changes
- **[VERSIONING.md](docs/reference/VERSIONING.md)** - Versioning policy
- **[CHANGELOG.md](docs/reference/CHANGELOG.md)** - Detailed changelog
- **[VERSIONING_AND_RELEASES.md](docs/reference/VERSIONING_AND_RELEASES.md)** - Versioning and release guide
- **[VERSIONING_FIX_SUMMARY.md](docs/reference/VERSIONING_FIX_SUMMARY.md)** - Latest versioning fixes
- **[PARENT_DIRECTORY_GUIDE.md](docs/reference/PARENT_DIRECTORY_GUIDE.md)** - Parent directory guide

### Testing & Performance (`docs/testing/`)
- **[TESTING_GUIDE.md](docs/testing/TESTING_GUIDE.md)** - Testing guide
- **[TESTING_PLAN.md](docs/testing/TESTING_PLAN.md)** - Testing plan
- **[STRESS_TEST_ANALYSIS.md](docs/testing/STRESS_TEST_ANALYSIS.md)** - Stress test analysis

---

## 🎯 Common Tasks

### Starting the Search Engine

**Linux/Mac:**
```bash
# Using production script (recommended)
./scripts/production_start.sh start

# Or using docker-compose directly
docker-compose up -d

# View logs
docker-compose logs -f intent-engine-api
```

**Windows PowerShell:**
```powershell
# Using production script (recommended)
.\scripts\production_start.ps1 start

# Or using docker-compose directly
docker-compose up -d

# View logs
docker-compose logs -f intent-engine-api
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Extract intent from a query
curl -X POST http://localhost:8000/extract-intent \
  -H "Content-Type: application/json" \
  -d '{
    "product": "search",
    "input": {"text": "best laptop for programming under 50000 rupees"},
    "context": {"sessionId": "test-123", "userLocale": "en-US"}
  }' | jq

# Unified search with intent ranking
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best python tutorials for beginners",
    "extract_intent": true,
    "rank_results": true,
    "max_results": 10
  }' | jq
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f intent-engine-api
docker-compose logs -f searxng
docker-compose logs -f postgres
docker-compose logs -f redis

# Last 100 lines
docker-compose logs --tail=100 intent-engine-api
```

### Health Checks

```bash
# API health
curl http://localhost:8000/

# Detailed health (checks DB, Redis, SearXNG)
curl http://localhost:8000/health

# SearXNG health
curl http://localhost:8080/healthz

# Go Crawler health
curl http://localhost:8081/health

# PostgreSQL health
docker exec intent-engine-postgres pg_isready -U intent_user

# Redis health
docker exec intent-redis redis-cli ping
```

### Service Status

```bash
# Check all services
docker-compose ps

# Check specific service
docker-compose ps intent-engine-api

# View resource usage
docker stats
```

### Database Operations

```bash
# Access PostgreSQL shell
docker exec -it intent-engine-postgres psql -U intent_user -d intent_engine

# Run migrations
docker-compose exec intent-engine-api python scripts/init_db_standalone.py

# Backup database
docker exec intent-engine-postgres pg_dump -U intent_user intent_engine > backup.sql

# Restore database
docker exec -i intent-engine-postgres psql -U intent_user intent_engine < backup.sql
```

### Running Tests

```bash
# Run all tests
docker-compose exec intent-engine-api pytest tests/ -v

# Run with coverage
docker-compose exec intent-engine-api pytest --cov=. --cov-report=html tests/

# Run specific test suite
docker-compose exec intent-engine-api pytest tests/test_extraction.py -v

# Run load tests
cd load_testing
locust -f locustfile.py
```

### Accessing Monitoring Dashboards

```bash
# Grafana (username: admin, password: grafana_secure_password_change_in_prod)
open http://localhost:3000

# Prometheus
open http://localhost:9090

# Jaeger (distributed tracing)
open http://localhost:16686

# API Documentation (Swagger)
open http://localhost:8000/docs

# API Documentation (ReDoc)
open http://localhost:8000/redoc
```

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes data)
docker-compose down -v

# Stop specific service
docker-compose stop intent-engine-api

# Restart specific service
docker-compose restart intent-engine-api
```

---

## 🔗 External Resources

### Project Links
- **GitHub Repository**: https://github.com/itxLikhith/intent-engine
- **Issues**: https://github.com/itxLikhith/intent-engine/issues
- **PyPI Package**: https://pypi.org/project/intent-engine/

### Technology Documentation
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **SearXNG Documentation**: https://docs.searxng.org/
- **Docker Documentation**: https://docs.docker.com/
- **PostgreSQL Documentation**: https://www.postgresql.org/docs/
- **Redis Documentation**: https://redis.io/docs/
- **Pydantic Documentation**: https://docs.pydantic.dev/
- **SQLAlchemy Documentation**: https://docs.sqlalchemy.org/

### Monitoring & Observability
- **Prometheus Documentation**: https://prometheus.io/docs/
- **Grafana Documentation**: https://grafana.com/docs/
- **Jaeger Documentation**: https://www.jaegertracing.io/docs/
- **OpenTelemetry Documentation**: https://opentelemetry.io/docs/

---

## 📞 Support

### Getting Help
- **GitHub Issues**: [Open an issue](https://github.com/itxLikhith/intent-engine/issues) for bug reports and feature requests
- **API Documentation**: http://localhost:8000/docs (when running)
- **Grafana Dashboards**: http://localhost:3000 (when running)
- **Email**: likhith.anony45@gmail.com

### Troubleshooting
- Check the [FAQ](docs/reference/FAQ.md) for common issues
- Review [logs](#viewing-logs) for error messages
- Verify [service health](#health-checks)
- Check [resource usage](#service-status)

### Contributing
- Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Follow the [Conventional Commits](https://www.conventionalcommits.org/) standard
- Run tests before submitting PRs: `pytest tests/ -v`

---

**Maintained by:** Intent Engine Team  
**License:** Intent Engine Community License (IECL) v1.0  
**Version:** v0.3.0  
**Last Updated:** March 15, 2026
