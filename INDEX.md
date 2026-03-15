# Intent Engine - Documentation Index

> **Organized Documentation Structure** - Find what you need quickly

**Last Updated:** March 15, 2026 | **Version:** v0.3.0

---

## 📚 Documentation Structure

```
docs/
├── getting-started/     # Quick start guides and tutorials
├── deployment/          # Production deployment and operations
├── architecture/        # System design and architecture
├── go-crawler/          # Go crawler and indexer documentation
├── reference/           # Technical reference and API docs
└── testing/             # Testing guides and performance analysis
```

---

## 🚀 Getting Started

**New to Intent Engine? Start here:**

| Document | Description | Time |
|----------|-------------|------|
| [Quick Start](docs/getting-started/QUICKSTART.md) | Complete installation guide | 5 min |
| [Production Setup](docs/getting-started/README_PRODUCTION.md) | Production-focused setup | 3 min |
| [Full Production Guide](docs/getting-started/README_PRODUCTION_FULL.md) | Complete deployment guide | 10 min |
| [Quick Reference](docs/getting-started/QUICK_REFERENCE.md) | Command cheat sheet | 1 min |

### Start the Search Engine (60 seconds)

```bash
# Linux/Mac
./scripts/production_start.sh start
sleep 60
curl http://localhost:8000/search -X POST -H "Content-Type: application/json" -d '{"query":"best laptop for programming"}'

# Windows PowerShell
.\scripts\production_start.ps1 start
Start-Sleep -Seconds 60
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
- **[CHANGELOG.md](docs/reference/CHANGELOG.md)** - Changelog
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
./scripts/production_start.sh start
```

**Windows:**
```powershell
.\scripts\production_start.ps1 start
```

### Testing the API

```bash
curl http://localhost:8000/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"best laptop for programming"}'
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f intent-engine-api
```

### Health Checks

```bash
# API
curl http://localhost:8000/

# Detailed health
curl http://localhost:8000/health

# SearXNG
curl http://localhost:8080/healthz
```

---

## 🔗 External Resources

- **GitHub Repository**: https://github.com/itxLikhith/intent-engine
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **SearXNG Documentation**: https://docs.searxng.org/
- **Docker Documentation**: https://docs.docker.com/
- **PostgreSQL Documentation**: https://www.postgresql.org/docs/

---

## 📞 Support

- **Issues**: https://github.com/itxLikhith/intent-engine/issues
- **API Docs**: http://localhost:8000/docs (when running)
- **Grafana Dashboards**: http://localhost:3000 (when running)

---

**Maintained by:** Intent Engine Team  
**License:** Intent Engine Community License (IECL) v1.0
