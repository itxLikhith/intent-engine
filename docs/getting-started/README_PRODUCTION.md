# Intent Engine - Production Search Backend

> **Privacy-First Search Engine Backend** - Works out of the box with SearXNG integration

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## Quick Start (60 seconds)

### Prerequisites
- Docker and Docker Compose installed
- 4GB+ RAM available
- 2GB disk space

### Start the Search Engine

**Linux/Mac:**
```bash
# Clone and start
git clone git@github.com-work:itxLikhith/intent-engine.git
cd intent-engine
./scripts/production_start.sh start

# Wait for initialization (~60 seconds)
sleep 60

# Test search
curl http://localhost:8000/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"best laptop for programming under $1000"}'
```

**Windows PowerShell:**
```powershell
# Clone and start
git clone git@github.com-work:itxLikhith/intent-engine.git
cd intent-engine
.\scripts\production_start.ps1 start

# Wait for initialization
Start-Sleep -Seconds 60

# Test search
curl http://localhost:8000/search -X POST -H "Content-Type: application/json" -d '{\"query\":\"best laptop for programming under $1000\"}'
```

### Alternative: Direct Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Wait for services
sleep 60

# Check status
docker-compose -f docker-compose.prod.yml ps

# Test search
curl http://localhost:8000/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"how to set up E2E encrypted email"}'
```

## What You Get

| Service | Port | Description |
|---------|------|-------------|
| **Search API** | 8000 | Main search endpoint with intent extraction |
| **SearXNG** | 8080 | Privacy-focused search backend |
| **PostgreSQL** | 5432 | Primary database |
| **Redis** | 6379 | Caching layer |

## Search API Endpoints

### Primary Search Endpoint

```bash
POST /search
```

**Request:**
```json
{
  "query": "best laptop for programming under $1000",
  "extract_intent": true,
  "rank_results": true,
  "categories": ["general"],
  "safe_search": 0,
  "language": "en"
}
```

**Response:**
```json
{
  "query": "best laptop for programming under $1000",
  "results": [
    {
      "url": "https://example.com/laptop-guide",
      "title": "Best Laptops for Programming 2024",
      "content": "Comprehensive guide to choosing...",
      "rank": 1,
      "ranked_score": 0.92,
      "original_score": 0.85,
      "privacy_score": 0.88,
      "match_reasons": ["Matches BUY intent", "High privacy rating"]
    }
  ],
  "total_results": 10,
  "processing_time_ms": 245,
  "extracted_intent": {
    "goal": "COMPARE",
    "constraints": [
      {"type": "range", "dimension": "price", "value": "0-1000"}
    ],
    "use_cases": ["SHOPPING"],
    "confidence": 0.89
  },
  "ranking_applied": true,
  "privacy_enhanced": true,
  "tracking_blocked": true
}
```

### Intent Extraction

```bash
POST /extract-intent
```

```json
{
  "product": "search",
  "input": {"text": "how to set up E2E encrypted email on Android"},
  "context": {"sessionId": "test-123", "userLocale": "en-US"}
}
```

### URL Ranking

```bash
POST /rank-urls
```

```json
{
  "query": "privacy-focused email setup",
  "urls": [
    "https://example.com/guide1",
    "https://example.com/guide2"
  ],
  "intent": {...}
}
```

## Management Commands

### Using Startup Scripts

**Linux/Mac:**
```bash
# Start services
./scripts/production_start.sh start

# Stop services
./scripts/production_start.sh stop

# Restart services
./scripts/production_start.sh restart

# View logs
./scripts/production_start.sh logs
./scripts/production_start.sh logs intent-engine-api

# Check health
./scripts/production_start.sh health

# Show status
./scripts/production_start.sh status
```

**Windows PowerShell:**
```powershell
# Start services
.\scripts\production_start.ps1 start

# Stop services
.\scripts\production_start.ps1 stop

# View logs
.\scripts\production_start.ps1 logs
.\scripts\production_start.ps1 logs intent-engine-api

# Check health
.\scripts\production_start.ps1 health
```

### Using Docker Compose Directly

```bash
# Start
docker-compose -f docker-compose.prod.yml up -d

# Stop
docker-compose -f docker-compose.prod.yml down

# Restart
docker-compose -f docker-compose.prod.yml restart

# View logs
docker-compose -f docker-compose.prod.yml logs -f
docker-compose -f docker-compose.prod.yml logs -f intent-engine-api

# Check status
docker-compose -f docker-compose.prod.yml ps

# Scale API workers
docker-compose -f docker-compose.prod.yml up -d --scale intent-engine-api=3
```

## Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=postgresql://intent_user:intent_secure_password_change_in_prod@postgres:5432/intent_engine

# Security (IMPORTANT: Change in production!)
SECRET_KEY=your-secure-random-string-here

# SearXNG
SEARXNG_BASE_URL=http://searxng:8080

# Redis
REDIS_URL=redis://redis:6379/0

# Rate Limiting
RATE_LIMIT_DEFAULT=100/minute
```

### Security Checklist

Before deploying to production:

- [ ] Change `SECRET_KEY` to a secure random string
- [ ] Change PostgreSQL password in `.env` and `docker-compose.prod.yml`
- [ ] Update `CORS_ORIGINS` to your actual domains
- [ ] Enable SSL/TLS for API endpoints
- [ ] Review rate limits for your use case

## Testing

### Health Checks

```bash
# API health
curl http://localhost:8000/

# Detailed health
curl http://localhost:8000/health

# SearXNG health
curl http://localhost:8080/healthz

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Search Tests

```bash
# Basic search
curl http://localhost:8000/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"weather today"}'

# Search with intent extraction
curl http://localhost:8000/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best smartphone under 500",
    "extract_intent": true,
    "rank_results": true
  }'

# Privacy-focused search
curl http://localhost:8000/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "encrypted messaging apps",
    "exclude_big_tech": true,
    "min_privacy_score": 0.7
  }'
```

## Troubleshooting

### Services Won't Start

```bash
# Check Docker logs
docker-compose -f docker-compose.prod.yml logs

# Restart specific service
docker-compose -f docker-compose.prod.yml restart intent-engine-api

# Rebuild containers
docker-compose -f docker-compose.prod.yml up -d --build
```

### Database Issues

```bash
# View database logs
docker logs intent-engine-postgres

# Reset database (WARNING: deletes all data)
docker-compose -f docker-compose.prod.yml down -v
docker-compose -f docker-compose.prod.yml up -d
```

### Search Returns No Results

1. Check SearXNG is running: `curl http://localhost:8080/healthz`
2. Check SearXNG logs: `docker logs searxng`
3. Test SearXNG directly: `curl http://localhost:8080/search?q=test`

### High Memory Usage

```bash
# Reduce API workers
# Edit docker-compose.prod.yml: WORKERS=1

# Limit Redis memory (already set to 512MB)
# Limit PostgreSQL memory (already set to 1G)
```

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Warm-up Time | <100ms |
| Processing Time | 200-400ms per search |
| Concurrent Requests | 200-300/sec |
| Memory Usage | ~1.5GB total |

### Optimization Tips

1. **Enable Redis caching** - Already enabled by default
2. **Scale API workers** - `docker-compose up -d --scale intent-engine-api=3`
3. **Adjust SearXNG engines** - Edit `searxng/settings.yml`
4. **Tune connection pools** - Adjust `DATABASE_POOL_SIZE` in `.env`

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Search API     │ Port 8000
│  (FastAPI)      │
└────────┬────────┘
         │
    ┌────┴────┬────────────┐
    │         │            │
    ▼         ▼            ▼
┌───────┐ ┌──────┐   ┌──────────┐
│SearXNG│ │Redis │   │PostgreSQL│
│:8080  │ │:6379 │   │  :5432   │
└───────┘ └──────┘   └──────────┘
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Monitoring

### Prometheus Metrics

```bash
# View metrics
curl http://localhost:8000/metrics

# Key metrics:
# - intent_extraction_requests_total
# - unified_search_requests_total
# - intent_extraction_latency_seconds
# - active_sessions
```

### Logs

```bash
# API logs
docker logs -f intent-engine-api

# Search logs
docker logs -f searxng

# Database logs
docker logs -f intent-engine-postgres

# Cache logs
docker logs -f intent-redis
```

## Backup & Restore

### Database Backup

```bash
# Backup
docker exec intent-engine-postgres pg_dump \
  -U intent_user \
  intent_engine \
  > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore
docker exec -i intent-engine-postgres psql \
  -U intent_user \
  intent_engine \
  < backup_20260314_120000.sql
```

## Updates

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose -f docker-compose.prod.yml up -d --build
```

## Support

- **Issues**: https://github.com/itxLikhith/intent-engine/issues
- **Documentation**: See main README.md for detailed guides

## License

This project is licensed under the **Intent Engine Community License (IECL) v1.0** - see the [LICENSE](../../LICENSE) file for details.

**Key Points:**
- ✅ Free for Non-Commercial Purposes (personal, educational, academic, internal evaluation)
- ❌ Commercial use requires separate Commercial License
- 📧 Contact: anony45.omnipresent@proton.me for Commercial Licensing
