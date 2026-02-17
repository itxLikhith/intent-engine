# Intent Engine - Comprehensive Documentation

## 📚 Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Architecture Overview](#architecture-overview)
3. [API Reference](#api-reference)
4. [Configuration Guide](#configuration-guide)
5. [Troubleshooting](#troubleshooting)
6. [Load Testing Guide](#load-testing-guide)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security Best Practices](#security-best-practices)
9. [Performance Tuning](#performance-tuning)
10. [Deployment Guide](#deployment-guide)

---

## Quick Start Guide

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- 4GB+ RAM (for ML models)
- 2GB disk space

### Installation

```bash
# Clone repository
git clone git@github.com-work:itxLikhith/intent-engine.git
cd intent-engine

# Start with Docker
docker-compose up -d

# Wait for initialization
sleep 45

# Verify installation
curl http://localhost:8000/
```

### First API Call

```bash
curl -X POST http://localhost:8000/extract-intent \
  -H "Content-Type: application/json" \
  -d '{
    "product": "search",
    "input": {"text": "best laptop for programming"},
    "context": {"sessionId": "test-123", "userLocale": "en-US"}
  }'
```

---

## Architecture Overview

### Four-Phase Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌──────────┐
│  1. INTENT      │     │  2. CONSTRAINT   │     │  3. SERVICE  │     │  4. AD   │
│  EXTRACTION     │────▶│  & RANKING       │────▶│  REC         │────▶│  MATCH   │
└─────────────────┘     └──────────────────┘     └──────────────┘     └──────────┘
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Intent Engine API                        │
├─────────────┬──────────────┬──────────────┬─────────────────┤
│ Extraction  │   Ranking    │  Services    │  Ads            │
│  Module     │   Module     │  Module      │  Module         │
├─────────────┴──────────────┴──────────────┴─────────────────┤
│                    Caching Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Embedding    │  │ Result       │  │ URL Analysis     │   │
│  │ Cache        │  │ Cache        │  │ Cache            │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Database Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ SQLite       │  │ PostgreSQL   │  │ Connection Pool  │   │
│  │ (dev)        │  │ (prod)       │  │                  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## API Reference

### Core Endpoints

#### 1. Health Check
```http
GET /
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-16T08:36:43.034818"
}
```

#### 2. Extract Intent
```http
POST /extract-intent
Content-Type: application/json

{
  "product": "search",
  "input": {"text": "How to set up encrypted email?"},
  "context": {
    "sessionId": "test-session",
    "userLocale": "en-US"
  }
}
```

Response:
```json
{
  "intent": {
    "intentId": "test-session_...",
    "declared": {
      "query": "How to set up encrypted email?",
      "goal": "learn",
      "skillLevel": "intermediate",
      "constraints": [...]
    },
    "inferred": {
      "useCases": ["learning", "comparison"],
      "complexity": "moderate"
    }
  },
  "extractionMetrics": {
    "confidence": 0.85,
    "processingTime": "45ms"
  }
}
```

#### 3. Rank URLs
```http
POST /rank-urls
Content-Type: application/json

{
  "query": "privacy email service",
  "urls": [
    "https://protonmail.com",
    "https://tutanota.com",
    "https://gmail.com"
  ],
  "options": {
    "exclude_big_tech": true,
    "min_privacy_score": 0.6,
    "weights": {
      "relevance": 0.40,
      "privacy": 0.30,
      "quality": 0.20,
      "ethics": 0.10
    }
  }
}
```

Response:
```json
{
  "query": "privacy email service",
  "ranked_urls": [
    {
      "url": "https://protonmail.com",
      "privacy_score": 0.95,
      "relevance_score": 0.85,
      "final_score": 0.89
    }
  ],
  "processing_time_ms": 120,
  "cache_hit_rate": 0.85
}
```

---

## API Usage Examples

### Intent Extraction Examples

#### Example 1: Basic Information Query
```bash
curl -X POST http://localhost:8000/extract-intent \
  -H "Content-Type: application/json" \
  -d '{
    "product": "search",
    "input": {"text": "How to learn Python programming?"},
    "context": {"sessionId": "session_123"}
  }'
```

#### Example 2: Comparison Query with Constraints
```bash
curl -X POST http://localhost:8000/extract-intent \
  -H "Content-Type: application/json" \
  -d '{
    "product": "search",
    "input": {"text": "best laptop for programming under 50000 rupees, open source preferred"},
    "context": {"sessionId": "session_456"}
  }'
```

#### Example 3: Troubleshooting Query
```bash
curl -X POST http://localhost:8000/extract-intent \
  -H "Content-Type: application/json" \
  -d '{
    "product": "search",
    "input": {"text": "fix E2E encrypted email not syncing on Android"},
    "context": {"sessionId": "session_789"}
  }'
```

### URL Ranking Examples

#### Example 1: Privacy-Focused Search
```bash
curl -X POST http://localhost:8000/rank-urls \
  -H "Content-Type: application/json" \
  -d '{
    "query": "privacy-focused email providers",
    "urls": [
      "https://protonmail.com",
      "https://tutanota.com",
      "https://mail.com"
    ],
    "options": {
      "exclude_big_tech": true,
      "min_privacy_score": 0.6
    }
  }'
```

#### Example 2: Custom Weights
```bash
curl -X POST http://localhost:8000/rank-urls \
  -H "Content-Type: application/json" \
  -d '{
    "query": "open source project management tools",
    "urls": [
      "https://github.com",
      "https://gitlab.com",
      "https://trello.com"
    ],
    "options": {
      "weights": {
        "relevance": 0.30,
        "privacy": 0.40,
        "quality": 0.20,
        "ethics": 0.10
      }
    }
  }'
```

### Campaign Management Examples

#### Example 1: Create Campaign
```bash
curl -X POST http://localhost:8000/campaigns \
  -H "Content-Type: application/json" \
  -d '{
    "advertiser_id": 1,
    "name": "Privacy Tools Campaign",
    "start_date": "2026-02-17T00:00:00",
    "end_date": "2026-03-17T23:59:59",
    "budget": 5000.00,
    "daily_budget": 200.00,
    "status": "active"
  }'
```

#### Example 2: List Campaigns with Filters
```bash
curl -X GET "http://localhost:8000/campaigns?advertiser_id=1&status=active&skip=0&limit=10"
```

#### Example 3: Update Campaign
```bash
curl -X PUT http://localhost:8000/campaigns/1 \
  -H "Content-Type: application/json" \
  -d '{
    "budget": 7500.00,
    "daily_budget": 300.00
  }'
```

### Ad Management Examples

#### Example 1: Create Ad
```bash
curl -X POST http://localhost:8000/ads \
  -H "Content-Type: application/json" \
  -d '{
    "advertiser_id": 1,
    "ad_group_id": 1,
    "title": "ProtonMail - Secure Email",
    "description": "Privacy-focused encrypted email service",
    "url": "https://protonmail.com",
    "targeting_constraints": {"device_type": ["mobile", "desktop"]},
    "ethical_tags": ["privacy", "open_source", "encryption"],
    "quality_score": 0.85,
    "creative_format": "banner",
    "bid_amount": 2.50,
    "status": "active",
    "approval_status": "pending"
  }'
```

#### Example 2: Match Ads to Intent
```bash
curl -X POST http://localhost:8000/match-ads \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "declared": {
        "query": "secure email provider",
        "goal": "learn"
      },
      "inferred": {
        "useCases": ["comparison"],
        "ethicalSignals": [{"dimension": "privacy", "preference": "high"}]
      }
    },
    "ad_inventory": [],
    "config": {"max_ads": 5}
  }'
```

### Privacy & Compliance Examples

#### Example 1: Record Consent
```bash
curl -X POST http://localhost:8000/consent/record \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "consent_type": "analytics",
    "granted": true,
    "expires_in_days": 365
  }'
```

#### Example 2: Get Consent Status
```bash
curl -X GET http://localhost:8000/consent/user_123/analytics
```

#### Example 3: Withdraw Consent
```bash
curl -X POST http://localhost:8000/consent/withdraw/user_123/analytics
```

### Analytics Examples

#### Example 1: Get Campaign ROI
```bash
curl -X GET http://localhost:8000/analytics/campaign-roi/1
```

#### Example 2: Get Attribution Data
```bash
curl -X GET http://localhost:8000/analytics/attribution/conversion_123
```

#### Example 3: Get Top Ads
```bash
curl -X GET http://localhost:8000/analytics/top-ads?limit=10&metric=ctr
```

### A/B Testing Examples

#### Example 1: Create A/B Test
```bash
curl -X POST http://localhost:8000/abtests \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ad Creative Test",
    "campaign_id": 1,
    "status": "draft",
    "primary_metric": "ctr",
    "traffic_allocation": 0.5,
    "min_sample_size": 1000,
    "confidence_level": 0.95
  }'
```

### Fraud Detection Examples

#### Example 1: Report Fraud Event
```bash
curl -X POST http://localhost:8000/fraud-detection \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": 123,
    "event_type": "click_fraud",
    "ad_id": 1,
    "reason": "Rapid clicks from same IP",
    "severity": "high",
    "metadata": {"ip_hash": "abc123", "click_count": 50}
  }'
```

---

## Troubleshooting

### Common Issues

#### 1. Server Won't Start

**Symptom:**
```
Error: Connection refused
```

**Solutions:**
```bash
# Check if port is already in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Check Docker logs
docker-compose logs intent-engine-api

# Verify database permissions
ls -la intent_engine.db  # Check file exists and is writable
```

#### 2. Slow Response Times

**Symptom:**
Response times > 500ms

**Check:**
```bash
# Check cache hit rate
curl http://localhost:8000/admin/cache-stats

# Monitor memory usage
ps aux | grep python

# Check database performance
docker exec -it <container> sqlite3 /app/intent_engine.db "PRAGMA cache_size;"
```

**Solutions:**
- Increase cache sizes in config
- Check if embedding model is cached
- Verify database indexes are created
- Check for memory leaks

#### 3. Embedding Model Loading Errors

**Symptom:**
```
Warning: Transformers library not available. Using mock embeddings.
```

**Solutions:**
```bash
# Verify torch installation
python -c "import torch; print(torch.__version__)"

# Reinstall with CPU support
pip install torch==2.1.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Check disk space (models need ~500MB)
df -h
```

#### 4. Database Lock Errors

**Symptom:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**
```python
# In database.py, ensure check_same_thread=False
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True
)
```

#### 5. Out of Memory Errors

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
```bash
# Reduce cache sizes
export EMBEDDING_CACHE_SIZE=5000
export QUERY_CACHE_SIZE=2500

# Limit Docker memory
docker-compose up -d --memory=2g

# Monitor memory
watch -n 1 'ps aux | grep python'
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Performance Profiling

```bash
# Profile with cProfile
python -m cProfile -o profile.stats main_api.py

# View results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

---

## Load Testing Guide

### Using Locust

#### Installation
```bash
pip install locust
```

#### Basic Load Test
```bash
# Start Locust
locust -f load_testing/locustfile.py --host=http://localhost:8000

# Open browser
# http://localhost:8089

# Configure:
# - Number of users: 100
# - Spawn rate: 10
# - Host: http://localhost:8000
```

#### Stress Test Scenarios

**Scenario 1: Normal Load (100 users)**
```bash
locust -f load_testing/locustfile.py \
  --host=http://localhost:8000 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m
```

**Scenario 2: Peak Load (500 users)**
```bash
locust -f load_testing/locustfile.py \
  --host=http://localhost:8000 \
  --users 500 \
  --spawn-rate 50 \
  --run-time 10m
```

**Scenario 3: Spike Test (1000 users)**
```bash
locust -f load_testing/locustfile.py \
  --host=http://localhost:8000 \
  --users 1000 \
  --spawn-rate 100 \
  --run-time 2m
```

### Interpreting Results

**Good Performance:**
- 95th percentile < 200ms
- Error rate < 1%
- RPS > 50
- Memory stable

**Warning Signs:**
- 95th percentile > 500ms
- Error rate > 5%
- Memory growing continuously
- Database locks

---

## Monitoring & Observability

### Prometheus Metrics

Available at `http://localhost:8000/metrics`:

```
# Request metrics
intent_extraction_latency_seconds_bucket{le="0.1"} 150
intent_extraction_latency_seconds_count 200
intent_extraction_latency_seconds_sum 15.5

ranking_throughput_total 500
ad_matching_success_total 450

# Custom metrics
cache_hit_rate 0.85
active_sessions 42
```

### Grafana Dashboard

Import dashboard JSON from `grafana/dashboards/intent_engine_overview.json`

Key panels:
- Request latency percentiles
- Cache hit rates
- Error rates
- Memory usage
- Database connections

### Health Checks

```bash
# Basic health
curl http://localhost:8000/

# Detailed status
curl http://localhost:8000/status

# Database health
python -c "from database import engine; print(engine.execute('SELECT 1').fetchone())"
```

---

## Security Best Practices

### 1. API Security

```python
# Rate limiting (add to main_api.py)
from slowapi import Limiter

limiter = Limiter(key_func=lambda: request.client.host)
app = FastAPI()
app.state.limiter = limiter

@app.post("/extract-intent")
@limiter.limit("100/minute")
async def extract_intent(request: Request, ...):
    ...
```

### 2. Input Validation

```python
# Validate all inputs
from pydantic import validator

class IntentRequest(BaseModel):
    query: str
    
    @validator('query')
    def validate_query(cls, v):
        if len(v) > 1000:
            raise ValueError('Query too long')
        return v
```

### 3. Database Security

- Use PostgreSQL in production (not SQLite)
- Enable SSL connections
- Use strong passwords
- Regular backups

---

## Performance Tuning

### Cache Configuration

```python
# config/optimized_cache.py
EMBEDDING_CACHE_SIZE = 10000  # Increase for more RAM
QUERY_CACHE_SIZE = 5000
DISK_CACHE_SIZE_MB = 500
```

### Database Optimization

```sql
-- PostgreSQL optimizations
SET shared_buffers = '256MB';
SET effective_cache_size = '1GB';
SET work_mem = '16MB';
```

### Worker Configuration

```bash
# For high traffic, increase workers
uvicorn main_api:app --workers 4 --host 0.0.0.0 --port 8000
```

---

## Deployment Guide

### Production Checklist

- [ ] Use PostgreSQL database
- [ ] Enable rate limiting
- [ ] Configure SSL/TLS
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Enable request logging
- [ ] Set resource limits
- [ ] Test failover scenarios

### Docker Production

```yaml
version: '3.8'

services:
  intent-engine:
    image: anony45/intent-engine-api:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/intent_engine
      - API_WORKERS=4
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 4G
          cpus: '2'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Support

For issues and questions:

1. Check this troubleshooting guide
2. Review logs: `docker-compose logs`
3. Open issue on GitHub

---

**Last Updated:** 2026-02-17  
**Version:** 2.0.0
