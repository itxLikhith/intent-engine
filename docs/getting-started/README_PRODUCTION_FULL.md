# Intent Engine - Production Deployment Guide

## 🚀 Complete Production-Ready Setup

This guide covers the complete production deployment of the Intent Engine with:
- **Python Intent Engine API** (FastAPI) - Port 8000
- **Go Crawler** (Colly-based) - Autonomous web crawling
- **Go Indexer** (Bleve-based) - Full-text search indexing
- **Go Search API** - Fast search endpoint (Port 8081)
- **SearXNG** - Fallback search engine (Port 8080)
- **PostgreSQL** - Primary database (Port 5432)
- **Redis/Valkey** - Caching & queue (Port 6379)
- **Prometheus** - Metrics collection (Port 9090)
- **Grafana** - Dashboards (Port 3000)

---

## 📋 Prerequisites

- **Docker** 20.10+
- **Docker Compose** 2.0+
- **4GB+ RAM** (8GB recommended)
- **10GB+ free disk space**
- **Ports available**: 5432, 6379, 8000, 8080, 8081, 9090, 3000

---

## 🎯 Quick Start (5 minutes)

### Step 1: Clone & Configure

```bash
cd /path/to/intent-engine

# Copy environment template
cp .env.example .env

# Generate secure secrets (Linux/Mac)
echo "POSTGRES_PASSWORD=$(openssl rand -hex 32)" >> .env
echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env
echo "SEARXNG_SECRET_KEY=$(openssl rand -hex 16)" >> .env
```

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
$postgresPwd = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | ForEach-Object {[char]$_})
$secretKey = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | ForEach-Object {[char]$_})
$searxngKey = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})
Add-Content .env "POSTGRES_PASSWORD=$postgresPwd"
Add-Content .env "SECRET_KEY=$secretKey"
Add-Content .env "SEARXNG_SECRET_KEY=$searxngKey"
```

### Step 2: Start all services

```bash
# Start everything (including monitoring)
docker compose -f docker-compose.prod.yml up -d

# Or start without monitoring
docker compose -f docker-compose.prod.yml --profile monitoring up -d
```

### Step 3: Wait for initialization

```bash
# Wait for databases (60 seconds recommended)
sleep 60

# Check service health
docker compose -f docker-compose.prod.yml ps
```

### Step 4: Verify setup

```bash
# Test Python API health
curl http://localhost:8000/health

# Test Go Search API health
curl http://localhost:8081/health

# Test SearXNG health
curl http://localhost:8080/healthz

# Check crawler is running
docker logs intent-go-crawler
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Intent Engine System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Requests:                                                 │
│  ┌──────────────┐                                              │
│  │ Port 8000    │ Python Intent Engine API                     │
│  │ (FastAPI)    │ - Intent extraction                          │
│  └──────────────┘ - Ranking                                    │
│         │         - Unified search                             │
│         ▼                                                      │
│  ┌──────────────────────────────────────────┐                 │
│  │          Search Layer                     │                 │
│  │  ┌────────────┐    ┌────────────┐        │                 │
│  │  │ Go Search  │    │  SearXNG   │        │                 │
│  │  │ API :8081  │    │  :8080     │        │                 │
│  │  │ (Bleve)    │    │ (Fallback) │        │                 │
│  │  └────────────┘    └────────────┘        │                 │
│  └──────────────────────────────────────────┘                 │
│         ▲                                                      │
│  ┌──────┴──────────────────────────────────────┐              │
│  │         Content Discovery                    │              │
│  │  ┌────────────┐    ┌────────────┐          │              │
│  │  │ Go Crawler │───▶│ Go Indexer │          │              │
│  │  │ (Colly)    │    │ (Bleve)    │          │              │
│  │  └────────────┘    └────────────┘          │              │
│  └──────────────────────────────────────────┘                 │
│                                                                 │
│  Shared Infrastructure:                                         │
│  ┌──────────────┐     ┌──────────────┐                        │
│  │  PostgreSQL  │     │ Redis/Valkey │                        │
│  │   :5432      │     │    :6379     │                        │
│  │  (Metadata)  │     │  (Queue)     │                        │
│  └──────────────┘     └──────────────┘                        │
│                                                                 │
│  Monitoring (Optional Profile):                                 │
│  ┌──────────────┐     ┌──────────────┐                        │
│  │ Prometheus   │     │   Grafana    │                        │
│  │   :9090      │     │    :3000     │                        │
│  └──────────────┘     └──────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Database
POSTGRES_PASSWORD=your_secure_password_here

# API Security
SECRET_KEY=your_secure_random_string_here

# SearXNG
SEARXNG_SECRET_KEY=your_searxng_secret_here

# Grafana (optional)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change_this_password
```

### Crawler Configuration

Edit the crawler command in `docker-compose.prod.yml`:

```yaml
go-crawler:
  command: >
    ./crawler
    -seed=https://go.dev,https://golang.org
    -max-pages=500          # Adjust based on needs
    -max-depth=3            # Crawl depth (1-5 recommended)
    -concurrency=5          # Parallel requests
    -delay=1000             # Delay between requests (ms)
```

### Resource Limits

Default resource allocation:
- **PostgreSQL**: 2GB RAM
- **Intent Engine API**: 2GB RAM
- **Go Search API**: 1GB RAM
- **Redis**: 512MB RAM
- **SearXNG**: 1GB RAM

Adjust in `docker-compose.prod.yml` under `deploy.resources`.

---

## 📊 Services & Ports

| Service | Container Name | Port | Health Check |
|---------|---------------|------|--------------|
| **Python API** | intent-engine-api | 8000 | `/health` |
| **Go Search API** | intent-go-search-api | 8081 | `/health` |
| **SearXNG** | searxng | 8080 | `/healthz` |
| **PostgreSQL** | intent-engine-postgres | 5432 | `pg_isready` |
| **Redis** | intent-redis | 6379 | `redis-cli ping` |
| **Prometheus** | intent-prometheus | 9090 | `/-/healthy` |
| **Grafana** | intent-grafana | 3000 | `/api/health` |

---

## 🧪 Testing & Verification

### Health Checks

```bash
# Python Intent Engine
curl http://localhost:8000/health

# Go Search API
curl http://localhost:8081/health

# Detailed health with component status
curl http://localhost:8000/health | jq

# SearXNG
curl http://localhost:8080/healthz
```

### Test Search Functionality

```bash
# Test Python API search (uses SearXNG + Go Search)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"best laptop for programming","limit":10}'

# Test Go Search API directly
curl -X POST http://localhost:8081/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"golang","limit":10}'

# Test SearXNG directly
curl "http://localhost:8080/search?q=programming&format=json"
```

### Check Crawler Status

```bash
# View crawler logs
docker logs -f intent-go-crawler

# Check indexer logs
docker logs -f intent-go-indexer

# Check crawled pages count
docker exec intent-engine-postgres psql -U intent_user -d intent_engine \
  -c "SELECT COUNT(*) FROM crawled_pages;"

# Check indexing status
docker exec intent-engine-postgres psql -U intent_user -d intent_engine \
  -c "SELECT is_indexed, COUNT(*) FROM crawled_pages GROUP BY is_indexed;"

# Check Redis queue size
docker exec intent-redis redis-cli ZCARD crawl_queue
```

### Database Verification

```bash
# Connect to PostgreSQL
docker exec -it intent-engine-postgres psql -U intent_user -d intent_engine

# Check tables
\dt

# View recent crawled pages
SELECT id, url, title, crawled_at 
FROM crawled_pages 
ORDER BY crawled_at DESC 
LIMIT 10;

# Check link graph
SELECT COUNT(*) FROM page_links;
```

---

## 📈 Monitoring

### Prometheus Metrics

Access Prometheus UI: http://localhost:9090

**Key Metrics:**
- `crawler_pages_total` - Total pages crawled
- `search_requests_total` - Search requests
- `indexer_documents_total` - Indexed documents
- `http_request_duration_seconds` - Request latency

### Grafana Dashboards

Access Grafana: http://localhost:3000
- **Username**: admin
- **Password**: (from .env or default: admin)

**Pre-configured Dashboards:**
- System Overview
- Crawler Performance
- Search API Metrics
- Database Health

### View Logs

```bash
# All services
docker compose -f docker-compose.prod.yml logs -f

# Specific service
docker compose -f docker-compose.prod.yml logs -f intent-go-crawler

# Last 100 lines
docker compose -f docker-compose.prod.yml logs --tail=100 intent-engine-api
```

---

## 🛠️ Common Operations

### Start/Stop Services

```bash
# Start all
docker compose -f docker-compose.prod.yml up -d

# Stop all
docker compose -f docker-compose.prod.yml down

# Restart specific service
docker compose -f docker-compose.prod.yml restart go-crawler

# Stop and remove volumes (WARNING: deletes all data)
docker compose -f docker-compose.prod.yml down -v
```

### Scale Services

```bash
# Scale search API (horizontal scaling)
docker compose -f docker-compose.prod.yml up -d --scale go-search-api=3

# Scale crawler
docker compose -f docker-compose.prod.yml up -d --scale go-crawler=2
```

### Update Services

```bash
# Rebuild and restart
docker compose -f docker-compose.prod.yml up -d --build

# Pull latest images
docker compose -f docker-compose.prod.yml pull

# Recreate containers
docker compose -f docker-compose.prod.yml up -d --force-recreate
```

### Add Seed URLs

```bash
# Via Go Search API
curl -X POST http://localhost:8081/api/v1/crawl/seed \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com",
      "https://blog.example.com"
    ],
    "priority": 5,
    "depth": 0
  }'

# Directly to Redis
docker exec intent-redis redis-cli ZADD crawl_queue 10 \
  '{"url":"https://newsite.com","priority":10,"depth":0}'
```

---

## 🔐 Security Checklist

Before production deployment:

- [ ] Change all default passwords in `.env`
- [ ] Generate secure `SECRET_KEY` (32+ random bytes)
- [ ] Enable SSL/TLS termination (nginx/traefik)
- [ ] Configure firewall rules
- [ ] Enable database SSL connections
- [ ] Set up Redis authentication
- [ ] Review CORS settings
- [ ] Enable rate limiting
- [ ] Set up log aggregation
- [ ] Configure backup strategy
- [ ] Enable monitoring alerts

---

## 🐛 Troubleshooting

### Services Won't Start

```bash
# Check logs
docker compose -f docker-compose.prod.yml logs postgres
docker compose -f docker-compose.prod.yml logs intent-engine-api

# Check resource usage
docker stats

# Restart services
docker compose -f docker-compose.prod.yml restart
```

### Database Connection Errors

```bash
# Check PostgreSQL health
docker exec intent-engine-postgres pg_isready -U intent_user

# Test connection
docker exec intent-engine-postgres psql -U intent_user -d intent_engine -c "SELECT 1"

# Check migrations ran
docker compose -f docker-compose.prod.yml logs migrations
```

### Crawler Not Crawling

```bash
# Check if queue has URLs
docker exec intent-redis redis-cli ZCARD crawl_queue

# View queued URLs
docker exec intent-redis redis-cli ZRANGE crawl_queue 0 10

# Check crawler logs for errors
docker logs -f intent-go-crawler

# Manually add seed URL
docker exec intent-redis redis-cli ZADD crawl_queue 10 \
  '{"url":"https://go.dev","priority":10,"depth":0}'
```

### Search Returns Empty Results

1. **Wait for crawler** - Check if pages are crawled:
   ```bash
   docker exec intent-engine-postgres psql -U intent_user -d intent_engine \
     -c "SELECT COUNT(*) FROM crawled_pages;"
   ```

2. **Check indexing status**:
   ```bash
   docker exec intent-engine-postgres psql -U intent_user -d intent_engine \
     -c "SELECT is_indexed, COUNT(*) FROM crawled_pages GROUP BY is_indexed;"
   ```

3. **Restart indexer**:
   ```bash
   docker restart intent-go-indexer
   ```

### High Memory Usage

```bash
# Check memory usage
docker stats

# Reduce PostgreSQL memory
# Edit docker-compose.prod.yml:
#   deploy:
#     resources:
#       limits:
#         memory: 1G  # Reduce from 2G

# Restart with reduced resources
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d
```

### Port Conflicts

If a port is already in use:

```bash
# Find process using port (Linux/Mac)
lsof -i :8000

# Windows PowerShell
netstat -ano | findstr :8000

# Change port in docker-compose.prod.yml
# Example: Change API port from 8000 to 8001
intent-engine-api:
  ports:
    - "8001:8000"  # Host:Container
```

### Complete Reset

```bash
# Stop and remove everything including volumes
docker compose -f docker-compose.prod.yml down -v

# Remove dangling images
docker image prune -f

# Start fresh
docker compose -f docker-compose.prod.yml up -d
```

---

## 📊 Performance Tuning

### Crawler Optimization

```yaml
# Increase concurrency (if you have bandwidth)
go-crawler:
  command: >
    ./crawler
    -concurrency=10
    -max-pages=1000
    -delay=500  # Reduce delay for faster crawling
```

### Database Optimization

```yaml
# Enable PgBouncer for connection pooling
pgbouncer:
  profiles: ["pgbouncer"]
```

### Search API Optimization

```yaml
# Scale search API horizontally
docker compose -f docker-compose.prod.yml up -d --scale go-search-api=3
```

---

## 🎓 API Reference

### Python Intent Engine API (Port 8000)

**Interactive Docs**: http://localhost:8000/docs

**Key Endpoints:**
```bash
# Health check
GET /health

# Unified search
POST /search
{
  "query": "your query",
  "limit": 10
}

# Intent extraction
POST /extract-intent
{
  "text": "I want to buy a laptop"
}

# Analytics
GET /analytics/summary
```

### Go Search API (Port 8081)

**Key Endpoints:**
```bash
# Health check
GET /health

# Statistics
GET /stats

# Search
POST /api/v1/search
{
  "query": "golang",
  "limit": 10,
  "filters": {}
}

# Add seed URLs
POST /api/v1/crawl/seed
{
  "urls": ["https://example.com"],
  "priority": 5,
  "depth": 0
}

# Crawl status
GET /api/v1/crawl/status

# Prometheus metrics
GET /metrics
```

---

## 📁 Data Persistence

All data is persisted in Docker volumes:

- `postgres_data` - PostgreSQL database
- `valkey-data` - Redis/Valkey data
- `go-api-data` - Bleve index & BadgerDB
- `go-crawler-data` - Crawler storage
- `go-indexer-data` - Indexer data
- `prometheus_data` - Prometheus metrics
- `grafana_data` - Grafana dashboards

**Backup volumes:**
```bash
# Backup PostgreSQL
docker run --rm \
  -v intent-engine_postgres_data:/data/postgres \
  -v $(pwd):/backup \
  postgres:15-alpine \
  pg_dump -h postgres -U intent_user intent_engine > backup.sql

# Backup entire volume
docker run --rm \
  -v intent-engine_postgres_data:/data \
  -v $(pwd):/backup \
  alpine \
  tar czf /backup/postgres_backup.tar.gz /data
```

---

## 🚀 Production Deployment

### Before Production

1. **Security**
   - [ ] Change all passwords
   - [ ] Enable SSL/TLS
   - [ ] Configure firewall
   - [ ] Review CORS settings

2. **Monitoring**
   - [ ] Set up alerts
   - [ ] Configure log aggregation
   - [ ] Set up uptime monitoring

3. **Backup**
   - [ ] Configure automated backups
   - [ ] Test restore procedure
   - [ ] Set up replication

4. **Performance**
   - [ ] Load test the system
   - [ ] Tune resource limits
   - [ ] Configure CDN (if needed)

### Docker Swarm Deployment

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml intent-engine

# Check status
docker stack ps intent-engine

# Scale service
docker service scale intent-engine_go-search-api=3
```

### Kubernetes

For Kubernetes deployment, see `go-crawler/deployments/k8s/`.

---

## 📞 Support & Resources

### Documentation
- [Main README](README.md)
- [Quick Start](QUICKSTART.md)
- [Go Crawler Guide](go-crawler/README.md)
- [API Docs](http://localhost:8000/docs)

### Logs & Debugging
```bash
# View all logs
docker compose -f docker-compose.prod.yml logs -f

# Export logs
docker compose -f docker-compose.prod.yml logs > logs.txt
```

### Health Dashboard
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

---

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] Review security settings
- [ ] Configure environment variables
- [ ] Set up monitoring
- [ ] Configure backups

### Deployment
- [ ] Run `docker compose -f docker-compose.prod.yml up -d`
- [ ] Wait for all services to be healthy
- [ ] Verify all health checks pass
- [ ] Test search functionality

### Post-Deployment
- [ ] Monitor logs for errors
- [ ] Check resource usage
- [ ] Verify crawler is running
- [ ] Test API endpoints
- [ ] Set up alerts

### Ongoing Maintenance
- [ ] Monitor disk usage
- [ ] Review logs regularly
- [ ] Update services periodically
- [ ] Backup data regularly

---

**Version:** 1.0.0
**Last Updated:** 2026-03-14
**Status:** Production Ready ✅
