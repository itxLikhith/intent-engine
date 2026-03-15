# 🚀 Intent Engine - Quick Reference Card

## One-Command Start

**Windows:**
```powershell
.\scripts\start_production.ps1 start
```

**Linux/Mac:**
```bash
./scripts/start_production.sh start
```

## Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Python API** | http://localhost:8000 | Main API |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **Go Search** | http://localhost:8081 | Crawler search |
| **SearXNG** | http://localhost:8080 | Fallback search |
| **Grafana** | http://localhost:3000 | Dashboards (admin/admin) |
| **Prometheus** | http://localhost:9090 | Metrics |

## Quick Test

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"best laptop for programming","limit":10}'
```

## Common Commands

```bash
# Status
docker compose -f docker-compose.prod.yml ps

# Logs
docker compose -f docker-compose.prod.yml logs -f

# Stop
docker compose -f docker-compose.prod.yml down

# Restart
docker compose -f docker-compose.prod.yml restart

# Verify
./scripts/verify_production_setup.sh
```

## Monitor Crawler

```bash
# Crawled pages count
docker exec intent-engine-postgres psql -U intent_user -d intent_engine \
  -c "SELECT COUNT(*) FROM crawled_pages;"

# Queue size
docker exec intent-redis redis-cli ZCARD crawl_queue

# Crawler logs
docker logs -f intent-go-crawler
```

## Troubleshooting

```bash
# Check health
curl http://localhost:8000/health
curl http://localhost:8081/health
curl http://localhost:8080/healthz

# View all logs
docker compose -f docker-compose.prod.yml logs -f

# Reset everything (WARNING: deletes data)
docker compose -f docker-compose.prod.yml down -v
```

## Add Seed URLs

```bash
curl -X POST http://localhost:8081/api/v1/crawl/seed \
  -H "Content-Type: application/json" \
  -d '{"urls":["https://example.com"],"priority":5}'
```

---

**Full Documentation:** [PRODUCTION_SETUP_COMPLETE.md](PRODUCTION_SETUP_COMPLETE.md)
