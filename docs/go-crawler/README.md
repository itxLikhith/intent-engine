# Go Crawler & Indexer - Quick Start Guide

## Overview

Intent-aware web crawler and indexer for the Intent Engine project. Crawls web pages, extracts intent signals, and indexes them for intent-aligned search.

## Quick Start

### Prerequisites

- Docker & Docker Compose installed
- At least 2GB free disk space
- Port 8081 available (search API)

### Step 1: Start All Services

```bash
cd go-crawler
docker-compose up -d
```

This starts:
- **PostgreSQL** (port 5432) - Page storage
- **Redis** (port 6379) - URL queue management
- **Crawler** - Web crawler service
- **Search API** (port 8081) - Intent-aware search endpoint

### Step 2: Verify Services

```bash
# Check all containers are running
docker-compose ps

# Check search API health
curl http://localhost:8081/health

# Check crawler logs
docker-compose logs -f crawler
```

### Step 3: Test Search API

```bash
# Search (will be empty until crawler indexes content)
curl -X POST http://localhost:8081/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "golang", "limit": 10}'

# View statistics
curl http://localhost:8081/stats
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Crawler   │────▶│    Redis     │────▶│  PostgreSQL │
│  (Colly)    │     │   (Queue)    │     │   (Pages)   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Search API   │
                    │   (Bleve)    │
                    └──────────────┘
```

## Configuration

Edit `docker-compose.yml` to customize:

| Service | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| crawler | `-seed` | `https://go.dev` | Seed URLs (comma-separated) |
| crawler | `-max-pages` | `50` | Maximum pages to crawl |
| crawler | `-max-depth` | `2` | Maximum crawl depth |
| crawler | `-concurrency` | `3` | Concurrent requests |
| search-api | `BLEVE_PATH` | `/data/bleve` | Bleve index path |
| search-api | `POSTGRES_DSN` | `postgresql://...` | Database connection |

## Monitoring

### Check Crawler Progress

```bash
# View live crawler logs
docker-compose logs -f crawler

# Check queue size
docker exec intent-redis redis-cli ZCARD crawl_queue

# Check crawled pages count
docker exec intent-postgres psql -U crawler -d intent_engine \
  -c "SELECT COUNT(*) FROM crawled_pages;"
```

### Check Database Stats

```bash
docker exec intent-postgres psql -U crawler -d intent_engine \
  -c "SELECT * FROM crawl_stats;"
```

## Troubleshooting

### Crawler not starting

```bash
# Check logs for errors
docker-compose logs crawler

# Restart crawler
docker-compose restart crawler
```

### Database connection issues

```bash
# Check PostgreSQL is healthy
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres
```

### Reset everything (WARNING: deletes all data)

```bash
docker-compose down -v
docker-compose up -d
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Indexer statistics |
| `/api/v1/search` | POST | Intent-aware search |

### Search Request Format

```json
{
  "query": "golang microservices",
  "limit": 10,
  "filters": {
    "min_depth": 0,
    "max_depth": 2
  }
}
```

## Data Persistence

All data is persisted in Docker volumes:

- `postgres_data` - PostgreSQL database
- `redis_data` - Redis queue (AOF persistence)
- `api_data` - Bleve index & BadgerDB HTML storage

## Next Steps

1. Wait for crawler to index content (check `/stats`)
2. Test search queries via API
3. Add more seed URLs via Redis:
   ```bash
   docker exec intent-redis redis-cli ZADD crawl_queue 10 '{"url":"https://example.com","priority":10,"depth":0}'
   ```

## License

Same as Intent Engine project
