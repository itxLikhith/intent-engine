# Quick Start Guide - Go Crawler & Indexer

This guide will help you get the Go crawler and indexer up and running in minutes.

## Prerequisites

Before you begin, ensure you have:

- **Go 1.21+** - [Download Go](https://golang.org/dl/)
- **Redis 7+** - [Download Redis](https://redis.io/download/)
- **PostgreSQL 15+** - [Download PostgreSQL](https://www.postgresql.org/download/)
- **Docker & Docker Compose** (optional, for easy deployment)

## Option 1: Quick Start with Docker (Recommended)

### 1. Start All Services

```bash
cd go-crawler

# Start all services (crawler, indexer, API, Redis, PostgreSQL, monitoring)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 2. Verify Services

```bash
# Health check
curl http://localhost:8080/health

# View metrics
curl http://localhost:8080/metrics

# Check crawler status
curl http://localhost:8080/api/v1/crawl/status
```

### 3. Add Seed URLs

```bash
curl -X POST http://localhost:8080/api/v1/crawl/seed \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com",
      "https://golang.org",
      "https://go.dev"
    ],
    "priority": 1,
    "depth": 0
  }'
```

### 4. Search Indexed Content

```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Go programming",
    "limit": 20
  }' | jq
```

### 5. Access Monitoring Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

---

## Option 2: Local Development Setup

### 1. Install Dependencies

```bash
cd go-crawler

# Download Go modules
go mod download

# Verify installation
go version
```

### 2. Start Redis

```bash
# Using Docker
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Or install locally and run
redis-server
```

### 3. Start PostgreSQL

```bash
# Using Docker
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=crawler \
  -e POSTGRES_PASSWORD=crawler \
  -e POSTGRES_DB=intent_engine \
  --name postgres postgres:15-alpine

# Or install locally and configure
```

### 4. Configure Environment

Create `config.yaml` from the example:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` and update:

```yaml
redis:
  addr: "localhost:6379"

postgres:
  dsn: "postgresql://crawler:crawler@localhost:5432/intent_engine"

storage:
  badger_path: "./data/badger"
  bleve_path: "./data/bleve"
```

### 5. Build Binaries

```bash
# Build all services
make build

# Or build individually
go build -o bin/crawler ./cmd/crawler
go build -o bin/indexer ./cmd/indexer
go build -o bin/search-api ./cmd/search-api
go build -o bin/pagerank ./cmd/pagerank
```

### 6. Run Services

Open separate terminals for each service:

**Terminal 1 - Crawler:**
```bash
make run-crawler
# Or: ./bin/crawler -seed "https://example.com,https://golang.org"
```

**Terminal 2 - Indexer:**
```bash
make run-indexer
# Or: ./bin/indexer -bleve-path ./data/bleve
```

**Terminal 3 - Search API:**
```bash
make run-api
# Or: ./bin/search-api
```

**Terminal 4 - PageRank (optional):**
```bash
make run-pagerank
# Or: ./bin/pagerank -redis localhost:6379
```

### 7. Test the System

```bash
# Health check
curl http://localhost:8080/health

# Add seed URLs
curl -X POST http://localhost:8080/api/v1/crawl/seed \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com"], "priority": 1}'

# Search
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "example", "limit": 10}'
```

---

## Option 3: Minimal Setup (Testing Only)

For quick testing without Redis/PostgreSQL:

```bash
# Just run the API server with in-memory index
go run ./cmd/search-api

# Test search (will be empty initially)
curl http://localhost:8080/health
```

---

## Common Commands

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run benchmarks
make bench
```

### Code Quality

```bash
# Format code
make fmt

# Lint code
make lint
```

### Cleanup

```bash
# Clean build artifacts
make clean

# Stop Docker services
docker-compose down

# Remove all data
docker-compose down -v
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8080
lsof -i :8080

# Kill the process
kill -9 <PID>
```

### Redis Connection Failed

```bash
# Check if Redis is running
redis-cli ping

# Should return: PONG
```

### Build Errors

```bash
# Clean module cache
go clean -modcache

# Re-download dependencies
go mod download
```

### Index Corruption

```bash
# Remove index and rebuild
rm -rf ./data/bleve
make run-indexer
```

---

## Next Steps

1. **Configure Crawling** - Edit `config.yaml` to customize crawling behavior
2. **Add More Seeds** - Submit more URLs via the API
3. **Monitor Performance** - Check Grafana dashboards
4. **Scale Up** - Deploy multiple crawler instances
5. **Integrate with Intent Engine** - Connect to Python API

---

## API Reference

### Search
```bash
POST /api/v1/search
Content-Type: application/json

{
  "query": "your search query",
  "limit": 20
}
```

### Add Seed URLs
```bash
POST /api/v1/crawl/seed
Content-Type: application/json

{
  "urls": ["https://example.com"],
  "priority": 1,
  "depth": 0
}
```

### Get Crawl Status
```bash
GET /api/v1/crawl/status
```

### Get Statistics
```bash
GET /api/v1/stats
```

### Health Check
```bash
GET /health
```

### Metrics
```bash
GET /metrics
```

---

## Architecture Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Crawler   │───▶│   Indexer   │───▶│ Search API  │
│   (Colly)   │    │   (Bleve)   │    │   (HTTP)    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Redis    │    │  BadgerDB   │    │ Prometheus  │
│   (Queue)   │    │  (Content)  │    │  (Metrics)  │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

For more detailed information, see [GO_CRAWLER_INDEXER_PLAN.md](../GO_CRAWLER_INDEXER_PLAN.md).
