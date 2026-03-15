# Intent Engine - Go Crawler & Indexer Production Setup

## Quick Start (60 seconds)

### Start All Services

```bash
cd go-crawler
docker-compose up -d
```

This starts:
- ✅ **Search API** (port 8081) - Bleve-based search
- ✅ **Crawler** - Web crawler (Colly)
- ✅ **Indexer** - Document indexer (Bleve)
- ✅ **Redis** - URL queue
- ✅ **PostgreSQL** - Page storage
- ✅ **SearXNG** - Fallback search (port 8082)

### Verify Services

```bash
# Wait for services to start
sleep 30

# Check all containers
docker-compose ps

# Test Search API health
curl http://localhost:8081/health

# Test Search API
curl -X POST http://localhost:8081/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "golang", "limit": 10}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Intent Engine Search                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Crawler  │───▶│ Indexer  │───▶│ Search   │              │
│  │  (Go)    │    │  (Go)    │    │ API(Go)  │              │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│       │               │               │                     │
│       ▼               ▼               ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Redis   │    │  Bleve   │    │ Python   │              │
│  │ (Queue)  │    │ (Index)  │    │ Intent   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                     │                       │
│                                     ▼                       │
│                              ┌──────────────┐               │
│                              │  PostgreSQL  │               │
│                              │  (Metadata)  │               │
│                              └──────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## Integration with Python Intent Engine

The Go crawler integrates seamlessly with your existing Python Intent Engine:

### 1. Search from Python

```python
import requests

# Search via Go Search API
response = requests.post(
    "http://localhost:8081/api/v1/search",
    json={"query": "golang microservices", "limit": 10}
)
results = response.json()

print(f"Found {results['total_results']} results")
for result in results['results']:
    print(f"- {result['title']} ({result['url']})")
```

### 2. Unified Search Endpoint

The Go Search API can be called from your Python Intent Engine's unified search:

```python
# In your Python Intent Engine
async def search_with_go_backend(query: str, limit: int = 10):
    """Search using Go crawler index"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8081/api/v1/search",
            json={"query": query, "limit": limit}
        ) as resp:
            return await resp.json()
```

## API Endpoints

### Search API (Port 8081)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Index statistics |
| `/api/v1/search` | POST | Search query |
| `/metrics` | GET | Prometheus metrics |

### Search Request

```json
{
  "query": "golang microservices",
  "limit": 10,
  "filters": {
    "min_depth": 0,
    "max_depth": 5
  }
}
```

### Search Response

```json
{
  "query": "golang microservices",
  "results": [
    {
      "url": "https://go.dev",
      "title": "The Go Programming Language",
      "content": "An open-source programming language...",
      "score": 0.95,
      "rank": 1
    }
  ],
  "total_results": 42,
  "processing_time_ms": 15.3,
  "engines_used": ["bleve"],
  "ranking_applied": true
}
```

## Configuration

### Environment Variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `REDIS_ADDR` | All | `redis:6379` | Redis address |
| `POSTGRES_DSN` | All | `postgresql://...` | PostgreSQL DSN |
| `BLEVE_PATH` | Indexer/API | `/data/bleve` | Bleve index path |
| `BADGER_PATH` | Crawler/API | `/data/badger` | BadgerDB path |
| `SERVER_PORT` | API | `8080` | API server port |

### Crawler Command-Line Options

```bash
./crawler \
  -redis=redis:6379 \
  -postgres=postgresql://crawler:crawler@postgres:5432/intent_engine \
  -badger=/data/badger \
  -seed=https://go.dev,https://golang.org \
  -max-pages=500 \
  -max-depth=3 \
  -concurrency=5 \
  -delay=1000
```

## Monitoring

### Check Service Health

```bash
# Search API
curl http://localhost:8081/health

# Crawler logs
docker-compose logs -f crawler

# Indexer logs
docker-compose logs -f indexer

# All services
docker-compose ps
```

### Check Statistics

```bash
# Search API stats
curl http://localhost:8081/stats

# Prometheus metrics
curl http://localhost:8081/metrics
```

### Check Database

```bash
# Connect to PostgreSQL
docker exec -it intent-postgres psql -U crawler -d intent_engine

# Check crawled pages
SELECT COUNT(*) FROM crawled_pages;

# Check indexing status
SELECT is_indexed, COUNT(*) FROM crawled_pages GROUP BY is_indexed;

# View recent pages
SELECT id, url, title, crawled_at FROM crawled_pages 
ORDER BY crawled_at DESC LIMIT 10;
```

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose logs

# Restart services
docker-compose restart

# Rebuild containers
docker-compose up -d --build
```

### Crawler Not Crawling

```bash
# Check crawler logs
docker-compose logs crawler

# Check Redis queue
docker exec intent-redis redis-cli ZCARD crawl_queue

# Add seed URL manually
docker exec intent-redis redis-cli ZADD crawl_queue 10 '{"url":"https://example.com","priority":10}'
```

### Search Returns No Results

1. Wait for crawler to index content (check `/stats`)
2. Check if pages are indexed:
   ```bash
   docker exec intent-postgres psql -U crawler -d intent_engine \
     -c "SELECT COUNT(*) FROM crawled_pages WHERE is_indexed = true;"
   ```
3. Manually trigger indexing:
   ```bash
   docker-compose restart indexer
   ```

### Reset Everything (WARNING: Deletes All Data)

```bash
docker-compose down -v
docker-compose up -d
```

## Performance Tuning

### Increase Crawler Speed

Edit `docker-compose.yml`:
```yaml
crawler:
  command: >
    ./crawler
    -concurrency=10        # Increase from 5
    -delay=500            # Decrease from 1000
    -max-pages=1000       # Increase limit
```

### Increase Index Batch Size

```yaml
indexer:
  command: >
    ./indexer
    -batch-size=100       # Increase from 50
    -interval=15          # Decrease from 30
```

### Scale Search API

```bash
docker-compose up -d --scale search-api=3
```

## Production Deployment

### Before Production

1. ✅ Change default PostgreSQL password
2. ✅ Configure Redis authentication
3. ✅ Enable SSL/TLS for database connections
4. ✅ Set up monitoring and alerting
5. ✅ Configure log aggregation
6. ✅ Set up backups

### Production Docker Compose

Add these to your production `docker-compose.yml`:

```yaml
services:
  search-api:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  crawler:
    deploy:
      resources:
        limits:
          memory: 2G

  indexer:
    deploy:
      resources:
        limits:
          memory: 2G

  redis:
    command: redis-server --appendonly yes --requirepass your-redis-password

  postgres:
    environment:
      - POSTGRES_PASSWORD=your-secure-password
```

## Development

### Build Locally

```bash
# Build all binaries
go build -o crawler ./cmd/crawler
go build -o indexer ./cmd/indexer
go build -o search-api ./cmd/search-api

# Run locally (need Redis and PostgreSQL running)
./crawler -seed=https://go.dev -max-pages=100
./indexer
./search-api
```

### Run Tests

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific test
go test ./pkg/...
```

## Next Steps

1. **Add More Seed URLs**: Edit crawler command in `docker-compose.yml`
2. **Monitor Crawling Progress**: Check `/stats` endpoint regularly
3. **Integrate with Python Intent Engine**: Use Go Search API as backend
4. **Configure Production Settings**: Security, monitoring, backups
5. **Scale Horizontally**: Add more crawler/indexer instances

## License

Same as Intent Engine project
