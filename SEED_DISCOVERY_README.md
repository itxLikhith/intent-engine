# Dynamic Seed URL Discovery & Topic Expansion

## Overview

The Intent Engine now features **automatic seed URL discovery** and **dynamic topic expansion** to continuously expand the Go crawler's coverage without manual intervention.

## Features

### 1. Automatic Seed URL Discovery
- Discovers relevant URLs using SearXNG meta-search
- Quality scoring based on search ranking and domain authority
- Automatic injection into crawler queue via Redis
- Domain deduplication to avoid redundant crawling

### 2. Dynamic Topic Expansion
- Learns from user search queries
- Automatically expands topic categories
- Discovers new topic categories from trending searches
- Persists learned topics to Redis

### 3. Scheduled Discovery
- **URL Discovery**: Runs daily at 3 AM (low-traffic time)
- **Topic Expansion**: Runs every 6 hours
- Automatic retry on failure
- Statistics tracking

### 4. Self-Learning System
The system learns and expands in three ways:

```
User Searches ──▶ Query Recording ──▶ Keyword Extraction
                                              │
                                              ▼
Trending Analysis ◀── Keyword Frequency ◀── Storage (Redis)
                                              │
                                              ▼
Topic Expansion ◀── Pattern Matching ◀── Category Detection
                                              │
                                              ▼
New Seed URLs ◀── SearXNG Search ◀── Expanded Topics
                                              │
                                              ▼
Crawler Queue ──▶ Go Crawler ──▶ Indexed Content
```

## API Endpoints

### Discovery Control

#### `POST /seed-discovery/run`
Manually trigger seed URL discovery.

```bash
curl -X POST http://localhost:8000/seed-discovery/run
```

Response:
```json
{
  "status": "success",
  "message": "Seed discovery completed",
  "results": {
    "go_urls_discovered": 28,
    "prog_urls_discovered": 19,
    "queue_size": 701318,
    "visited_count": 0
  }
}
```

#### `GET /seed-discovery/status`
Get discovery system status and statistics.

```bash
curl http://localhost:8000/seed-discovery/status
```

### Topic Management

#### `GET /seed-discovery/topics`
List all discovery topics (including learned ones).

```bash
curl http://localhost:8000/seed-discovery/topics
```

Response:
```json
{
  "topics": {
    "programming": [
      "programming language tutorials",
      "rust tutorial",
      "rust examples",
      ...
    ],
    "go_language": [...],
    ...
  },
  "stats": {
    "total_categories": 5,
    "total_topics": 31,
    "queries_analyzed": 1
  }
}
```

#### `POST /seed-discovery/topics/expand`
Manually trigger topic expansion based on trending queries.

```bash
curl -X POST http://localhost:8000/seed-discovery/topics/expand
```

#### `DELETE /seed-discovery/topics/reset`
Reset topics to defaults (clear learned topics).

```bash
curl -X DELETE http://localhost:8000/seed-discovery/topics/reset
```

## Default Topic Categories

| Category | Description | Example Topics |
|----------|-------------|----------------|
| `programming` | General programming | "programming language tutorials", "software development best practices" |
| `go_language` | Go-specific | "Go programming language tutorial", "Go concurrency patterns" |
| `python` | Python programming | "Python web development Django Flask", "Python machine learning examples" |
| `web_dev` | Web development | "JavaScript frameworks React Vue", "backend development Node.js" |
| `devops` | DevOps & infra | "Docker Kubernetes guide", "CI/CD pipeline setup" |

## How It Works

### 1. Query Recording (Automatic)
Every user search query is automatically recorded:
```python
# In unified_search.py - search() method
asyncio.create_task(expander.add_search_query(request.query))
```

### 2. Keyword Extraction
Keywords are extracted from queries:
- Stop words removed (the, a, tutorial, etc.)
- Meaningful terms kept (rust, programming, kubernetes)
- Frequency tracked in Redis

### 3. Topic Categorization
Keywords are categorized using pattern matching:
```python
"go" or "golang" → go_language
"python" or "django" → python
"docker" or "kubernetes" → devops
```

### 4. Topic Expansion
New topics are generated from trending keywords:
```
Keyword: "rust"
↓
New Topics:
- "rust tutorial"
- "rust examples"
- "rust best practices"
```

### 5. URL Discovery
For each topic, SearXNG is queried:
```python
response = await client.search(
    query="rust tutorial",
    categories=["general"],
    pageno=1
)
```

### 6. URL Injection
Discovered URLs are injected into crawler queue:
```python
queue_item = {
    "id": "seed_xxxxx",
    "url": "https://doc.rust-lang.org/book/",
    "priority": 8,
    "depth": 0,
    "source": "seed_discovery",
    "topic": "go_language"
}
await redis.zadd("crawl_queue", {json.dumps(item): priority})
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEED_DISCOVERY_INTERVAL_HOURS` | 24 | Hours between URL discovery runs |
| `TOPIC_EXPANSION_INTERVAL_HOURS` | 6 | Hours between topic expansion runs |
| `MAX_URLS_PER_RUN` | 50 | Maximum URLs to discover per run |
| `REDIS_URL` | `redis://redis:6379` | Redis connection URL |

### Scheduler Configuration

In `scheduled_seed_discovery.py`:
```python
ScheduledSeedDiscovery(
    discovery_interval_hours=24,      # Daily URL discovery
    topic_expansion_interval_hours=6, # 6-hour topic expansion
    max_urls_per_run=50               # Max 50 URLs per run
)
```

## Monitoring

### Check System Health
```bash
# Check scheduler status
curl http://localhost:8000/seed-discovery/status

# Check current topics
curl http://localhost:8000/seed-discovery/topics

# Check crawler queue size
docker exec intent-go-crawler redis-cli ZCARD crawl_queue
```

### View Logs
```bash
# API logs (discovery activity)
docker logs intent-engine-intent-engine-api-1 | grep -i "seed\|discovery\|topic"

# Crawler logs (URL processing)
docker logs intent-go-crawler | grep -i "visiting\|seed"
```

## Redis Keys

| Key | Type | Description |
|-----|------|-------------|
| `seed_discovery:topics` | String | JSON of all topics |
| `seed_discovery:query_history` | Sorted Set | User queries with frequency |
| `seed_discovery:suggestions:keywords` | Sorted Set | Extracted keywords |
| `crawl_queue` | Sorted Set | Crawler URL queue |
| `visited_urls` | Set | Already crawled URLs |

## Manual Operations

### Add Custom Topic
```python
from searxng.topic_expander import get_topic_expander

expander = get_topic_expander()
await expander.connect()

# Add to existing category
expander.default_topics["programming"].append("my custom topic")
await expander._persist_topics()

# Add new category
expander.default_topics["rust"] = [
    "rust tutorial",
    "rust programming",
    "rust web assembly"
]
await expander._persist_topics()
```

### Force URL Discovery for Specific Topic
```python
from searxng.seed_discovery import SeedURLDiscovery

discovery = SeedURLDiscovery()
urls = await discovery.discover_urls_for_topic(
    topic="rust web assembly",
    max_urls=20
)

from searxng.seed_injector import SeedURLInjector
injector = SeedURLInjector()
await injector.inject_seed_urls(urls, priority=10)
```

## Troubleshooting

### No URLs Being Discovered
1. Check SearXNG connectivity:
```bash
docker exec intent-engine-intent-engine-api-1 curl http://searxng:8080/search?q=test&format=json
```

2. Check logs for errors:
```bash
docker logs intent-engine-intent-engine-api-1 | grep "seed_discovery"
```

### Topics Not Expanding
1. Check if queries are being recorded:
```bash
curl http://localhost:8000/seed-discovery/status
# Check queries_analyzed count
```

2. Manually trigger expansion:
```bash
curl -X POST http://localhost:8000/seed-discovery/topics/expand
```

### Crawler Not Processing URLs
1. Check queue size:
```bash
docker exec intent-engine-intent-engine-api-1 python -c "
from searxng.seed_injector import SeedURLInjector
import asyncio
async def check():
    inj = SeedURLInjector()
    await inj.connect()
    stats = await inj.get_queue_stats()
    print(stats)
asyncio.run(check())
"
```

2. Restart crawler:
```bash
docker-compose restart go-crawler
```

## Future Enhancements

- [ ] ML-based topic categorization
- [ ] User feedback integration for topic quality
- [ ] Trending topic alerts
- [ ] Automatic category merging
- [ ] Multi-language support for topics
- [ ] Integration with external knowledge graphs
