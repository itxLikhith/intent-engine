# Go Crawler & Indexer for Intent Engine

**Intent-Aware Search Infrastructure** - NOT a typical search engine

This is a custom **intent-aware crawler and indexer** built in Go for the Intent Engine project. Unlike traditional search engines that index only keywords, this system extracts and indexes **intent signals** (goals, use cases, ethical preferences, complexity levels) for intent-aligned retrieval.

## Key Differentiators from Typical Search Engines

| Traditional Search | Intent-Aware Index |
|-------------------|-------------------|
| Indexes keywords | Indexes **intent signals** (goals, use cases, ethics) |
| Ranks by PageRank + TF-IDF | Ranks by **intent alignment** + quality |
| One-size-fits-all results | **Personalized by intent** (skill level, ethical preferences) |
| No context awareness | **Context-aware** (temporal, complexity, ethical) |
| Tracks users for personalization | **Privacy-first** - no user tracking, intent expires per session |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Intent Engine Search                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Crawler    │    │   Indexer    │    │   Search     │      │
│  │   Service    │───▶│   Service    │───▶│   Service    │      │
│  │    (Go)      │    │    (Go)      │    │    (Go)      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Redis     │    │   Bleve      │    │   Intent     │      │
│  │    (Queue)   │    │   (Index)    │    │  Alignment   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                            │                                    │
│                            ▼                                    │
│                   ┌──────────────────┐                         │
│                   │   PostgreSQL     │                         │
│                   │   (Metadata)     │                         │
│                   └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Python Intent API   │
              │   (Existing System)   │
              └───────────────────────┘
```

## Intent Signals Indexed

This system extracts and indexes the following intent signals from content:

### 1. **Goal Classification**
- `learn`, `comparison`, `troubleshooting`, `purchase`, `local_service`, `navigation`, `find_information`

### 2. **Use Cases**
- `learning`, `comparison`, `troubleshooting`, `verification`, `professional_development`, `market_research`

### 3. **Complexity/Skill Level**
- `beginner`, `intermediate`, `advanced`, `expert`

### 4. **Result Type**
- `answer`, `tutorial`, `tool`, `marketplace`, `community`

### 5. **Ethical Signals**
- `privacy-first`, `open-source`, `sustainable`, `ethical`, `accessible`

### 6. **Topics & Key Phrases**
- Extracted topics and domain-specific phrases

## Project Structure

```
go-crawler/
├── cmd/
│   ├── crawler/           # Crawler service (fetches pages)
│   ├── indexer/           # Indexer service (extracts intent + indexes)
│   └── search-api/        # Search API (intent-aware retrieval)
│
├── pkg/
│   ├── models/            # Data models
│   │   └── models.go
│   ├── intent/            # Intent schema & analyzer (NEW!)
│   │   ├── schema.go      # Intent data structures
│   │   └── analyzer.go    # Intent extraction (rule-based)
│   └── indexer/           # Intent-aware indexer (NEW!)
│       ├── intent_indexer.go  # Bleve indexer with intent
│       └── worker.go      # Background indexing worker
│
├── internal/
│   ├── storage/           # PostgreSQL + BadgerDB storage
│   │   └── storage.go
│   └── frontier/          # URL queue (Redis)
│       └── queue.go
│
├── migrations/
│   └── 001_create_crawler_tables.sql
│
├── config.example.yaml
├── docker-compose.yml
├── Makefile
└── README.md
```

## Quick Start

### Prerequisites

- Go 1.21+
- Redis 7+
- PostgreSQL 15+

### Installation

```bash
cd go-crawler

# Download dependencies
go mod download

# Build all binaries
make build
```

### Running Services

```bash
# Option 1: Run individual services
make run-crawler    # Start crawler
make run-indexer    # Start indexer (extracts intent + indexes)
make run-api        # Start search API

# Option 2: Run with Docker
docker-compose up -d
```

### Configuration

Create `config.yaml`:

```yaml
crawler:
  max_concurrent_requests: 10
  max_depth: 5
  crawl_delay: 1s
  user_agent: "IntentEngineBot/1.0"

redis:
  addr: "localhost:6379"

postgres:
  dsn: "postgresql://crawler:crawler@localhost:5432/crawler?sslmode=disable"

storage:
  badger_path: "./data/badger"
  bleve_path: "./data/bleve"

indexer:
  batch_size: 50
  interval_seconds: 30
```

## API Endpoints

### Intent-Aware Search (NEW!)

This is the **key differentiator** - search with intent alignment.

```bash
curl -X POST http://localhost:8080/api/v1/intent-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how to set up encrypted email",
    "limit": 20,
    "intent": {
      "goal": "learn",
      "useCases": ["learning", "troubleshooting"],
      "skillLevel": "beginner",
      "ethicalSignals": [
        {"dimension": "privacy", " preference": "privacy-first"}
      ]
    }
  }'
```

**Response:**
```json
{
  "query": "how to set up encrypted email",
  "total_hits": 15,
  "results": [
    {
      "page_id": "page_123",
      "url": "https://protonmail.com/setup-guide",
      "title": "How to Set Up Encrypted Email",
      "snippet": "Step-by-step guide for beginners",
      "score": 0.95,
      "final_score": 0.92,
      "intent_metadata": {
        "primary_goal": "learn",
        "use_cases": ["learning", "troubleshooting"],
        "complexity": "simple",
        "result_type": "tutorial",
        "ethical_signals": [
          {"dimension": "privacy", "preference": "privacy-first"}
        ],
        "target_skill_level": "beginner",
        "extraction_confidence": 0.85
      },
      "intent_alignment": {
        "total_score": 0.95,
        "goal_match": 1.0,
        "use_case_match": 1.0,
        "complexity_match": 1.0,
        "ethical_match": 1.0,
        "match_reasons": [
          "matches-learn-intent",
          "use-case-alignment",
          "skill-level-beginner",
          "ethical-alignment"
        ]
      }
    }
  ],
  "processing_time_ms": 45.2
}
```

### Traditional Search (Fallback)

```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "encrypted email setup", "limit": 20}'
```

### Health & Stats

```bash
# Health check
curl http://localhost:8080/health

# Metrics (Prometheus)
curl http://localhost:8080/metrics

# Indexer stats
curl http://localhost:8080/stats
```

## Intent Extraction (How It Works)

### Rule-Based Extraction (Production-Ready)

The current implementation uses **rule-based pattern matching** (matching Python's approach):

```go
analyzer := intent.NewIntentAnalyzer()
metadata := analyzer.AnalyzeContent(title, content, metaDescription)

// Extracted:
// - PrimaryGoal: intent.IntentGoalLearn
// - UseCases: []UseCase{UseCaseLearning, UseCaseTroubleshooting}
// - Complexity: ComplexitySimple
// - ResultType: ResultTypeTutorial
// - EthicalSignals: []EthicalSignal{...}
// - Topics: ["email", "encryption", "setup", ...]
// - Confidence: 0.85
```

### Future: ML-Based Extraction (Planned)

In production, this would use:
- **ONNX Runtime** for CPU-efficient ML inference
- **Quantized transformers** (DistilBERT, MiniLM)
- **Fine-tuned on intent classification** tasks

## Intent Alignment Scoring

When you search with intent, results are scored on:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Goal Match** | 35% | Does content goal match query goal? |
| **Use Case Match** | 25% | Does content support the use case? |
| **Complexity Match** | 15% | Is skill level appropriate? |
| **Ethical Match** | 15% | Does it align with ethical preferences? |
| **Temporal Match** | 10% | Is content timely/relevant? |

**Final Score** = `keyword_score * 0.5 + intent_alignment * 0.5`

This ensures **intent-aligned results** while maintaining relevance.

## Indexing Flow

```
1. Crawler fetches page → stores in PostgreSQL
2. Indexer worker polls PostgreSQL for unindexed pages
3. For each page:
   a. Extract intent signals (goal, use cases, ethics, etc.)
   b. Create IntentIndexedDocument with metadata
   c. Index in Bleve with intent fields
   d. Mark page as indexed in PostgreSQL
4. Search API queries Bleve with intent-aware queries
5. Results ranked by keyword + intent alignment
```

## Comparison: Before vs After

### Before (SearXNG Dependency)

```
User Query → SearXNG (external) → Intent Ranking → Results
                     ↑
              No intent indexing
```

### After (Native Intent Index)

```
User Query → Native Intent Index → Intent Alignment → Results
                    ↑
              Crawler + Indexer
              (intent signals indexed)
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Crawl Throughput** | 100+ pages/sec | Single instance |
| **Indexing Latency** | <100ms per doc | Intent extraction included |
| **Search P99** | <50ms | Intent-aware retrieval |
| **Intent Extraction** | <10ms per doc | Rule-based (fast) |
| **Memory Usage** | <500MB | With 100k pages indexed |

## Development

```bash
# Run tests
make test

# Run benchmarks
make bench

# Format code
make fmt

# Lint
make lint
```

## Integration with Python Intent Engine

The Go indexer integrates with the existing Python system:

```python
# Python side: Call Go search API instead of SearXNG
import httpx

async def search_with_intent(query: str, intent: UniversalIntent):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/api/v1/intent-search",
            json={
                "query": query,
                "intent": convert_intent_to_go_format(intent)
            }
        )
        return response.json()
```

## Future Enhancements

1. **Semantic Search** - Vector embeddings for semantic intent matching
2. **Distributed Crawling** - Multiple crawler nodes
3. **Real-time Indexing** - Stream processing
4. **ML-based Intent Extraction** - Fine-tuned transformers
5. **Query Intent Classification** - Classify query intent before search
6. **Intent Clustering** - Group similar intents for better matching

## License

MIT

## Acknowledgments

- Built with [Bleve](https://github.com/blevesearch/bleve) for full-text search
- Uses [Colly](https://github.com/gocolly/colly) for crawling (planned)
- Implements privacy-first design principles
- Intent schema matches Python Intent Engine
