# Go-Based Crawler & Indexer Architecture Plan

## Intent Engine - Custom Search Infrastructure

**Version:** 1.0.0  
**Date:** March 14, 2026  
**Status:** Planning Phase

---

## Executive Summary

This document outlines the architecture and implementation plan for building a **custom web crawler and search indexer in Go** to replace the dependency on SearXNG for the Intent Engine project. The new system will provide:

- **Autonomous content discovery** - No reliance on external search aggregators
- **High-performance crawling** - 1000+ pages/sec with Go's concurrency model
- **Intent-aware indexing** - Tight integration with existing Intent Engine
- **Distributed architecture** - Scalable across multiple nodes
- **Privacy-first design** - Respects robots.txt, rate limiting, ethical crawling

---

## Current State Analysis

### Existing Python Implementation (Removed)

The previous implementation used:
- `aiohttp` + `BeautifulSoup` for crawling
- Basic TF-IDF + PageRank in SQLite/PostgreSQL
- Single-threaded async with limited concurrency
- Tightly coupled with database
- No distributed capabilities

### Current Search Flow (SearXNG Dependent)

```
User Query → SearXNG (external) → Intent Extraction → Ranking → Results
```

**Problem:** Complete dependency on SearXNG for content discovery

### Target Search Flow (Native)

```
User Query → Native Index → Intent Ranking → Results
                    ↑
              Crawler (background)
```

---

## Architecture Overview

### System Diagram

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
│  │    Redis     │    │    Bleve     │    │   Intent     │      │
│  │    (Queue)   │    │    (Index)   │    │   Ranking    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Python Intent API   │
              │   (Existing System)   │
              └───────────────────────┘
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      URL Frontier                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Redis Queue │  │ Bloom Filter│  │ Per-Domain Rate     │  │
│  │ (Priority)  │  │ (Dedup)     │  │ Limiting (Colly)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Crawler Workers (Colly)                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐  │
│  │ Fetcher   │  │ Parser    │  │ Link      │  │ Storage  │  │
│  │ (HTTP)    │  │ (goquery) │  │ Extractor │  │ Pipeline │  │
│  └───────────┘  └───────────┘  └───────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ PostgreSQL  │  │ BadgerDB    │  │ Bleve Index         │  │
│  │ (Metadata)  │  │ (Raw HTML)  │  │ (Full-text Search)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Components

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Crawler Framework** | Colly v2 | Production-ready, 22k+ stars, rate limiting, caching |
| **HTML Parser** | goquery | jQuery-like API, fast, familiar |
| **Search Index** | Bleve | Pure Go, BM25/TF-IDF, vector search support |
| **URL Queue** | Redis | Distributed, fast, priority queues |
| **Bloom Filter** | RedisBloom | Memory-efficient dedup (<1% error rate) |
| **Content Store** | BadgerDB | High write throughput, Go-native, LSM-tree |
| **Metadata Store** | PostgreSQL | Existing DB, ACID transactions |
| **HTTP API** | FastHTTP | High performance, low memory |
| **gRPC** | gRPC-Go | Internal service communication |
| **Monitoring** | Prometheus | Industry standard, Grafana integration |

### Libraries & Dependencies

```go
// Crawler
github.com/gocolly/colly/v2
github.com/PuerkitoBio/goquery
github.com/temoto/robotstxt

// Search Index
github.com/blevesearch/bleve/v2

// Storage
github.com/dgraph-io/badger/v4
github.com/go-redis/redis/v8
github.com/lib/pq

// API
github.com/valyala/fasthttp
google.golang.org/grpc

// Monitoring
github.com/prometheus/client_golang

// Utilities
github.com/bits-and-blooms/bloom
github.com/hashicorp/go-retryablehttp
```

---

## Project Structure

```
intent-engine/
├── cmd/
│   ├── crawler/           # Crawler service binary
│   │   └── main.go
│   ├── indexer/           # Indexer service binary
│   │   └── main.go
│   ├── search-api/        # HTTP/gRPC API server
│   │   └── main.go
│   └── pagerank/          # PageRank calculator (batch job)
│       └── main.go
│
├── internal/
│   ├── crawler/
│   │   ├── collector.go   # Colly-based crawler
│   │   ├── fetcher.go     # HTTP fetching with retry
│   │   ├── parser.go      # HTML parsing (goquery)
│   │   ├── robots.go      # Robots.txt handling
│   │   └── config.go      # Crawler configuration
│   │
│   ├── frontier/
│   │   ├── queue.go       # Redis URL queue
│   │   ├── bloom.go       # Bloom filter for dedup
│   │   └── rate.go        # Per-domain rate limiting
│   │
│   ├── indexer/
│   │   ├── bleve.go       # Bleve index wrapper
│   │   ├── analyzer.go    # Custom text analyzer
│   │   └── tfidf.go       # TF-IDF scoring
│   │
│   ├── pagerank/
│   │   ├── calculator.go  # PageRank algorithm
│   │   └── graph.go       # Link graph
│   │
│   ├── storage/
│   │   ├── badger.go      # BadgerDB storage
│   │   ├── postgres.go    # PostgreSQL metadata
│   │   └── cache.go       # Redis caching
│   │
│   ├── api/
│   │   ├── http.go        # HTTP REST API
│   │   ├── grpc.go        # gRPC service
│   │   └── handlers.go    # Request handlers
│   │
│   └── metrics/
│       ├── prometheus.go  # Prometheus metrics
│       └── health.go      # Health checks
│
├── pkg/
│   ├── models/            # Shared data models
│   │   ├── page.go
│   │   ├── link.go
│   │   └── search.go
│   │
│   └── proto/             # Protocol buffers
│       ├── crawler.proto
│       └── search.proto
│
├── configs/
│   ├── crawler.yaml
│   ├── indexer.yaml
│   └── docker-compose.yml
│
├── deployments/
│   ├── Dockerfile.crawler
│   ├── Dockerfile.indexer
│   └── k8s/
│       ├── crawler-deployment.yaml
│       └── indexer-deployment.yaml
│
├── scripts/
│   ├── seed-urls.txt
│   └── benchmark.sh
│
├── go.mod
├── go.sum
└── README.md
```

---

## Implementation Phases

### Phase 1: Core Crawler (Week 1)

**Goal:** Build a functional web crawler with basic features

#### Tasks:
- [ ] **Day 1-2: Project Setup**
  - Initialize Go module
  - Set up project structure
  - Configure dependencies
  - Create basic Makefile

- [ ] **Day 3-4: Crawler Engine**
  - Implement Colly-based collector
  - Configure rate limiting
  - Add robots.txt compliance
  - Basic error handling

- [ ] **Day 5-7: URL Frontier**
  - Redis queue implementation
  - Bloom filter for deduplication
  - Per-domain rate limiting
  - Priority-based scheduling

#### Deliverables:
- ✅ Working crawler binary
- ✅ Redis-based URL queue
- ✅ Basic content extraction
- ✅ Robots.txt respect

#### Code Example:
```go
package crawler

import (
    "github.com/gocolly/colly/v2"
)

type Crawler struct {
    collector *colly.Collector
    config    *Config
}

func NewCrawler(config *Config) *Crawler {
    c := colly.NewCollector(
        colly.AllowedDomains(config.AllowedDomains...),
        colly.MaxDepth(config.MaxDepth),
        colly.MaxBodySize(config.MaxBodySize),
        colly.UserAgent(config.UserAgent),
        colly.Async(true),
    )
    
    c.Limit(&colly.LimitRule{
        DomainGlob:  "*",
        Parallelism: config.Parallelism,
        Delay:       config.Delay,
        RandomDelay: config.RandomDelay,
    })
    
    return &Crawler{
        collector: c,
        config:    config,
    }
}
```

---

### Phase 2: Indexer (Week 2)

**Goal:** Build full-text search index with Bleve

#### Tasks:
- [ ] **Day 1-2: Bleve Setup**
  - Initialize Bleve index
  - Configure field mappings
  - Set up analyzers

- [ ] **Day 3-4: Text Processing**
  - Tokenization
  - Stemming
  - Stop word removal
  - Custom analyzers

- [ ] **Day 5-7: Storage Integration**
  - BadgerDB for raw content
  - PostgreSQL for metadata
  - Index synchronization

#### Deliverables:
- ✅ Bleve search index
- ✅ Text processing pipeline
- ✅ Dual storage (BadgerDB + PostgreSQL)

#### Code Example:
```go
package indexer

import (
    "github.com/blevesearch/bleve/v2"
)

type SearchIndexer struct {
    index bleve.Index
}

func NewSearchIndexer(path string) (*SearchIndexer, error) {
    mapping := bleve.NewIndexMapping()
    mapping.DefaultMapping.AddFieldMappingsAt("title", 
        bleve.NewTextFieldMapping())
    mapping.DefaultMapping.AddFieldMappingsAt("content",
        bleve.NewTextFieldMapping())
    
    index, err := bleve.New(path, mapping)
    if err != nil {
        return nil, err
    }
    
    return &SearchIndexer{index: index}, nil
}

func (i *SearchIndexer) IndexDocument(doc *Document) error {
    return i.index.Index(doc.ID, doc)
}

func (i *SearchIndexer) Search(query string, limit int) (*SearchResult, error) {
    q := bleve.NewQueryStringQuery(query)
    req := bleve.NewSearchRequest(q)
    req.Size = limit
    return i.index.Search(req)
}
```

---

### Phase 3: PageRank (Week 3)

**Goal:** Implement distributed PageRank calculation

#### Tasks:
- [ ] **Day 1-2: Link Graph**
  - Extract links from crawled pages
  - Build adjacency list
  - Store in Redis/PostgreSQL

- [ ] **Day 3-4: PageRank Algorithm**
  - Implement iterative PageRank
  - Damping factor configuration
  - Convergence detection

- [ ] **Day 5-7: Optimization**
  - Incremental updates
  - Distributed calculation
  - Score persistence

#### Deliverables:
- ✅ PageRank calculator
- ✅ Link graph storage
- ✅ Score integration with search

#### Code Example:
```go
package pagerank

type PageRankCalculator struct {
    damping    float64
    iterations int
}

func (c *PageRankCalculator) Calculate(graph *LinkGraph) map[string]float64 {
    pages := graph.Pages()
    n := len(pages)
    
    // Initialize
    scores := make(map[string]float64)
    for _, page := range pages {
        scores[page] = 1.0 / float64(n)
    }
    
    // Iterate
    for i := 0; i < c.iterations; i++ {
        newScores := make(map[string]float64)
        
        for _, page := range pages {
            rankSum := 0.0
            for _, incoming := range graph.IncomingLinks(page) {
                outCount := graph.OutgoingCount(incoming)
                if outCount > 0 {
                    rankSum += scores[incoming] / float64(outCount)
                }
            }
            newScores[page] = (1 - c.damping) / float64(n) + c.damping*rankSum
        }
        
        scores = newScores
    }
    
    return scores
}
```

---

### Phase 4: API & Integration (Week 4)

**Goal:** Expose search functionality via HTTP/gRPC

#### Tasks:
- [ ] **Day 1-2: HTTP API**
  - Search endpoint
  - Health checks
  - Metrics endpoint

- [ ] **Day 3-4: gRPC Service**
  - Define protobuf schemas
  - Implement gRPC server
  - Internal service communication

- [ ] **Day 5-7: Python Integration**
  - gRPC client for Python
  - Replace SearXNG calls
  - Intent ranking integration

#### Deliverables:
- ✅ REST API (search, health, metrics)
- ✅ gRPC service
- ✅ Python integration layer

#### API Endpoints:

**HTTP REST:**
```
GET  /health              # Health check
GET  /metrics             # Prometheus metrics
POST /search              # Search query
GET  /stats               # Crawler/indexer stats
POST /crawl/seed          # Add seed URLs
GET  /crawl/status        # Crawl status
```

**gRPC Service:**
```protobuf
service SearchService {
    rpc Search(SearchRequest) returns (SearchResponse);
    rpc Crawl(CrawlRequest) returns (CrawlResponse);
    rpc GetStats(StatsRequest) returns (StatsResponse);
}
```

---

### Phase 5: Production Hardening (Week 5)

**Goal:** Make the system production-ready

#### Tasks:
- [ ] **Day 1-2: Monitoring**
  - Prometheus metrics
  - Grafana dashboards
  - Alerting rules

- [ ] **Day 3-4: Error Handling**
  - Retry logic with backoff
  - Circuit breakers
  - Graceful degradation

- [ ] **Day 5-7: Testing & Docs**
  - Unit tests
  - Integration tests
  - Load testing
  - Documentation

#### Deliverables:
- ✅ Monitoring dashboards
- ✅ Comprehensive tests
- ✅ Production deployment guide

#### Metrics to Track:
```go
// Crawler Metrics
crawler_pages_total
crawler_pages_success
crawler_pages_failed
crawler_duration_seconds
crawler_queue_size

// Indexer Metrics
indexer_documents_total
indexer_index_duration_seconds
indexer_index_size_bytes

// Search Metrics
search_requests_total
search_duration_seconds
search_results_count

// System Metrics
system_memory_bytes
system_goroutines_count
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Crawl Throughput** | 1000+ pages/sec | Single instance |
| **Index Latency** | <50ms per doc | From crawl to searchable |
| **Search P99** | <100ms | Including intent ranking |
| **Memory Usage** | <2GB | With 1M pages indexed |
| **URL Dedup Accuracy** | >99.9% | Bloom filter + Redis |
| **PageRank Convergence** | <10 iterations | For 100k pages |

---

## Key Design Decisions

### 1. Why Colly over Building from Scratch?

**Pros:**
- Production-ready (22k+ stars)
- Built-in rate limiting
- Robots.txt compliance
- Distributed crawling support
- Active maintenance

**Cons:**
- Less control over low-level details
- Dependency on external library

**Decision:** Use Colly for speed and reliability

### 2. Why Bleve over External Search Service?

**Pros:**
- Pure Go (no external dependencies)
- Embedded (no network overhead)
- BM25/TF-IDF built-in
- Vector search support (future)
- Full control over indexing

**Cons:**
- Index stored on disk (scaling limits)
- No distributed search out-of-box

**Decision:** Use Bleve for simplicity and performance

### 3. Why BadgerDB for Content Storage?

**Pros:**
- Go-native
- LSM-tree (high write throughput)
- ACID transactions
- Built-in compression
- Low memory footprint

**Cons:**
- Less mature than LevelDB/RocksDB
- Single-node only

**Decision:** Use BadgerDB for raw HTML storage

### 4. Why Redis for URL Queue?

**Pros:**
- Fast (in-memory)
- Distributed (multiple crawlers)
- Priority queues
- Bloom filter support
- Persistence options

**Cons:**
- Additional infrastructure
- Memory cost

**Decision:** Use Redis for distributed coordination

---

## Data Models

### Page Model
```go
type CrawledPage struct {
    ID              string    `json:"id"`
    URL             string    `json:"url"`
    FinalURL        string    `json:"final_url"`
    Title           string    `json:"title"`
    Content         string    `json:"content"`
    HTMLContent     []byte    `json:"-"` // Stored in BadgerDB
    MetaDescription string    `json:"meta_description"`
    MetaKeywords    string    `json:"meta_keywords"`
    StatusCode      int       `json:"status_code"`
    ContentType     string    `json:"content_type"`
    ContentLength   int       `json:"content_length"`
    LoadTimeMs      float64   `json:"load_time_ms"`
    CrawlDepth      int       `json:"crawl_depth"`
    OutboundLinks   int       `json:"outbound_links"`
    PageRank        float64   `json:"pagerank"`
    CrawledAt       time.Time `json:"crawled_at"`
    UpdatedAt       time.Time `json:"updated_at"`
}
```

### Link Model
```go
type PageLink struct {
    SourcePageID string    `json:"source_page_id"`
    TargetURL    string    `json:"target_url"`
    AnchorText   string    `json:"anchor_text"`
    LinkType     string    `json:"link_type"` // dofollow, nofollow
    CreatedAt    time.Time `json:"created_at"`
}
```

### Search Document
```go
type SearchDocument struct {
    ID              string            `json:"id"`
    URL             string            `json:"url"`
    Title           string            `json:"title"`
    Content         string            `json:"content"`
    MetaDescription string            `json:"meta_description"`
    TermFrequencies map[string]int    `json:"term_frequencies"`
    WordCount       int               `json:"word_count"`
    PageRank        float64           `json:"pagerank"`
    IndexedAt       time.Time         `json:"indexed_at"`
}
```

---

## Configuration

### Crawler Configuration (crawler.yaml)
```yaml
crawler:
  # Concurrency
  max_concurrent_requests: 10
  max_connections_per_host: 2
  
  # Limits
  max_pages: 10000
  max_depth: 5
  max_pages_per_domain: 1000
  
  # Timing
  request_timeout: 30s
  crawl_delay: 1s
  respect_robots_txt: true
  user_agent: "IntentEngineBot/1.0"
  
  # Content
  max_content_size: 10485760  # 10MB
  allowed_content_types:
    - "text/html"
    - "application/xhtml+xml"
  
  # Domains
  allowed_domains: []  # Empty = all
  blocked_domains:
    - "facebook.com"
    - "twitter.com"
    - "instagram.com"

redis:
  addr: "localhost:6379"
  password: ""
  db: 0

storage:
  badger_path: "./data/badger"
  postgres_dsn: "postgresql://user:pass@localhost:5432/intent_engine"
  bleve_path: "./data/bleve"
```

---

## Deployment

### Docker Compose (Development)
```yaml
version: '3.8'

services:
  crawler:
    build:
      context: .
      dockerfile: deployments/Dockerfile.crawler
    environment:
      - REDIS_ADDR=redis:6379
      - POSTGRES_DSN=postgresql://user:pass@postgres:5432/intent_engine
    volumes:
      - crawler_data:/data
    depends_on:
      - redis
      - postgres

  indexer:
    build:
      context: .
      dockerfile: deployments/Dockerfile.indexer
    environment:
      - REDIS_ADDR=redis:6379
      - POSTGRES_DSN=postgresql://user:pass@postgres:5432/intent_engine
    volumes:
      - indexer_data:/data
    depends_on:
      - redis
      - postgres

  search-api:
    build:
      context: .
      dockerfile: deployments/Dockerfile.api
    ports:
      - "8080:8080"
    environment:
      - REDIS_ADDR=redis:6379
      - POSTGRES_DSN=postgresql://user:pass@postgres:5432/intent_engine
    depends_on:
      - crawler
      - indexer

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=intent_engine
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  crawler_data:
  indexer_data:
  redis_data:
  postgres_data:
```

---

## Testing Strategy

### Unit Tests
```go
func TestCrawler_CrawlURL(t *testing.T) {
    config := &Config{
        MaxDepth: 2,
        Timeout:  10 * time.Second,
    }
    
    crawler := NewCrawler(config)
    result, err := crawler.CrawlURL("https://example.com")
    
    assert.NoError(t, err)
    assert.Equal(t, "Example Domain", result.Title)
    assert.Greater(t, len(result.Content), 0)
}

func TestIndexer_IndexDocument(t *testing.T) {
    indexer := NewTestIndexer()
    defer indexer.Close()
    
    doc := &Document{
        ID:      "test-1",
        Title:   "Test Document",
        Content: "This is a test",
    }
    
    err := indexer.IndexDocument(doc)
    assert.NoError(t, err)
    
    results, _ := indexer.Search("test", 10)
    assert.Equal(t, 1, len(results))
}
```

### Integration Tests
```go
func TestCrawlerIndexer_Integration(t *testing.T) {
    // Start Redis, PostgreSQL
    // Seed URLs
    // Run crawler
    // Verify index
    // Search and validate results
}
```

### Load Tests
```bash
# Benchmark search
ab -n 10000 -c 100 http://localhost:8080/search?q=test

# Benchmark crawl
go test -bench=BenchmarkCrawl -benchtime=1m
```

---

## Migration Plan

### From Python to Go

1. **Phase 1:** Run Go crawler alongside Python system
2. **Phase 2:** Gradually shift traffic to Go search
3. **Phase 3:** Deprecate Python crawler (if any)
4. **Phase 4:** Remove SearXNG dependency

### Data Migration

- Existing PostgreSQL tables compatible
- New BadgerDB for raw content
- Bleve index built from scratch

---

## Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Colly limitations | Medium | Low | Fork and modify if needed |
| Bleve scaling issues | High | Medium | Plan for Elasticsearch migration |
| Redis single point of failure | High | Low | Redis Sentinel/Cluster |
| Memory exhaustion | Medium | Medium | Implement backpressure |
| Legal/ethical concerns | High | Low | Respect robots.txt, rate limiting |

---

## Success Criteria

- [ ] Crawl 100k pages without failures
- [ ] Search latency <100ms P99
- [ ] Zero SearXNG dependencies
- [ ] Full integration with Intent Engine
- [ ] Production deployment successful
- [ ] Comprehensive documentation

---

## Future Enhancements

1. **Semantic Search** - Vector embeddings with Bleve
2. **Distributed Crawling** - Multiple crawler nodes
3. **Real-time Indexing** - Stream processing
4. **ML-based Ranking** - Intent-aware ML models
5. **CDN Integration** - Cache crawled content
6. **JavaScript Rendering** - Headless Chrome for SPAs

---

## References

- [Colly Documentation](https://pkg.go.dev/github.com/gocolly/colly/v2)
- [Bleve Documentation](https://pkg.go.dev/github.com/blevesearch/bleve/v2)
- [Redis Bloom Filter](https://redis.io/docs/data-types/probabilistic/bloom-filters/)
- [BadgerDB Documentation](https://dgraph.io/docs/badger/)
- [PageRank Paper](https://pagerank.stanford.edu/)

---

## Appendix A: Command Reference

```bash
# Build all binaries
make build

# Run crawler
make run-crawler

# Run indexer
make run-indexer

# Run search API
make run-api

# Run tests
make test

# Run benchmarks
make bench

# Docker deployment
docker-compose up -d

# Load test
make load-test
```

---

**Document End**
