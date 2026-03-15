package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/gorilla/mux"
	"github.com/itxLikhith/intent-engine/go-crawler/internal/indexer"
	"github.com/itxLikhith/intent-engine/go-crawler/internal/storage"
	"github.com/itxLikhith/intent-engine/go-crawler/pkg/intent"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	searchIndexer    *indexer.SearchIndexer
	store            *storage.Storage
	intentAnalyzer   *intent.IntentAnalyzer
	redisClient      *redis.Client
	searxngURL       string
	cacheEnabled     bool
	cacheTTL         time.Duration
	parallelSearch   bool
	
	// Prometheus metrics
	searchRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "unified_search_requests_total",
			Help: "Total number of search requests",
		},
		[]string{"source", "status"},
	)
	
	searchLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "unified_search_latency_seconds",
			Help:    "Search request latency in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"source"},
	)
	
	cacheHitsTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "unified_search_cache_hits_total",
			Help: "Total number of cache hits",
		},
	)
	
	cacheMissesTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "unified_search_cache_misses_total",
			Help: "Total number of cache misses",
		},
	)
	
	intentExtractionTime = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "unified_search_intent_extraction_seconds",
			Help:    "Intent extraction time in seconds",
			Buckets: prometheus.DefBuckets,
		},
	)
)

func init() {
	prometheus.MustRegister(searchRequestsTotal)
	prometheus.MustRegister(searchLatency)
	prometheus.MustRegister(cacheHitsTotal)
	prometheus.MustRegister(cacheMissesTotal)
	prometheus.MustRegister(intentExtractionTime)
}

// UnifiedSearchRequest represents a unified search request
type UnifiedSearchRequest struct {
	Query   string                 `json:"query"`
	Limit   int                    `json:"limit"`
	Filters map[string]interface{} `json:"filters,omitempty"`
}

// UnifiedSearchResult represents a unified search result
type UnifiedSearchResult struct {
	URL             string                        `json:"url"`
	Title           string                        `json:"title"`
	Content         string                        `json:"content"`
	Score           float64                       `json:"score"`
	Rank            int                           `json:"rank"`
	Source          string                        `json:"source"`
	IntentMetadata  *intent.IntentExtractionMetadata `json:"intent_metadata,omitempty"`
	AlignmentScore  *intent.IntentAlignmentScore     `json:"alignment_score,omitempty"`
	MatchReasons    []string                      `json:"match_reasons,omitempty"`
	VectorScore     float64                       `json:"vector_score,omitempty"`
}

// UnifiedSearchResponse represents a unified search response
type UnifiedSearchResponse struct {
	Query            string                     `json:"query"`
	Results          []UnifiedSearchResult      `json:"results"`
	TotalResults     int64                      `json:"total_results"`
	GoIndexResults   int64                      `json:"go_index_results"`
	SearxngResults   int64                      `json:"searxng_results"`
	VectorResults    int64                      `json:"vector_results"`
	ProcessingTimeMs float64                    `json:"processing_time_ms"`
	IntentExtracted  *intent.DeclaredIntent     `json:"intent_extracted,omitempty"`
	TopicsExpanded   []string                   `json:"topics_expanded,omitempty"`
	URLsAddedToQueue int                        `json:"urls_added_to_queue"`
	CacheHit         bool                       `json:"cache_hit"`
}

// SearchCacheItem represents a cached search result
type SearchCacheItem struct {
	Response   *UnifiedSearchResponse `json:"response"`
	Expiration time.Time              `json:"expiration"`
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8082"
	}

	searxngURL = os.Getenv("SEARXNG_URL")
	if searxngURL == "" {
		searxngURL = "http://searxng:8080"
	}

	blevePath := os.Getenv("BLEVE_PATH")
	if blevePath == "" {
		blevePath = "./data/bleve"
	}

	badgerPath := os.Getenv("BADGER_PATH")
	if badgerPath == "" {
		badgerPath = "./data/badger"
	}

	postgresDSN := os.Getenv("POSTGRES_DSN")
	if postgresDSN == "" {
		postgresDSN = "postgresql://intent_user:intent_secure_password_change_in_prod@postgres:5432/intent_engine?sslmode=disable"
	}

	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "redis:6379"
	}

	// Cache configuration
	cacheEnabled = os.Getenv("CACHE_ENABLED") != "false"
	cacheTTL = 1 * time.Hour
	if ttlEnv := os.Getenv("CACHE_TTL_SECONDS"); ttlEnv != "" {
		var ttl int
		fmt.Sscanf(ttlEnv, "%d", &ttl)
		cacheTTL = time.Duration(ttl) * time.Second
	}

	// Parallel search configuration
	parallelSearch = os.Getenv("PARALLEL_SEARCH") != "false"

	// Vector search configuration
	qdrantAddr := os.Getenv("QDRANT_ADDR")
	if qdrantAddr == "" {
		qdrantAddr = "qdrant:6333"
	}

	log.Printf("Starting Unified Search Service on port %s", port)
	log.Printf("SearXNG URL: %s", searxngURL)
	log.Printf("Bleve Path: %s", blevePath)
	log.Printf("PostgreSQL: %s", postgresDSN)
	log.Printf("Cache Enabled: %v, TTL: %v", cacheEnabled, cacheTTL)
	log.Printf("Parallel Search: %v", parallelSearch)
	log.Printf("Qdrant Address: %s", qdrantAddr)

	// Initialize storage
	storeCfg := &storage.StorageConfig{
		PostgresDSN: postgresDSN,
		BadgerPath:  badgerPath,
	}
	var err error
	store, err = storage.NewStorage(storeCfg)
	if err != nil {
		log.Fatalf("Failed to initialize storage: %v", err)
	}
	defer store.Close()

	// Initialize indexer
	searchIndexer, err = indexer.NewSearchIndexer(blevePath, store)
	if err != nil {
		log.Fatalf("Failed to initialize indexer: %v", err)
	}
	defer searchIndexer.Close()

	// Initialize intent analyzer
	intentAnalyzer = intent.NewIntentAnalyzer()
	log.Println("Intent analyzer initialized")

	// Initialize Redis client
	redisClient = redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})
	if err := redisClient.Ping(context.Background()).Err(); err != nil {
		log.Printf("Warning: Failed to connect to Redis: %v (cache disabled)", err)
		cacheEnabled = false
	} else {
		log.Println("Connected to Redis")
	}

	// Initialize vector search (Qdrant)
	var vectorSearch *VectorSearch
	if qdrantAddr != "" {
		vectorSearch = NewVectorSearch(qdrantAddr)
		if err := vectorSearch.Ping(); err != nil {
			log.Printf("Warning: Qdrant not available: %v (vector search disabled)", err)
			vectorSearch = nil
		} else {
			log.Println("Connected to Qdrant for vector search")
		}
	}

	// Setup routes
	r := mux.NewRouter()
	r.HandleFunc("/health", healthHandler).Methods("GET")
	r.HandleFunc("/stats", statsHandler).Methods("GET")
	r.HandleFunc("/api/v1/search", createSearchHandler(vectorSearch)).Methods("POST")
	r.HandleFunc("/api/v1/add-urls", addURLsHandler).Methods("POST")
	r.Handle("/metrics", promhttp.Handler()).Methods("GET")

	log.Printf("Unified Search Service ready on :%s", port)
	if err := http.ListenAndServe(":"+port, r); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func createSearchHandler(vectorSearch *VectorSearch) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		startTime := time.Now()

		// Parse request
		var req UnifiedSearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}

		if req.Query == "" {
			http.Error(w, "Query is required", http.StatusBadRequest)
			return
		}

		if req.Limit <= 0 {
			req.Limit = 10
		}
		if req.Limit > 100 {
			req.Limit = 100
		}

		log.Printf("Unified search query: %s (limit: %d)", req.Query, req.Limit)

		// Generate cache key
		cacheKey := generateCacheKey(req.Query, req.Limit)

		// Try cache first
		if cacheEnabled {
			if cached, found := getFromCache(cacheKey); found {
				cacheHitsTotal.Inc()
				cached.CacheHit = true
				cached.ProcessingTimeMs = float64(time.Since(startTime).Milliseconds())
				
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(cached)
				return
			}
			cacheMissesTotal.Inc()
		}

		// Step 1: Extract intent from query
		intentStartTime := time.Now()
		queryIntent := extractIntent(req.Query)
		intentExtractionTime.Observe(time.Since(intentStartTime).Seconds())
		log.Printf("Extracted intent: goal=%s, complexity=%s", queryIntent.Goal, queryIntent.SkillLevel)

		// Step 2: Parallel search - Go index + SearXNG + Vector
		var goResults, searxngResults, vectorResults []UnifiedSearchResult
		var goTotal, searxngTotal, vectorTotal int64
		urlsToAddToQueue := make([]string, 0)

		if parallelSearch {
			// Run searches in parallel
			var wg sync.WaitGroup
			
			// Go index search
			wg.Add(1)
			go func() {
				defer wg.Done()
				goResults, goTotal = searchGoIndex(req.Query, req.Limit, queryIntent)
				log.Printf("Go index results: %d/%d", len(goResults), goTotal)
			}()

			// SearXNG search
			wg.Add(1)
			go func() {
				defer wg.Done()
				searxngResults, searxngTotal, urlsToAddToQueue = searchSearxng(req.Query, req.Limit, queryIntent)
				log.Printf("SearXNG results: %d/%d, URLs to queue: %d", len(searxngResults), searxngTotal, len(urlsToAddToQueue))
			}()

			// Vector search (if available)
			if vectorSearch != nil {
				wg.Add(1)
				go func() {
					defer wg.Done()
					vectorResults, vectorTotal = vectorSearch.Search(req.Query, req.Limit, queryIntent)
					log.Printf("Vector results: %d/%d", len(vectorResults), vectorTotal)
				}()
			}

			wg.Wait()
		} else {
			// Sequential search (fallback)
			goResults, goTotal = searchGoIndex(req.Query, req.Limit, queryIntent)
			searxngResults, searxngTotal, urlsToAddToQueue = searchSearxng(req.Query, req.Limit, queryIntent)
			if vectorSearch != nil {
				vectorResults, vectorTotal = vectorSearch.Search(req.Query, req.Limit, queryIntent)
			}
		}

		// Step 3: Add new URLs to crawl queue
		if len(urlsToAddToQueue) > 0 {
			addedCount := addURLsToQueue(urlsToAddToQueue)
			log.Printf("Added %d URLs to crawl queue", addedCount)
		}

		// Step 4: Combine and rank results
		allResults := combineResults(goResults, searxngResults, vectorResults, queryIntent)

		// Step 5: Extract topics from results for expansion
		topicsExpanded := extractTopicsFromResults(allResults)

		// Build response
		response := UnifiedSearchResponse{
			Query:            req.Query,
			Results:          allResults,
			TotalResults:     int64(len(allResults)),
			GoIndexResults:   goTotal,
			SearxngResults:   searxngTotal,
			VectorResults:    vectorTotal,
			ProcessingTimeMs: float64(time.Since(startTime).Milliseconds()),
			IntentExtracted:  queryIntent,
			TopicsExpanded:   topicsExpanded,
			URLsAddedToQueue: len(urlsToAddToQueue),
			CacheHit:         false,
		}

		// Cache the response
		if cacheEnabled && len(allResults) > 0 {
			setInCache(cacheKey, &response)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}

// generateCacheKey creates a unique cache key for a query
func generateCacheKey(query string, limit int) string {
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", query, limit)))
	return fmt.Sprintf("search:%s", hex.EncodeToString(hash[:]))
}

// getFromCache retrieves a cached search result
func getFromCache(key string) (*UnifiedSearchResponse, bool) {
	ctx := context.Background()
	data, err := redisClient.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return nil, false
	}
	if err != nil {
		log.Printf("Cache get error: %v", err)
		return nil, false
	}

	var item SearchCacheItem
	if err := json.Unmarshal(data, &item); err != nil {
		log.Printf("Cache unmarshal error: %v", err)
		return nil, false
	}

	if time.Now().After(item.Expiration) {
		redisClient.Del(ctx, key)
		return nil, false
	}

	return item.Response, true
}

// setInCache stores a search result in cache
func setInCache(key string, response *UnifiedSearchResponse) {
	ctx := context.Background()
	item := SearchCacheItem{
		Response:   response,
		Expiration: time.Now().Add(cacheTTL),
	}

	data, err := json.Marshal(item)
	if err != nil {
		log.Printf("Cache marshal error: %v", err)
		return
	}

	if err := redisClient.Set(ctx, key, data, cacheTTL).Err(); err != nil {
		log.Printf("Cache set error: %v", err)
		return
	}

	log.Printf("Cached search results for key: %s (TTL: %v)", key, cacheTTL)
}

// VectorSearch provides vector similarity search using Qdrant
type VectorSearch struct {
	client      *http.Client
	qdrantAddr  string
	collection  string
}

// NewVectorSearch creates a new vector search client
func NewVectorSearch(qdrantAddr string) *VectorSearch {
	return &VectorSearch{
		client:     &http.Client{Timeout: 5 * time.Second},
		qdrantAddr: qdrantAddr,
		collection: "intent_vectors",
	}
}

// Ping checks if Qdrant is available
func (vs *VectorSearch) Ping() error {
	resp, err := vs.client.Get(fmt.Sprintf("http://%s/", vs.qdrantAddr))
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// Search performs vector similarity search
func (vs *VectorSearch) Search(query string, limit int, queryIntent *intent.DeclaredIntent) ([]UnifiedSearchResult, int64) {
	results := make([]UnifiedSearchResult, 0)
	startTime := time.Now()
	
	// For now, return empty results since we need embedding service
	// In production, would:
	// 1. Call Python embedding service to get query vector
	// 2. Search Qdrant for similar vectors
	// 3. Return results with vector scores
	
	// Placeholder: query Qdrant directly with a simple vector
	// This would be replaced with real embeddings
	queryVector := make([]float64, 384)
	for i := range queryVector {
		queryVector[i] = 0.01 // Dummy vector for testing
	}
	
	searchReq := map[string]interface{}{
		"vector": queryVector,
		"limit": limit,
		"with_payload": true,
		"with_vector": false,
	}
	
	resp, err := vs.client.Post(
		fmt.Sprintf("http://%s/collections/%s/points/search", vs.qdrantAddr, vs.collection),
		"application/json",
		strings.NewReader(fmt.Sprintf("%v", searchReq)),
	)
	
	if err == nil {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		var qdrantResp struct {
			Result []struct {
				Score   float64            `json:"score"`
				Payload map[string]interface{} `json:"payload"`
			} `json:"result"`
		}
		json.Unmarshal(body, &qdrantResp)
		
		for i, res := range qdrantResp.Result {
			result := UnifiedSearchResult{
				URL:     fmt.Sprintf("%v", res.Payload["url"]),
				Title:   fmt.Sprintf("%v", res.Payload["title"]),
				Content: truncateContent(fmt.Sprintf("%v", res.Payload["content"]), 200),
				Score:   res.Score,
				Rank:    i + 1,
				Source:  "vector",
				VectorScore: res.Score,
			}
			results = append(results, result)
		}
	}
	
	searchLatency.WithLabelValues("vector").Observe(time.Since(startTime).Seconds())
	searchRequestsTotal.WithLabelValues("vector", "success").Inc()
	
	return results, int64(len(results))
}

// extractIntent extracts intent from query using rule-based analyzer
func extractIntent(query string) *intent.DeclaredIntent {
	queryMetadata := intentAnalyzer.AnalyzeContent(query, query, query)

	declaredIntent := &intent.DeclaredIntent{
		Query:           query,
		Goal:            queryMetadata.PrimaryGoal,
		Constraints:     make([]intent.Constraint, 0),
		Urgency:         intent.UrgencyFlexible,
		SkillLevel:      queryMetadata.TargetSkillLevel,
		UseCases:        queryMetadata.UseCases,
		EthicalSignals:  queryMetadata.EthicalSignals,
	}

	declaredIntent.TemporalIntent = &intent.TemporalIntent{
		Horizon:   intent.TemporalHorizonFlexible,
		Recency:   intent.RecencyEvergreen,
		Frequency: intent.FrequencyFlexible,
	}

	return declaredIntent
}

// searchGoIndex searches the local Go index
func searchGoIndex(query string, limit int, queryIntent *intent.DeclaredIntent) ([]UnifiedSearchResult, int64) {
	results := make([]UnifiedSearchResult, 0)
	startTime := time.Now()

	searchResult, err := searchIndexer.Search(query, limit)
	if err != nil {
		log.Printf("Go index search error: %v", err)
		searchRequestsTotal.WithLabelValues("go_index", "error").Inc()
		return results, 0
	}

	for i, hit := range searchResult.Hits {
		result := UnifiedSearchResult{
			URL:     hit.Document.URL,
			Title:   hit.Document.Title,
			Content: truncateContent(hit.Document.Content, 200),
			Score:   hit.Score,
			Rank:    i + 1,
			Source:  "go_index",
		}

		if hit.Document.IntentMetadata != nil {
			doc := &intent.IntentIndexedDocument{
				ID:             hit.Document.ID,
				URL:            hit.Document.URL,
				Title:          hit.Document.Title,
				Content:        hit.Document.Content,
				IntentMetadata: hit.Document.IntentMetadata,
			}
			alignment := intent.ComputeIntentAlignment(doc, queryIntent)
			result.AlignmentScore = alignment
			result.MatchReasons = alignment.MatchReasons
			result.Score = hit.Score * (0.5 + 0.5*alignment.TotalScore)
		}

		results = append(results, result)
	}

	searchLatency.WithLabelValues("go_index").Observe(time.Since(startTime).Seconds())
	searchRequestsTotal.WithLabelValues("go_index", "success").Inc()

	return results, searchResult.Total
}

// searchSearxng searches SearXNG metasearch engine
func searchSearxng(query string, limit int, queryIntent *intent.DeclaredIntent) ([]UnifiedSearchResult, int64, []string) {
	results := make([]UnifiedSearchResult, 0)
	urlsToAdd := make([]string, 0)
	startTime := time.Now()

	searchURL := fmt.Sprintf("%s/search?q=%s&format=json&language=en&categories=general&pageno=1",
		searxngURL,
		query)

	resp, err := http.Get(searchURL)
	if err != nil {
		log.Printf("SearXNG search error: %v", err)
		searchRequestsTotal.WithLabelValues("searxng", "error").Inc()
		return results, 0, urlsToAdd
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("SearXNG response read error: %v", err)
		return results, 0, urlsToAdd
	}

	// Check for empty response
	if len(body) == 0 {
		log.Printf("SearXNG returned empty response")
		return results, 0, urlsToAdd
	}

	var searxngResp struct {
		Query           string `json:"query"`
		Results         []struct {
			URL     string `json:"url"`
			Title   string `json:"title"`
			Content string `json:"content"`
			Engine  string `json:"engine"`
		} `json:"results"`
		NumberOfResults int64 `json:"number_of_results"`
	}

	if err := json.Unmarshal(body, &searxngResp); err != nil {
		log.Printf("SearXNG response parse error: %v (body: %s)", err, string(body[:min(100, len(body))]))
		return results, 0, urlsToAdd
	}

	for i, res := range searxngResp.Results {
		if i >= limit {
			break
		}

		result := UnifiedSearchResult{
			URL:     res.URL,
			Title:   res.Title,
			Content: truncateContent(res.Content, 200),
			Score:   float64(limit - i),
			Rank:    i + 1,
			Source:  "searxng",
		}

		result.IntentMetadata = intentAnalyzer.AnalyzeContent(res.Title, res.Content, "")
		result.MatchReasons = []string{fmt.Sprintf("matches-%s-intent", queryIntent.Goal)}

		results = append(results, result)

		if !isKnownDomain(res.URL) {
			urlsToAdd = append(urlsToAdd, res.URL)
		}
	}

	searchLatency.WithLabelValues("searxng").Observe(time.Since(startTime).Seconds())
	searchRequestsTotal.WithLabelValues("searxng", "success").Inc()

	return results, searxngResp.NumberOfResults, urlsToAdd
}

func isKnownDomain(url string) bool {
	knownDomains := []string{
		"go.dev", "golang.org", "pkg.go.dev", "github.com/golang",
	}

	for _, domain := range knownDomains {
		if strings.Contains(url, domain) {
			return true
		}
	}
	return false
}

func addURLsToQueue(urls []string) int {
	if len(urls) == 0 || redisClient == nil {
		return 0
	}

	added := 0
	for _, url := range urls {
		visitedKey := fmt.Sprintf("visited_urls:%d", hashURL(url))
		exists, err := redisClient.Exists(context.Background(), visitedKey).Result()
		if err == nil && exists == 0 {
			item := map[string]interface{}{
				"id":          fmt.Sprintf("crawl_%d", time.Now().UnixNano()),
				"url":         url,
				"priority":    5,
				"depth":       1,
				"status":      "pending",
				"scheduledAt": time.Now(),
				"createdAt":   time.Now(),
				"updatedAt":   time.Now(),
			}

			itemJSON, _ := json.Marshal(item)
			redisClient.ZAdd(context.Background(), "crawl_queue", &redis.Z{
				Score:  5.0,
				Member: string(itemJSON),
			})

			redisClient.Set(context.Background(), visitedKey, "1", 24*time.Hour)
			added++
		}
	}

	return added
}

func combineResults(goResults, searxngResults, vectorResults []UnifiedSearchResult, queryIntent *intent.DeclaredIntent) []UnifiedSearchResult {
	allResults := append(append(goResults, searxngResults...), vectorResults...)

	for i := range allResults {
		if allResults[i].AlignmentScore != nil {
			allResults[i].Score *= (0.5 + 0.5*allResults[i].AlignmentScore.TotalScore)
		}
		if allResults[i].VectorScore > 0 {
			allResults[i].Score *= (0.5 + 0.5*allResults[i].VectorScore)
		}
	}

	for i := 0; i < len(allResults)-1; i++ {
		for j := 0; j < len(allResults)-i-1; j++ {
			if allResults[j].Score < allResults[j+1].Score {
				allResults[j], allResults[j+1] = allResults[j+1], allResults[j]
			}
		}
	}

	for i := range allResults {
		allResults[i].Rank = i + 1
	}

	return allResults
}

func extractTopicsFromResults(results []UnifiedSearchResult) []string {
	topicMap := make(map[string]int)

	for _, result := range results {
		if result.IntentMetadata != nil {
			for _, topic := range result.IntentMetadata.Topics {
				topicMap[topic]++
			}
			for _, phrase := range result.IntentMetadata.KeyPhrases {
				topicMap[phrase]++
			}
		}
	}

	topics := make([]string, 0, 10)
	for topic := range topicMap {
		topics = append(topics, topic)
		if len(topics) >= 10 {
			break
		}
	}

	return topics
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	checks := map[string]bool{
		"indexer": searchIndexer != nil,
		"storage": store != nil,
		"intent":  intentAnalyzer != nil,
		"redis":   redisClient != nil,
		"searxng": searxngURL != "",
	}

	status := "healthy"
	for _, ok := range checks {
		if !ok {
			status = "degraded"
			break
		}
	}

	response := map[string]interface{}{
		"status":    status,
		"timestamp": time.Now(),
		"checks":    checks,
		"version":   "1.0.0",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	stats := map[string]interface{}{}

	if searchIndexer != nil {
		indexStats, err := searchIndexer.Stats()
		if err == nil {
			stats["indexed_documents"] = indexStats.DocumentCount
		}
	}

	if store != nil {
		crawlStats, err := store.GetStats()
		if err == nil {
			stats["crawled_pages"] = crawlStats.TotalPages
			stats["pages_indexed"] = crawlStats.PagesIndexed
		}
	}

	if redisClient != nil {
		queueSize, err := redisClient.ZCard(context.Background(), "crawl_queue").Result()
		if err == nil {
			stats["queue_size"] = queueSize
		}

		info, err := redisClient.Info(context.Background(), "memory").Result()
		if err == nil {
			stats["redis_memory"] = info
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func addURLsHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		URLs     []string `json:"urls"`
		Priority int      `json:"priority"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.Priority <= 0 {
		req.Priority = 5
	}

	added := addURLsToQueueWithPriority(req.URLs, req.Priority)

	response := map[string]interface{}{
		"success":   true,
		"urlsAdded": added,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func addURLsToQueueWithPriority(urls []string, priority int) int {
	if len(urls) == 0 || redisClient == nil {
		return 0
	}

	added := 0
	for _, url := range urls {
		visitedKey := fmt.Sprintf("visited_urls:%d", hashURL(url))
		exists, err := redisClient.Exists(context.Background(), visitedKey).Result()
		if err == nil && exists == 0 {
			item := map[string]interface{}{
				"id":          fmt.Sprintf("crawl_%d", time.Now().UnixNano()),
				"url":         url,
				"priority":    priority,
				"depth":       0,
				"status":      "pending",
				"scheduledAt": time.Now(),
				"createdAt":   time.Now(),
				"updatedAt":   time.Now(),
			}

			itemJSON, _ := json.Marshal(item)
			redisClient.ZAdd(context.Background(), "crawl_queue", &redis.Z{
				Score:  float64(priority),
				Member: string(itemJSON),
			})

			redisClient.Set(context.Background(), visitedKey, "1", 24*time.Hour)
			added++
		}
	}

	return added
}

func hashURL(url string) int64 {
	h := int64(0)
	for i := 0; i < len(url); i++ {
		h = 31*h + int64(url[i])
	}
	return h
}

func truncateContent(content string, maxLen int) string {
	if len(content) <= maxLen {
		return content
	}
	return content[:maxLen] + "..."
}
