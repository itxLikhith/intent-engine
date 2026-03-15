package pagerank

import (
	"context"
	"fmt"
	"log"
	"math"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/itxLikhith/intent-engine/go-crawler/pkg/models"
)

// LinkGraph represents the web link graph
type LinkGraph struct {
	outboundLinks map[string][]string // page -> outgoing links
	inboundLinks  map[string][]string // page -> incoming links
	pages         map[string]bool
	mu            sync.RWMutex
}

// NewLinkGraph creates a new link graph
func NewLinkGraph() *LinkGraph {
	return &LinkGraph{
		outboundLinks: make(map[string][]string),
		inboundLinks:  make(map[string][]string),
		pages:         make(map[string]bool),
	}
}

// AddPage adds a page to the graph
func (g *LinkGraph) AddPage(url string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.pages[url] = true
}

// AddLink adds a link from source to target
func (g *LinkGraph) AddLink(source, target string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.pages[source] = true
	g.pages[target] = true

	// Add outbound link
	g.outboundLinks[source] = append(g.outboundLinks[source], target)

	// Add inbound link
	g.inboundLinks[target] = append(g.inboundLinks[target], source)
}

// Pages returns all pages in the graph
func (g *LinkGraph) Pages() []string {
	g.mu.RLock()
	defer g.mu.RUnlock()

	pages := make([]string, 0, len(g.pages))
	for page := range g.pages {
		pages = append(pages, page)
	}
	return pages
}

// OutgoingLinks returns outgoing links for a page
func (g *LinkGraph) OutgoingLinks(page string) []string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.outboundLinks[page]
}

// IncomingLinks returns incoming links for a page
func (g *LinkGraph) IncomingLinks(page string) []string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.inboundLinks[page]
}

// OutgoingCount returns the count of outgoing links
func (g *LinkGraph) OutgoingCount(page string) int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.outboundLinks[page])
}

// PageRankCalculator calculates PageRank scores
type PageRankCalculator struct {
	damping    float64
	iterations int
	tolerance  float64
	redis      *redis.Client
}

// NewPageRankCalculator creates a new PageRank calculator
func NewPageRankCalculator(redisAddr, password string, db int) (*PageRankCalculator, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     redisAddr,
		Password: password,
		DB:       db,
	})

	ctx := context.Background()
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &PageRankCalculator{
		damping:    0.85,
		iterations: 20,
		tolerance:  0.001,
		redis:      client,
	}, nil
}

// Calculate calculates PageRank scores for all pages
func (c *PageRankCalculator) Calculate(graph *LinkGraph) (map[string]float64, error) {
	pages := graph.Pages()
	n := len(pages)

	if n == 0 {
		return map[string]float64{}, nil
	}

	log.Printf("Calculating PageRank for %d pages", n)
	startTime := time.Now()

	// Initialize PageRank scores
	scores := make(map[string]float64)
	initialScore := 1.0 / float64(n)
	for _, page := range pages {
		scores[page] = initialScore
	}

	// Iterate until convergence or max iterations
	var converged bool
	for iteration := 0; iteration < c.iterations && !converged; iteration++ {
		newScores := make(map[string]float64)
		totalDiff := 0.0

		// Calculate new scores
		for _, page := range pages {
			rankSum := 0.0

			// Sum incoming PageRank
			for _, incoming := range graph.IncomingLinks(page) {
				outCount := graph.OutgoingCount(incoming)
				if outCount > 0 {
					rankSum += scores[incoming] / float64(outCount)
				}
			}

			// Apply damping factor
			newScore := (1-c.damping)/float64(n) + c.damping*rankSum
			newScores[page] = newScore

			// Track convergence
			totalDiff += math.Abs(newScore - scores[page])
		}

		scores = newScores

		// Check convergence
		avgDiff := totalDiff / float64(n)
		if avgDiff < c.tolerance {
			converged = true
			log.Printf("PageRank converged after %d iterations (avg diff: %.6f)", iteration+1, avgDiff)
		}

		if iteration%5 == 0 {
			log.Printf("PageRank iteration %d/%d (avg diff: %.6f)", iteration+1, c.iterations, totalDiff/float64(n))
		}
	}

	// Normalize scores (optional)
	c.normalize(scores)

	elapsed := time.Since(startTime)
	log.Printf("PageRank calculation completed in %v", elapsed)

	return scores, nil
}

// CalculateIncremental updates PageRank for new/changed pages
func (c *PageRankCalculator) CalculateIncremental(graph *LinkGraph, changedPages []string) (map[string]float64, error) {
	// In production, implement incremental PageRank
	// For now, recalculate all
	return c.Calculate(graph)
}

// StoreScores stores PageRank scores in Redis
func (c *PageRankCalculator) StoreScores(scores map[string]float64) error {
	ctx := context.Background()
	pipe := c.redis.Pipeline()

	for url, score := range scores {
		key := fmt.Sprintf("pagerank:%s", hashURL(url))
		pipe.Set(ctx, key, score, 0)
	}

	_, err := pipe.Exec(ctx)
	return err
}

// GetScore retrieves PageRank score for a URL
func (c *PageRankCalculator) GetScore(url string) (float64, error) {
	ctx := context.Background()
	key := fmt.Sprintf("pagerank:%s", hashURL(url))

	val, err := c.redis.Get(ctx, key).Float64()
	if err == redis.Nil {
		return 0.0, nil
	}
	return val, err
}

// GetScores retrieves PageRank scores for multiple URLs
func (c *PageRankCalculator) GetScores(urls []string) (map[string]float64, error) {
	ctx := context.Background()
	scores := make(map[string]float64)

	for _, url := range urls {
		score, err := c.GetScore(url)
		if err != nil {
			log.Printf("Failed to get score for %s: %v", url, err)
			continue
		}
		scores[url] = score
	}

	return scores, nil
}

// normalize normalizes PageRank scores to sum to 1
func (c *PageRankCalculator) normalize(scores map[string]float64) {
	total := 0.0
	for _, score := range scores {
		total += score
	}

	if total > 0 {
		for url := range scores {
			scores[url] /= total
		}
	}
}

// Close closes the Redis connection
func (c *PageRankCalculator) Close() error {
	return c.redis.Close()
}

// hashURL creates a hash of the URL for storage
func hashURL(url string) string {
	h := 0
	for i := 0; i < len(url); i++ {
		h = 31*h + int(url[i])
	}
	return fmt.Sprintf("%d", h)
}

// BuildGraphFromPages builds a link graph from crawled pages
func BuildGraphFromPages(pages []*models.CrawledPage, links []*models.PageLink) *LinkGraph {
	graph := NewLinkGraph()

	// Add all pages
	for _, page := range pages {
		graph.AddPage(page.URL)
	}

	// Add all links
	for _, link := range links {
		// Find source page URL
		var sourceURL string
		for _, page := range pages {
			if page.ID == link.SourcePageID {
				sourceURL = page.URL
				break
			}
		}
		if sourceURL != "" {
			graph.AddLink(sourceURL, link.TargetURL)
		}
	}

	return graph
}
