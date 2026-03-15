package main

import (
	"flag"
	"log"
	"time"

	"github.com/itxLikhith/intent-engine/go-crawler/internal/pagerank"
)

func main() {
	// Command-line flags
	redisAddr := flag.String("redis", "localhost:6379", "Redis address")
	redisPassword := flag.String("redis-password", "", "Redis password")
	redisDB := flag.Int("redis-db", 0, "Redis database")
	iterations := flag.Int("iterations", 20, "Number of PageRank iterations")
	damping := flag.Float64("damping", 0.85, "Damping factor")
	flag.Parse()

	log.Println("Starting PageRank calculator...")
	startTime := time.Now()

	// Create PageRank calculator
	calc, err := pagerank.NewPageRankCalculator(*redisAddr, *redisPassword, *redisDB)
	if err != nil {
		log.Fatalf("Failed to create PageRank calculator: %v", err)
	}
	defer calc.Close()

	// In production, load link graph from database
	// For now, create a sample graph
	graph := createSampleGraph()

	log.Printf("Graph has %d pages", len(graph.Pages()))

	// Calculate PageRank
	scores, err := calc.Calculate(graph)
	if err != nil {
		log.Fatalf("PageRank calculation failed: %v", err)
	}

	// Store scores
	if err := calc.StoreScores(scores); err != nil {
		log.Printf("Failed to store scores: %v", err)
	}

	// Print top pages
	log.Println("\nTop 10 pages by PageRank:")
	printTopPages(scores, 10)

	elapsed := time.Since(startTime)
	log.Printf("\nPageRank calculation completed in %v", elapsed)
}

// createSampleGraph creates a sample link graph for testing
func createSampleGraph() *pagerank.LinkGraph {
	graph := pagerank.NewLinkGraph()

	// Add sample pages and links
	// This is a simplified example - in production, load from database
	pages := []string{
		"https://example.com",
		"https://example.com/page1",
		"https://example.com/page2",
		"https://golang.org",
		"https://golang.org/doc",
	}

	for _, page := range pages {
		graph.AddPage(page)
	}

	// Add sample links
	graph.AddLink("https://example.com", "https://example.com/page1")
	graph.AddLink("https://example.com", "https://example.com/page2")
	graph.AddLink("https://example.com/page1", "https://golang.org")
	graph.AddLink("https://example.com/page2", "https://golang.org/doc")
	graph.AddLink("https://golang.org", "https://golang.org/doc")
	graph.AddLink("https://golang.org/doc", "https://example.com")

	return graph
}

// printTopPages prints the top N pages by PageRank score
func printTopPages(scores map[string]float64, n int) {
	// Sort and print top N
	// Simplified implementation
	count := 0
	for url, score := range scores {
		if count >= n {
			break
		}
		log.Printf("  %.6f - %s", score, url)
		count++
	}
}
