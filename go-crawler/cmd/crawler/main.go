package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/itxLikhith/intent-engine/go-crawler/internal/crawler"
	"github.com/itxLikhith/intent-engine/go-crawler/internal/frontier"
	"github.com/itxLikhith/intent-engine/go-crawler/internal/storage"
)

func main() {
	// Command-line flags
	redisAddr := flag.String("redis", "localhost:6379", "Redis address")
	postgresDSN := flag.String("postgres", "", "PostgreSQL DSN")
	badgerPath := flag.String("badger", "./data/badger", "Badger DB path")
	seedURLs := flag.String("seed", "", "Seed URLs (comma-separated)")
	maxPages := flag.Int("max-pages", 100, "Maximum pages to crawl")
	maxDepth := flag.Int("max-depth", 2, "Maximum crawl depth")
	concurrency := flag.Int("concurrency", 3, "Number of concurrent crawlers")
	delay := flag.Int("delay", 2000, "Delay between requests in ms")
	configFile := flag.String("config", "", "Config file path")
	flag.Parse()

	// Setup logging
	log.Println("Starting Go Crawler...")
	log.Printf("Redis: %s", *redisAddr)
	log.Printf("PostgreSQL: %s", *postgresDSN)
	log.Printf("Badger: %s", *badgerPath)
	log.Printf("Seed URLs: %s", *seedURLs)
	log.Printf("Max Pages: %d", *maxPages)
	log.Printf("Max Depth: %d", *maxDepth)
	log.Printf("Concurrency: %d", *concurrency)
	log.Printf("Delay: %dms", *delay)

	// Validate required parameters
	if *postgresDSN == "" {
		log.Fatal("PostgreSQL DSN is required")
	}

	// Initialize storage
	storeCfg := &storage.StorageConfig{
		PostgresDSN: *postgresDSN,
		BadgerPath:  *badgerPath,
	}
	store, err := storage.NewStorage(storeCfg)
	if err != nil {
		log.Fatalf("Failed to initialize storage: %v", err)
	}
	defer store.Close()

	// Initialize Redis frontier
	_, err = frontier.NewURLFrontier(*redisAddr, "", 0)
	if err != nil {
		log.Fatalf("Failed to initialize URL frontier: %v", err)
	}

	// Parse seed URLs
	seeds := parseCommaSeparated(*seedURLs)

	// Load config file if provided
	if *configFile != "" {
		log.Printf("Loading config from: %s", *configFile)
		// Config loading logic can be added here
	}

	// Create crawler config
	cfg := &crawler.Config{
		MaxPages:              *maxPages,
		MaxDepth:              *maxDepth,
		MaxConcurrentRequests: *concurrency,
		CrawlDelay:            time.Duration(*delay) * time.Millisecond,
		UserAgent:             "IntentEngine-Crawler/1.0",
		PostgresDSN:           *postgresDSN,
		BadgerPath:            *badgerPath,
		RedisAddr:             *redisAddr,
	}

	// Initialize crawler
	c, err := crawler.NewCrawler(cfg, store, *redisAddr)
	if err != nil {
		log.Fatalf("Failed to initialize crawler: %v", err)
	}

	// Add seed URLs to queue
	if len(seeds) > 0 {
		for _, seed := range seeds {
			if err := c.Seed(seed); err != nil {
				log.Printf("Failed to seed URL %s: %v", seed, err)
			}
		}
		log.Printf("Added %d seed URLs to queue", len(seeds))
	}

	// Setup signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start crawling in background
	log.Println("Starting crawl...")
	done := make(chan error, 1)

	go func() {
		stats, err := c.Crawl(ctx)
		if err != nil {
			log.Printf("Crawler error: %v", err)
			done <- err
			return
		}
		log.Printf("Crawl completed: Pages=%d, Success=%d, Failed=%d, Links=%d",
			stats.PagesCrawled, stats.PagesSuccess, stats.PagesFailed, stats.LinksFound)
		done <- nil
	}()

	// Wait for completion or signal
	select {
	case err := <-done:
		if err == nil {
			log.Println("Crawl finished successfully")
		}
	case <-sigChan:
		log.Println("Received shutdown signal, stopping crawler...")
		cancel()
		<-done
		log.Println("Crawler stopped gracefully")
	}
}

// parseCommaSeparated parses a comma-separated string into a slice of trimmed strings
func parseCommaSeparated(s string) []string {
	if s == "" {
		return nil
	}
	result := []string{}
	parts := strings.Split(s, ",")
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}
