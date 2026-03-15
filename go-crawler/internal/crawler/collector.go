package crawler

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/gocolly/colly/v2"
	"github.com/itxLikhith/intent-engine/go-crawler/internal/frontier"
	"github.com/itxLikhith/intent-engine/go-crawler/internal/storage"
	"github.com/itxLikhith/intent-engine/go-crawler/pkg/models"
)

// Crawler represents the web crawler
type Crawler struct {
	collector  *colly.Collector
	config     *Config
	store      *storage.Storage
	frontier   *frontier.URLFrontier
	stats      *CrawlStats
	statsMutex sync.RWMutex
	ctx        context.Context
	cancel     context.CancelFunc
}

// CrawlStats holds crawling statistics
type CrawlStats struct {
	PagesCrawled  int64
	PagesSuccess  int64
	PagesFailed   int64
	LinksFound    int64
	Duplicates    int64
	StartTime     time.Time
	Duration      time.Duration
}

// NewCrawler creates a new crawler instance
func NewCrawler(cfg *Config, store *storage.Storage, redisAddr string) (*Crawler, error) {
	// Initialize frontier
	f, err := frontier.NewURLFrontier(redisAddr, "", 0)
	if err != nil {
		return nil, fmt.Errorf("failed to create frontier: %w", err)
	}

	// Create Colly collector
	c := colly.NewCollector(
		colly.MaxDepth(cfg.MaxDepth),
		colly.MaxBodySize(10 * 1024 * 1024), // 10MB
		colly.UserAgent(cfg.UserAgent),
		colly.Async(true),
	)

	// Set concurrency limits
	c.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: cfg.MaxConcurrentRequests,
		Delay:       cfg.CrawlDelay,
		RandomDelay: cfg.RandomDelay,
	})

	// Configure timeouts
	c.SetRequestTimeout(cfg.RequestTimeout)

	ctx, cancel := context.WithCancel(context.Background())

	crawler := &Crawler{
		collector: c,
		config:    cfg,
		store:     store,
		frontier:  f,
		stats: &CrawlStats{
			StartTime: time.Now(),
		},
		ctx:    ctx,
		cancel: cancel,
	}

	// Setup callbacks
	crawler.setupCallbacks()

	return crawler, nil
}

// setupCallbacks configures Colly callbacks
func (c *Crawler) setupCallbacks() {
	// Before request
	c.collector.OnRequest(func(r *colly.Request) {
		log.Printf("Visiting: %s", r.URL.String())
	})

	// On response
	c.collector.OnResponse(func(r *colly.Response) {
		c.statsMutex.Lock()
		c.stats.PagesCrawled++
		c.statsMutex.Unlock()

		// Parse HTML
		doc, err := goquery.NewDocumentFromReader(strings.NewReader(string(r.Body)))
		if err != nil {
			log.Printf("Failed to parse HTML: %v", err)
			return
		}

		// Extract page data
		page := c.extractPageData(r, doc)

		// Store page
		if err := c.store.SavePage(page); err != nil {
			log.Printf("Failed to save page: %v", err)
			return
		}

		c.statsMutex.Lock()
		c.stats.PagesSuccess++
		c.statsMutex.Unlock()

		// Extract and queue links
		c.extractLinks(r, doc)
	})

	// On error
	c.collector.OnError(func(r *colly.Response, err error) {
		c.statsMutex.Lock()
		c.stats.PagesFailed++
		c.statsMutex.Unlock()
		log.Printf("Error crawling %s: %v", r.Request.URL, err)
	})

	// On link discovered
	c.collector.OnHTML("a[href]", func(e *colly.HTMLElement) {
		// Links are handled in extractLinks
	})
}

// extractPageData extracts page information from response
func (c *Crawler) extractPageData(r *colly.Response, doc *goquery.Document) *models.CrawledPage {
	title := doc.Find("title").First().Text()
	description := ""
	doc.Find("meta[name=description]").Each(func(i int, s *goquery.Selection) {
		description, _ = s.Attr("content")
	})

	content := doc.Find("body").Text()
	content = strings.TrimSpace(strings.ReplaceAll(content, "\n", " "))

	// Limit content length
	if len(content) > 50000 {
		content = content[:50000]
	}

	htmlContent := string(r.Body)

	return &models.CrawledPage{
		URL:             r.Request.URL.String(),
		FinalURL:        r.Request.URL.String(),
		Title:           title,
		Content:         content,
		MetaDescription: description,
		StatusCode:      r.StatusCode,
		ContentType:     r.Headers.Get("Content-Type"),
		ContentLength:   len(r.Body),
		HTMLContent:     htmlContent,
		CrawledAt:       time.Now(),
		UpdatedAt:       time.Now(),
		IsIndexed:       false,
		Language:        "en",
	}
}

// extractLinks extracts links from page and queues them
func (c *Crawler) extractLinks(r *colly.Response, doc *goquery.Document) {
	linksFound := 0
	var linksToSave []*models.PageLink

	doc.Find("a[href]").Each(func(i int, s *goquery.Selection) {
		href, exists := s.Attr("href")
		if !exists {
			return
		}

		// Resolve relative URLs
		baseURL, _ := url.Parse(r.Request.URL.String())
		linkURL, err := baseURL.Parse(href)
		if err != nil {
			return
		}

		// Skip invalid URLs
		if linkURL.Scheme == "" || linkURL.Host == "" {
			return
		}

		// Skip blocked domains
		if c.isBlockedDomain(linkURL.Host) {
			return
		}

		// Skip non-HTML content
		if c.skipURL(linkURL.String()) {
			return
		}

		// Add to frontier
		link := &models.PageLink{
			TargetURL:  linkURL.String(),
			AnchorText: strings.TrimSpace(s.Text()),
			LinkType:   "dofollow",
			CreatedAt:  time.Now(),
		}

		linksToSave = append(linksToSave, link)
		linksFound++
	})

	// Save links in batch - get the page ID from the database
	if len(linksToSave) > 0 {
		// Get the integer page ID from the database
		pageIntID, err := c.store.GetPageIDByURL(r.Request.URL.String())
		if err != nil {
			log.Printf("Failed to get page ID for %s: %v", r.Request.URL.String(), err)
			// Skip saving links if we can't get the page ID
			return
		}

		if err := c.store.SaveLinks(linksToSave, pageIntID); err != nil {
			log.Printf("Failed to save links: %v", err)
		}
	}

	// Add URLs to frontier
	urls := make([]string, len(linksToSave))
	for i, link := range linksToSave {
		urls[i] = link.TargetURL
	}
	if len(urls) > 0 {
		if _, err := c.frontier.AddURLs(urls, 1, 1); err != nil {
			log.Printf("Failed to add URLs to frontier: %v", err)
		}
	}

	c.statsMutex.Lock()
	c.stats.LinksFound += int64(linksFound)
	c.statsMutex.Unlock()

	log.Printf("Found %d links on %s", linksFound, r.Request.URL)
}

// isBlockedDomain checks if domain is blocked
func (c *Crawler) isBlockedDomain(domain string) bool {
	for _, blocked := range c.config.BlockedDomains {
		if strings.Contains(domain, blocked) {
			return true
		}
	}
	return false
}

// skipURL checks if URL should be skipped
func (c *Crawler) skipURL(urlStr string) bool {
	skipPatterns := []string{
		".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
		".pdf", ".doc", ".docx", ".xls", ".xlsx",
		".zip", ".rar", ".tar", ".gz",
		".mp3", ".mp4", ".avi", ".mov",
		".css", ".js", ".woff", ".woff2",
		"login", "signup", "register", "checkout",
	}

	lowerURL := strings.ToLower(urlStr)
	for _, pattern := range skipPatterns {
		if strings.Contains(lowerURL, pattern) {
			return true
		}
	}
	return false
}

// Seed adds initial URL to crawl queue
func (c *Crawler) Seed(urlStr string) error {
	log.Printf("Seeding URL: %s", urlStr)
	_, err := c.frontier.AddURLs([]string{urlStr}, 10, 0) // High priority for seed
	return err
}

// Crawl starts the crawling process
func (c *Crawler) Crawl(ctx context.Context) (*CrawlStats, error) {
	log.Println("Starting crawl process...")

	// Create ticker for periodic checks
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("Crawl cancelled")
			return c.getStats(), nil
		case <-c.ctx.Done():
			log.Println("Crawl cancelled (internal)")
			return c.getStats(), nil
		case <-ticker.C:
		}

		// Check if we've reached max pages
		c.statsMutex.RLock()
		if c.stats.PagesCrawled >= int64(c.config.MaxPages) {
			c.statsMutex.RUnlock()
			log.Println("Reached max pages limit")
			break
		}
		c.statsMutex.RUnlock()

		// Get next URL from frontier
		urlItem, err := c.frontier.GetNextURL()
		if err != nil {
			log.Printf("Failed to get next URL: %v", err)
			time.Sleep(time.Second)
			continue
		}

		if urlItem == nil {
			// Queue is empty, wait a bit and check again
			time.Sleep(500 * time.Millisecond)
			continue
		}

		// Visit URL
		if err := c.collector.Visit(urlItem.URL); err != nil {
			log.Printf("Failed to visit %s: %v", urlItem.URL, err)
			// Mark as failed
			if markErr := c.frontier.MarkFailed(urlItem.URL, err.Error()); markErr != nil {
				log.Printf("Failed to mark URL as failed: %v", markErr)
			}
		}
	}

	return c.getStats(), nil
}

// Stop stops the crawler gracefully
func (c *Crawler) Stop() {
	log.Println("Stopping crawler...")
	if c.cancel != nil {
		c.cancel()
	}
	// Colly collector doesn't have a Stop method
	// The context cancellation will stop the crawl loop
}

// getStats returns current crawl statistics
func (c *Crawler) getStats() *CrawlStats {
	c.statsMutex.RLock()
	defer c.statsMutex.RUnlock()

	stats := *c.stats
	stats.Duration = time.Since(stats.StartTime)
	return &stats
}

// GetStats returns crawl statistics
func (c *Crawler) GetStats() *CrawlStats {
	return c.getStats()
}

// generatePageID generates a unique ID for a page
func generatePageID(urlStr string) string {
	// Simple hash-based ID - in production use proper hashing
	h := 0
	for i := 0; i < len(urlStr); i++ {
		h = 31*h + int(urlStr[i])
	}
	return fmt.Sprintf("page_%d", h)
}
