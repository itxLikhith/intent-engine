package models

import "time"

// CrawledPage represents a crawled web page
type CrawledPage struct {
	ID              string    `json:"id"`
	URL             string    `json:"url"`
	FinalURL        string    `json:"final_url"`
	Title           string    `json:"title"`
	Content         string    `json:"content"`
	HTMLContent     string    `json:"-"` // Raw HTML, not serialized to JSON
	MetaDescription string    `json:"meta_description"`
	MetaKeywords    string    `json:"meta_keywords"`
	StatusCode      int       `json:"status_code"`
	ContentType     string    `json:"content_type"`
	ContentLength   int       `json:"content_length"`
	LoadTimeMs      float64   `json:"load_time_ms"`
	CrawlDepth      int       `json:"crawl_depth"`
	OutboundLinks   int       `json:"outbound_links"`
	InboundLinks    int       `json:"inbound_links"`
	PageRank        float64   `json:"pagerank"`
	Language        string    `json:"language"`
	IsIndexed       bool      `json:"is_indexed"`
	CrawledAt       time.Time `json:"crawled_at"`
	UpdatedAt       time.Time `json:"updated_at"`
	NextCrawlAt     time.Time `json:"next_crawl_at"`
}

// PageLink represents a link between pages
type PageLink struct {
	SourcePageID string    `json:"source_page_id"`
	TargetURL    string    `json:"target_url"`
	AnchorText   string    `json:"anchor_text"`
	LinkType     string    `json:"link_type"` // dofollow, nofollow
	CreatedAt    time.Time `json:"created_at"`
}

// SearchDocument represents an indexed document for search
type SearchDocument struct {
	ID              string         `json:"id"`
	PageID          string         `json:"page_id"`
	URL             string         `json:"url"`
	Title           string         `json:"title"`
	Content         string         `json:"content"`
	MetaDescription string         `json:"meta_description"`
	TermFrequencies map[string]int `json:"term_frequencies"`
	WordCount       int            `json:"word_count"`
	PageRank        float64        `json:"pagerank"`
	IndexedAt       time.Time      `json:"indexed_at"`
}

// SearchResult represents a search result
type SearchResult struct {
	PageID       string   `json:"page_id"`
	URL          string   `json:"url"`
	Title        string   `json:"title"`
	Snippet      string   `json:"snippet"`
	Score        float64  `json:"score"`
	PageRank     float64  `json:"pagerank"`
	MatchedTerms []string `json:"matched_terms"`
}

// CrawlQueueItem represents an item in the crawl queue
type CrawlQueueItem struct {
	ID         string    `json:"id"`
	URL        string    `json:"url"`
	Priority   int       `json:"priority"`
	Depth      int       `json:"depth"`
	Status     string    `json:"status"` // pending, crawling, completed, failed
	ScheduledAt time.Time `json:"scheduled_at"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
	Error      string    `json:"error,omitempty"`
}

// CrawlStats represents crawler statistics
type CrawlStats struct {
	TotalPages      int64   `json:"total_pages"`
	PagesCrawled    int64   `json:"pages_crawled"`
	PagesIndexed    int64   `json:"pages_indexed"`
	PagesFailed     int64   `json:"pages_failed"`
	QueueSize       int64   `json:"queue_size"`
	AvgLoadTimeMs   float64 `json:"avg_load_time_ms"`
	PagesPerSecond  float64 `json:"pages_per_second"`
	LastCrawlAt     time.Time `json:"last_crawl_at"`
}

// IndexStats represents indexer statistics
type IndexStats struct {
	TotalDocuments  int64   `json:"total_documents"`
	TotalTerms      int64   `json:"total_terms"`
	IndexSizeBytes  int64   `json:"index_size_bytes"`
	AvgDocLength    float64 `json:"avg_doc_length"`
	LastIndexedAt   time.Time `json:"last_indexed_at"`
}

// SearchRequest represents a search request
type SearchRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

// SearchResponse represents a search response
type SearchResponse struct {
	Query           string         `json:"query"`
	Results         []SearchResult `json:"results"`
	TotalResults    int            `json:"total_results"`
	ProcessingTimeMs float64       `json:"processing_time_ms"`
}

// SeedURLRequest represents a request to add seed URLs
type SeedURLRequest struct {
	URLs     []string `json:"urls"`
	Priority int      `json:"priority"`
	Depth    int      `json:"depth"`
}

// CrawlResponse represents a crawl operation response
type CrawlResponse struct {
	Success       bool   `json:"success"`
	Message       string `json:"message"`
	URLsAdded     int    `json:"urls_added"`
	QueuePosition int    `json:"queue_position,omitempty"`
}
