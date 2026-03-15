package storage

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/dgraph-io/badger/v4"
	_ "github.com/lib/pq"
	"github.com/itxLikhith/intent-engine/go-crawler/pkg/models"
)

// Storage manages both PostgreSQL and BadgerDB
type Storage struct {
	postgres *sql.DB
	badger   *badger.DB
}

// StorageConfig holds storage configuration
type StorageConfig struct {
	PostgresDSN string
	BadgerPath  string
}

// NewStorage creates a new storage instance
func NewStorage(config *StorageConfig) (*Storage, error) {
	// Connect to PostgreSQL
	postgres, err := sql.Open("postgres", config.PostgresDSN)
	if err != nil {
		return nil, fmt.Errorf("failed to open PostgreSQL: %w", err)
	}

	// Test connection
	if err := postgres.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping PostgreSQL: %w", err)
	}

	// Set connection pool settings
	postgres.SetMaxOpenConns(25)
	postgres.SetMaxIdleConns(5)
	postgres.SetConnMaxLifetime(5 * time.Minute)

	log.Println("Connected to PostgreSQL")

	// Open BadgerDB
	badger, err := badger.Open(badger.DefaultOptions(config.BadgerPath))
	if err != nil {
		postgres.Close()
		return nil, fmt.Errorf("failed to open BadgerDB: %w", err)
	}

	log.Println("Opened BadgerDB")

	return &Storage{
		postgres: postgres,
		badger:   badger,
	}, nil
}

// SavePage saves a crawled page to both PostgreSQL and BadgerDB
func (s *Storage) SavePage(page *models.CrawledPage) error {
	// Start PostgreSQL transaction
	tx, err := s.postgres.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Insert or update page in PostgreSQL
	query := `
		INSERT INTO crawled_pages (
			id, url, final_url, title, content, meta_description, meta_keywords,
			status_code, content_type, content_length, load_time_ms,
			crawl_depth, outbound_links, inbound_links, pagerank,
			language, is_indexed, crawled_at, updated_at, next_crawl_at
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
		)
		ON CONFLICT (id) DO UPDATE SET
			final_url = EXCLUDED.final_url,
			title = EXCLUDED.title,
			content = EXCLUDED.content,
			meta_description = EXCLUDED.meta_description,
			meta_keywords = EXCLUDED.meta_keywords,
			status_code = EXCLUDED.status_code,
			content_type = EXCLUDED.content_type,
			content_length = EXCLUDED.content_length,
			load_time_ms = EXCLUDED.load_time_ms,
			crawl_depth = EXCLUDED.crawl_depth,
			outbound_links = EXCLUDED.outbound_links,
			updated_at = EXCLUDED.updated_at
	`

	_, err = tx.Exec(query,
		page.ID, page.URL, page.FinalURL, page.Title, page.Content,
		page.MetaDescription, page.MetaKeywords, page.StatusCode,
		page.ContentType, page.ContentLength, page.LoadTimeMs,
		page.CrawlDepth, page.OutboundLinks, page.InboundLinks, page.PageRank,
		page.Language, page.IsIndexed, page.CrawledAt, page.UpdatedAt, page.NextCrawlAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert page: %w", err)
	}

	// Store raw HTML in BadgerDB (if available)
	if page.HTMLContent != "" {
		err = s.badger.Update(func(txn *badger.Txn) error {
			key := []byte("html:" + page.ID)
			return txn.Set(key, []byte(page.HTMLContent))
		})
		if err != nil {
			return fmt.Errorf("failed to store HTML in BadgerDB: %w", err)
		}
	}

	// Commit PostgreSQL transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// SaveLinks saves page links to PostgreSQL
func (s *Storage) SaveLinks(links []*models.PageLink) error {
	if len(links) == 0 {
		return nil
	}

	tx, err := s.postgres.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	query := `
		INSERT INTO page_links (source_page_id, target_url, anchor_text, link_type, created_at)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (source_page_id, target_url) DO NOTHING
	`

	for _, link := range links {
		_, err := tx.Exec(query,
			link.SourcePageID, link.TargetURL, link.AnchorText,
			link.LinkType, link.CreatedAt,
		)
		if err != nil {
			return fmt.Errorf("failed to insert link: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// GetPage retrieves a page by ID
func (s *Storage) GetPage(id string) (*models.CrawledPage, error) {
	query := `
		SELECT id, url, final_url, title, content, meta_description, meta_keywords,
		       status_code, content_type, content_length, load_time_ms,
		       crawl_depth, outbound_links, inbound_links, pagerank,
		       language, is_indexed, crawled_at, updated_at, next_crawl_at
		FROM crawled_pages
		WHERE id = $1
	`

	page := &models.CrawledPage{}
	err := s.postgres.QueryRow(query, id).Scan(
		&page.ID, &page.URL, &page.FinalURL, &page.Title, &page.Content,
		&page.MetaDescription, &page.MetaKeywords, &page.StatusCode,
		&page.ContentType, &page.ContentLength, &page.LoadTimeMs,
		&page.CrawlDepth, &page.OutboundLinks, &page.InboundLinks, &page.PageRank,
		&page.Language, &page.IsIndexed, &page.CrawledAt, &page.UpdatedAt, &page.NextCrawlAt,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to query page: %w", err)
	}

	return page, nil
}

// GetPageHTML retrieves raw HTML from BadgerDB
func (s *Storage) GetPageHTML(id string) (string, error) {
	var html string
	err := s.badger.View(func(txn *badger.Txn) error {
		key := []byte("html:" + id)
		item, err := txn.Get(key)
		if err != nil {
			return err
		}
		return item.Value(func(val []byte) error {
			html = string(val)
			return nil
		})
	})
	if err != nil {
		return "", fmt.Errorf("failed to get HTML from BadgerDB: %w", err)
	}
	return html, nil
}

// UpdatePageRank updates PageRank score for a page
func (s *Storage) UpdatePageRank(id string, score float64) error {
	query := `UPDATE crawled_pages SET pagerank = $1, updated_at = $2 WHERE id = $3`
	_, err := s.postgres.Exec(query, score, time.Now(), id)
	return err
}

// GetStats returns crawl statistics
func (s *Storage) GetStats() (*models.CrawlStats, error) {
	query := `
		SELECT 
			COUNT(*) as total_pages,
			COUNT(CASE WHEN is_indexed = true THEN 1 END) as pages_indexed,
			AVG(load_time_ms) as avg_load_time
		FROM crawled_pages
	`

	stats := &models.CrawlStats{}
	err := s.postgres.QueryRow(query).Scan(
		&stats.TotalPages, &stats.PagesIndexed, &stats.AvgLoadTimeMs,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to query stats: %w", err)
	}

	return stats, nil
}

// Close closes both database connections
func (s *Storage) Close() error {
	var errs []error

	if err := s.badger.Close(); err != nil {
		errs = append(errs, fmt.Errorf("failed to close BadgerDB: %w", err))
	}

	if err := s.postgres.Close(); err != nil {
		errs = append(errs, fmt.Errorf("failed to close PostgreSQL: %w", err))
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing storage: %v", errs)
	}

	return nil
}

// UpdateStats updates daily crawl statistics
func (s *Storage) UpdateStats(pagesCrawled, pagesFailed, pagesIndexed, linksExtracted int) error {
	query := `
		INSERT INTO crawl_stats (stat_date, pages_crawled, pages_failed, pages_indexed, links_extracted, updated_at)
		VALUES (CURRENT_DATE, $1, $2, $3, $4, NOW())
		ON CONFLICT (stat_date) DO UPDATE SET
			pages_crawled = crawl_stats.pages_crawled + EXCLUDED.pages_crawled,
			pages_failed = crawl_stats.pages_failed + EXCLUDED.pages_failed,
			pages_indexed = crawl_stats.pages_indexed + EXCLUDED.pages_indexed,
			links_extracted = crawl_stats.links_extracted + EXCLUDED.links_extracted,
			updated_at = NOW()
	`
	_, err := s.postgres.Exec(query, pagesCrawled, pagesFailed, pagesIndexed, linksExtracted)
	return err
}

// GetUnindexedPages returns pages that haven't been indexed yet
func (s *Storage) GetUnindexedPages(limit int) ([]*models.CrawledPage, error) {
	query := `
		SELECT id, url, final_url, title, content, meta_description, meta_keywords,
		       status_code, content_type, content_length, load_time_ms,
		       crawl_depth, outbound_links, inbound_links, pagerank,
		       language, is_indexed, crawled_at, updated_at, next_crawl_at
		FROM crawled_pages
		WHERE is_indexed = false
		ORDER BY crawled_at ASC
		LIMIT $1
	`

	rows, err := s.postgres.Query(query, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query unindexed pages: %w", err)
	}
	defer rows.Close()

	var pages []*models.CrawledPage
	for rows.Next() {
		page := &models.CrawledPage{}
		err := rows.Scan(
			&page.ID, &page.URL, &page.FinalURL, &page.Title, &page.Content,
			&page.MetaDescription, &page.MetaKeywords, &page.StatusCode,
			&page.ContentType, &page.ContentLength, &page.LoadTimeMs,
			&page.CrawlDepth, &page.OutboundLinks, &page.InboundLinks, &page.PageRank,
			&page.Language, &page.IsIndexed, &page.CrawledAt, &page.UpdatedAt, &page.NextCrawlAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan page: %w", err)
		}
		pages = append(pages, page)
	}

	return pages, rows.Err()
}

// MarkAsIndexed marks a page as indexed
func (s *Storage) MarkAsIndexed(id string) error {
	query := `UPDATE crawled_pages SET is_indexed = true, updated_at = $1 WHERE id = $2`
	_, err := s.postgres.Exec(query, time.Now(), id)
	return err
}

// SerializeTermFrequencies converts term frequencies to JSON
func SerializeTermFrequencies(terms map[string]int) (string, error) {
	data, err := json.Marshal(terms)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// DeserializeTermFrequencies converts JSON to term frequencies
func DeserializeTermFrequencies(data string) (map[string]int, error) {
	var terms map[string]int
	err := json.Unmarshal([]byte(data), &terms)
	return terms, err
}
