package indexer

import (
	"os"
	"testing"
	"time"

	"github.com/itxLikhith/intent-engine/go-crawler/pkg/models"
)

func TestNewSearchIndexer(t *testing.T) {
	// Create temp directory for test
	tmpDir := t.TempDir()
	config := &IndexerConfig{
		IndexPath: tmpDir + "/test_index",
	}

	idx, err := NewSearchIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer idx.Close()

	// Verify indexer is not nil
	if idx == nil {
		t.Fatal("Indexer is nil")
	}
}

func TestIndexDocument(t *testing.T) {
	tmpDir := t.TempDir()
	config := &IndexerConfig{
		IndexPath: tmpDir + "/test_index",
	}

	idx, err := NewSearchIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer idx.Close()

	// Create test page
	page := &models.CrawledPage{
		ID:              "test-1",
		URL:             "https://example.com/test",
		Title:           "Test Page",
		Content:         "This is a test page with some content about Go programming.",
		MetaDescription: "A test page",
		PageRank:        0.5,
		CrawledAt:       time.Now(),
	}

	// Index the page
	err = idx.IndexDocument(page)
	if err != nil {
		t.Fatalf("Failed to index document: %v", err)
	}

	// Verify stats
	stats, err := idx.GetStats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.TotalDocuments != 1 {
		t.Errorf("Expected 1 document, got %d", stats.TotalDocuments)
	}
}

func TestSearch(t *testing.T) {
	tmpDir := t.TempDir()
	config := &IndexerConfig{
		IndexPath: tmpDir + "/test_index",
	}

	idx, err := NewSearchIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer idx.Close()

	// Index multiple test pages
	pages := []*models.CrawledPage{
		{
			ID:      "test-1",
			URL:     "https://example.com/go",
			Title:   "Go Programming",
			Content: "Go is a programming language designed at Google.",
			PageRank: 0.8,
		},
		{
			ID:      "test-2",
			URL:     "https://example.com/python",
			Title:   "Python Programming",
			Content: "Python is a high-level programming language.",
			PageRank: 0.6,
		},
		{
			ID:      "test-3",
			URL:     "https://example.com/rust",
			Title:   "Rust Programming",
			Content: "Rust is a systems programming language.",
			PageRank: 0.7,
		},
	}

	for _, page := range pages {
		if err := idx.IndexDocument(page); err != nil {
			t.Fatalf("Failed to index document: %v", err)
		}
	}

	// Search for "programming"
	response, err := idx.Search("programming", 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if response.TotalResults == 0 {
		t.Error("Expected results, got 0")
	}

	if response.Query != "programming" {
		t.Errorf("Expected query 'programming', got '%s'", response.Query)
	}

	// Verify processing time is reasonable
	if response.ProcessingTimeMs < 0 {
		t.Error("Processing time should be positive")
	}
}

func TestIndexBatch(t *testing.T) {
	tmpDir := t.TempDir()
	config := &IndexerConfig{
		IndexPath: tmpDir + "/test_index",
	}

	idx, err := NewSearchIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer idx.Close()

	// Create test pages
	pages := []*models.CrawledPage{
		{
			ID:      "batch-1",
			URL:     "https://example.com/1",
			Title:   "Page 1",
			Content: "Content for page 1",
			PageRank: 0.5,
		},
		{
			ID:      "batch-2",
			URL:     "https://example.com/2",
			Title:   "Page 2",
			Content: "Content for page 2",
			PageRank: 0.6,
		},
		{
			ID:      "batch-3",
			URL:     "https://example.com/3",
			Title:   "Page 3",
			Content: "Content for page 3",
			PageRank: 0.7,
		},
	}

	// Index batch
	indexed, err := idx.IndexBatch(pages)
	if err != nil {
		t.Fatalf("Batch indexing failed: %v", err)
	}

	if indexed != len(pages) {
		t.Errorf("Expected %d indexed, got %d", len(pages), indexed)
	}

	// Verify stats
	stats, err := idx.GetStats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.TotalDocuments != int64(len(pages)) {
		t.Errorf("Expected %d documents, got %d", len(pages), stats.TotalDocuments)
	}
}

func TestDeleteDocument(t *testing.T) {
	tmpDir := t.TempDir()
	config := &IndexerConfig{
		IndexPath: tmpDir + "/test_index",
	}

	idx, err := NewSearchIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer idx.Close()

	// Index a document
	page := &models.CrawledPage{
		ID:      "delete-test",
		URL:     "https://example.com/delete",
		Title:   "Delete Me",
		Content: "This page will be deleted",
		PageRank: 0.5,
	}

	if err := idx.IndexDocument(page); err != nil {
		t.Fatalf("Failed to index document: %v", err)
	}

	// Delete the document
	err = idx.DeleteDocument(page.ID)
	if err != nil {
		t.Fatalf("Failed to delete document: %v", err)
	}

	// Verify deletion
	stats, err := idx.GetStats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.TotalDocuments != 0 {
		t.Errorf("Expected 0 documents after deletion, got %d", stats.TotalDocuments)
	}
}

func BenchmarkIndexDocument(b *testing.B) {
	tmpDir := b.TempDir()
	config := &IndexerConfig{
		IndexPath: tmpDir + "/bench_index",
	}

	idx, err := NewSearchIndexer(config)
	if err != nil {
		b.Fatalf("Failed to create indexer: %v", err)
	}
	defer idx.Close()

	page := &models.CrawledPage{
		ID:      "bench-1",
		URL:     "https://example.com/bench",
		Title:   "Benchmark Page",
		Content: "This is a benchmark test page with substantial content for testing performance.",
		PageRank: 0.5,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := idx.IndexDocument(page); err != nil {
			b.Fatalf("Index failed: %v", err)
		}
	}
}

func BenchmarkSearch(b *testing.B) {
	tmpDir := b.TempDir()
	config := &IndexerConfig{
		IndexPath: tmpDir + "/bench_index",
	}

	idx, err := NewSearchIndexer(config)
	if err != nil {
		b.Fatalf("Failed to create indexer: %v", err)
	}
	defer idx.Close()

	// Index 1000 documents
	for i := 0; i < 1000; i++ {
		page := &models.CrawledPage{
			ID:      fmt.Sprintf("bench-%d", i),
			URL:     fmt.Sprintf("https://example.com/%d", i),
			Title:   fmt.Sprintf("Page %d", i),
			Content: fmt.Sprintf("Content for page %d with some keywords", i),
			PageRank: float64(i) / 1000.0,
		}
		idx.IndexDocument(page)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := idx.Search("content", 20)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}
