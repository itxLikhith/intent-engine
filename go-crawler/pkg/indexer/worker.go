package indexer

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/itxLikhith/intent-engine/go-crawler/internal/storage"
	"github.com/itxLikhith/intent-engine/go-crawler/pkg/models"
)

// IndexerWorker processes unindexed pages from PostgreSQL and indexes them
type IndexerWorker struct {
	storage    *storage.Storage
	indexer    *IntentIndexer
	batchSize  int
	interval   time.Duration
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
	isRunning  bool
	mu         sync.RWMutex
}

// IndexerWorkerConfig holds worker configuration
type IndexerWorkerConfig struct {
	BatchSize int           `yaml:"batch_size"`
	Interval  time.Duration `yaml:"interval"`
}

// DefaultIndexerWorkerConfig returns default worker configuration
func DefaultIndexerWorkerConfig() *IndexerWorkerConfig {
	return &IndexerWorkerConfig{
		BatchSize: 50,
		Interval:  30 * time.Second,
	}
}

// NewIndexerWorker creates a new indexer worker
func NewIndexerWorker(
	storage *storage.Storage,
	indexer *IntentIndexer,
	config *IndexerWorkerConfig,
) *IndexerWorker {
	if config == nil {
		config = DefaultIndexerWorkerConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &IndexerWorker{
		storage:   storage,
		indexer:   indexer,
		batchSize: config.BatchSize,
		interval:  config.Interval,
		ctx:       ctx,
		cancel:    cancel,
	}
}

// Start starts the background indexing worker
func (w *IndexerWorker) Start() error {
	w.mu.Lock()
	if w.isRunning {
		w.mu.Unlock()
		return fmt.Errorf("indexer worker already running")
	}
	w.isRunning = true
	w.mu.Unlock()

	log.Printf("Starting indexer worker (batch_size=%d, interval=%v)", w.batchSize, w.interval)

	w.wg.Add(1)
	go w.runLoop()

	return nil
}

// Stop stops the background indexing worker
func (w *IndexerWorker) Stop() error {
	w.mu.Lock()
	if !w.isRunning {
		w.mu.Unlock()
		return fmt.Errorf("indexer worker not running")
	}
	w.isRunning = false
	w.mu.Unlock()

	log.Println("Stopping indexer worker...")
	w.cancel()
	w.wg.Wait()

	log.Println("Indexer worker stopped")
	return nil
}

// IsRunning returns whether the worker is running
func (w *IndexerWorker) IsRunning() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.isRunning
}

// runLoop runs the main indexing loop
func (w *IndexerWorker) runLoop() {
	defer w.wg.Done()

	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()

	for {
		select {
		case <-w.ctx.Done():
			return
		case <-ticker.C:
			w.processBatch()
		}
	}
}

// processBatch processes a batch of unindexed pages
func (w *IndexerWorker) processBatch() {
	log.Println("Indexer worker: fetching unindexed pages...")

	// Fetch unindexed pages from PostgreSQL
	pages, err := w.storage.GetUnindexedPages(w.batchSize)
	if err != nil {
		log.Printf("Indexer worker error: failed to get unindexed pages: %v", err)
		return
	}

	if len(pages) == 0 {
		log.Println("Indexer worker: no unindexed pages found")
		return
	}

	log.Printf("Indexer worker: processing %d unindexed pages", len(pages))

	// Index pages in batch
	successCount, err := w.indexer.IndexDocuments(pages)
	if err != nil {
		log.Printf("Indexer worker error: failed to index documents: %v", err)
		return
	}

	// Mark pages as indexed in PostgreSQL
	markedCount := 0
	for _, page := range pages[:successCount] {
		if err := w.storage.MarkAsIndexed(page.ID); err != nil {
			log.Printf("Indexer worker warning: failed to mark page %s as indexed: %v", page.ID, err)
			continue
		}
		markedCount++
	}

	log.Printf("Indexer worker: indexed %d pages, marked %d as indexed", successCount, markedCount)

	// Update crawl stats
	if err := w.storage.UpdateStats(0, 0, successCount, 0); err != nil {
		log.Printf("Indexer worker warning: failed to update stats: %v", err)
	}
}

// IndexPageImmediately indexes a single page immediately (not waiting for batch)
func (w *IndexerWorker) IndexPageImmediately(page *models.CrawledPage) error {
	log.Printf("Indexing page immediately: %s", page.ID)

	// Index the page
	if err := w.indexer.IndexDocument(page); err != nil {
		return fmt.Errorf("failed to index document: %w", err)
	}

	// Mark as indexed
	if err := w.storage.MarkAsIndexed(page.ID); err != nil {
		return fmt.Errorf("failed to mark as indexed: %w", err)
	}

	log.Printf("Successfully indexed page: %s", page.ID)
	return nil
}

// IndexPagesBatch indexes multiple pages immediately
func (w *IndexerWorker) IndexPagesBatch(pages []*models.CrawledPage) (int, error) {
	log.Printf("Indexing batch of %d pages", len(pages))

	// Index pages
	successCount, err := w.indexer.IndexDocuments(pages)
	if err != nil {
		return successCount, fmt.Errorf("failed to index documents: %w", err)
	}

	// Mark as indexed
	markedCount := 0
	for _, page := range pages[:successCount] {
		if err := w.storage.MarkAsIndexed(page.ID); err != nil {
			log.Printf("Warning: failed to mark page %s as indexed: %v", page.ID, err)
			continue
		}
		markedCount++
	}

	log.Printf("Successfully indexed %d/%d pages, marked %d as indexed",
		successCount, len(pages), markedCount)

	return successCount, nil
}

// GetStats returns worker statistics
func (w *IndexerWorker) GetStats() map[string]interface{} {
	w.mu.RLock()
	defer w.mu.RUnlock()

	stats := map[string]interface{}{
		"is_running": w.isRunning,
		"batch_size": w.batchSize,
		"interval":   w.interval.String(),
	}

	// Get indexer stats
	if indexerStats, err := w.indexer.GetStats(); err == nil {
		stats["indexer"] = indexerStats
	}

	return stats
}

// TriggerIndexing triggers immediate indexing of all unindexed pages
// Useful for manual intervention or after bulk crawls
func (w *IndexerWorker) TriggerIndexing() (int, error) {
	log.Println("Triggering immediate indexing of all unindexed pages...")

	totalIndexed := 0
	for {
		// Check if context is cancelled
		select {
		case <-w.ctx.Done():
			return totalIndexed, fmt.Errorf("indexing cancelled")
		default:
		}

		// Fetch unindexed pages
		pages, err := w.storage.GetUnindexedPages(w.batchSize)
		if err != nil {
			return totalIndexed, fmt.Errorf("failed to get unindexed pages: %w", err)
		}

		if len(pages) == 0 {
			break // No more pages to index
		}

		// Index batch
		successCount, err := w.indexer.IndexDocuments(pages)
		if err != nil {
			log.Printf("Warning: batch indexing error: %v", err)
			// Continue with next batch even if this one failed
		}

		totalIndexed += successCount

		// Mark as indexed
		for _, page := range pages[:successCount] {
			if err := w.storage.MarkAsIndexed(page.ID); err != nil {
				log.Printf("Warning: failed to mark page %s as indexed: %v", page.ID, err)
			}
		}

		log.Printf("Indexed batch: %d pages (total: %d)", successCount, totalIndexed)

		// Small delay to avoid overwhelming the system
		time.Sleep(100 * time.Millisecond)
	}

	log.Printf("Triggered indexing complete: %d pages indexed", totalIndexed)
	return totalIndexed, nil
}
