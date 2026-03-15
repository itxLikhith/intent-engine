package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/itxLikhith/intent-engine/go-crawler/internal/indexer"
	"github.com/itxLikhith/intent-engine/go-crawler/internal/storage"
)

func main() {
	// Command line flags
	redisAddr := flag.String("redis", "localhost:6379", "Redis address")
	postgresDSN := flag.String("postgres", "postgresql://crawler:crawler@localhost:5432/intent_engine?sslmode=disable", "PostgreSQL DSN")
	blevePath := flag.String("bleve", "./data/bleve", "Bleve index path")
	badgerPath := flag.String("badger", "./data/badger", "BadgerDB path")
	batchSize := flag.Int("batch-size", 100, "Number of documents to index at once")
	interval := flag.Int("interval", 10, "Indexing interval in seconds")
	flag.Parse()

	log.Printf("Starting Intent Engine Indexer...")
	log.Printf("Redis: %s", *redisAddr)
	log.Printf("PostgreSQL: %s", *postgresDSN)
	log.Printf("Bleve Path: %s", *blevePath)
	log.Printf("Batch Size: %d, Interval: %ds", *batchSize, *interval)

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

	// Initialize indexer
	idx, err := indexer.NewSearchIndexer(*blevePath, store)
	if err != nil {
		log.Fatalf("Failed to initialize indexer: %v", err)
	}
	defer idx.Close()

	log.Println("Indexer initialized, starting indexing loop...")

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Shutting down indexer...")
		cancel()
	}()

	// Start indexing loop
	ticker := time.NewTicker(time.Duration(*interval) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("Indexer stopped")
			return
		case <-ticker.C:
			log.Println("Running indexing cycle...")
			
			count, err := idx.IndexPendingDocuments(*batchSize)
			if err != nil {
				log.Printf("Indexing error: %v", err)
				continue
			}

			if count > 0 {
				log.Printf("Indexed %d documents", count)
			} else {
				log.Println("No pending documents to index")
			}
		}
	}
}
