package frontier

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/itxLikhith/intent-engine/go-crawler/pkg/models"
)

// URLFrontier manages the URL queue for crawling
type URLFrontier struct {
	client    *redis.Client
	queueKey  string
	bloomKey  string
	ctx       context.Context
}

// NewURLFrontier creates a new URL frontier
func NewURLFrontier(addr, password string, db int) (*URLFrontier, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})

	ctx := context.Background()

	// Test connection
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &URLFrontier{
		client:   client,
		queueKey: "crawl_queue",
		bloomKey: "visited_urls",
		ctx:      ctx,
	}, nil
}

// AddURLs adds URLs to the crawl queue
func (f *URLFrontier) AddURLs(urls []string, priority int, depth int) (int, error) {
	added := 0

	for _, url := range urls {
		// Create queue item
		item := &models.CrawlQueueItem{
			ID:          generateID(),
			URL:         url,
			Priority:    priority,
			Depth:       depth,
			Status:      "pending",
			ScheduledAt: time.Now(),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		}

		// Serialize item
		data, err := json.Marshal(item)
		if err != nil {
			return added, fmt.Errorf("failed to serialize queue item: %w", err)
		}

		// Add to sorted set with priority as score
		pipe := f.client.Pipeline()
		pipe.ZAdd(f.ctx, f.queueKey, &redis.Z{
			Score:  float64(priority),
			Member: string(data),
		})

		_, err = pipe.Exec(f.ctx)
		if err != nil {
			return added, fmt.Errorf("failed to add URL to queue: %w", err)
		}

		added++
	}

	return added, nil
}

// GetNextURL gets the next URL to crawl based on priority
func (f *URLFrontier) GetNextURL() (*models.CrawlQueueItem, error) {
	// Get highest priority item from sorted set
	results, err := f.client.ZRangeWithScores(f.ctx, f.queueKey, 0, 0).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get next URL: %w", err)
	}

	if len(results) == 0 {
		return nil, nil // Queue empty
	}

	// Parse queue item
	var item models.CrawlQueueItem
	memberStr, ok := results[0].Member.(string)
	if !ok {
		return nil, fmt.Errorf("invalid queue item type")
	}

	if err := json.Unmarshal([]byte(memberStr), &item); err != nil {
		return nil, fmt.Errorf("failed to parse queue item: %w", err)
	}

	// Remove from queue
	f.client.ZRem(f.ctx, f.queueKey, memberStr)

	// Update status to crawling
	item.Status = "crawling"
	item.UpdatedAt = time.Now()

	return &item, nil
}

// MarkCompleted marks a URL as completed (visited)
func (f *URLFrontier) MarkCompleted(url string) error {
	// Mark as visited in Redis to prevent revisiting
	key := fmt.Sprintf("%s:%s", f.bloomKey, hashURL(url))
	return f.client.Set(f.ctx, key, "1", 0).Err()
}

// MarkFailed marks a URL as failed
func (f *URLFrontier) MarkFailed(url, errMsg string) error {
	// Also mark failed URLs as visited to avoid retry loops
	key := fmt.Sprintf("%s:%s", f.bloomKey, hashURL(url))
	return f.client.Set(f.ctx, key, "1", 0).Err()
}

// IsVisited checks if a URL has been visited using Bloom filter
func (f *URLFrontier) IsVisited(url string) (bool, error) {
	key := fmt.Sprintf("%s:%s", f.bloomKey, hashURL(url))
	val, err := f.client.Get(f.ctx, key).Result()
	if err == redis.Nil {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return val == "1", nil
}

// GetQueueSize returns the number of URLs in the queue
func (f *URLFrontier) GetQueueSize() (int64, error) {
	return f.client.ZCard(f.ctx, f.queueKey).Result()
}

// GetStats returns frontier statistics
func (f *URLFrontier) GetStats() (map[string]interface{}, error) {
	queueSize, err := f.GetQueueSize()
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"queue_size": queueSize,
	}, nil
}

// ClearQueue removes all items from the queue (for cleanup)
func (f *URLFrontier) ClearQueue() error {
	return f.client.Del(f.ctx, f.queueKey).Err()
}

// Close closes the Redis connection
func (f *URLFrontier) Close() error {
	return f.client.Close()
}

// generateID generates a unique ID
func generateID() string {
	return fmt.Sprintf("crawl_%d", time.Now().UnixNano())
}

// hashURL creates a hash of the URL for Bloom filter
func hashURL(url string) string {
	// Simple hash - in production use proper hashing
	h := 0
	for i := 0; i < len(url); i++ {
		h = 31*h + int(url[i])
	}
	return fmt.Sprintf("%d", h)
}
