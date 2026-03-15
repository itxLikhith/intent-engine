package indexer

import (
	"fmt"
	"log"
	"time"

	"github.com/blevesearch/bleve/v2"
	"github.com/itxLikhith/intent-engine/go-crawler/internal/storage"
	"github.com/itxLikhith/intent-engine/go-crawler/pkg/intent"
)

// SearchIndexer wraps Bleve index
type SearchIndexer struct {
	index          bleve.Index
	store          *storage.Storage
	intentAnalyzer *intent.IntentAnalyzer
}

// SearchDocument represents a document in the search index
type SearchDocument struct {
	ID              string                      `json:"id"`
	URL             string                      `json:"url"`
	Title           string                      `json:"title"`
	Content         string                      `json:"content"`
	MetaDescription string                      `json:"meta_description"`
	WordCount       int                         `json:"word_count"`
	PageRank        float64                     `json:"pagerank"`
	IntentMetadata  *intent.IntentExtractionMetadata `json:"intent_metadata,omitempty"`
	IndexedAt       time.Time                   `json:"indexed_at"`
}

// SearchResult represents search results
type SearchResult struct {
	Total    int64              `json:"total"`
	MaxScore float64            `json:"max_score"`
	Hits     []SearchHit        `json:"hits"`
}

// SearchHit represents a single search hit
type SearchHit struct {
	ID       string                 `json:"id"`
	Score    float64                `json:"score"`
	Document *SearchDocument        `json:"document"`
}

// NewSearchIndexer creates a new search indexer
func NewSearchIndexer(blevePath string, store *storage.Storage) (*SearchIndexer, error) {
	// Create index mapping with intent fields
	mapping := bleve.NewIndexMapping()
	
	// Add fields for intent-aware search
	mapping.DefaultMapping.AddFieldMappingsAt("title", bleve.NewTextFieldMapping())
	mapping.DefaultMapping.AddFieldMappingsAt("content", bleve.NewTextFieldMapping())
	mapping.DefaultMapping.AddFieldMappingsAt("meta_description", bleve.NewTextFieldMapping())
	mapping.DefaultMapping.AddFieldMappingsAt("url", bleve.NewTextFieldMapping())
	
	// Add intent-specific fields
	mapping.DefaultMapping.AddFieldMappingsAt("intent_metadata.primary_goal", bleve.NewTextFieldMapping())
	mapping.DefaultMapping.AddFieldMappingsAt("intent_metadata.topics", bleve.NewTextFieldMapping())
	mapping.DefaultMapping.AddFieldMappingsAt("intent_metadata.use_cases", bleve.NewTextFieldMapping())
	mapping.DefaultMapping.AddFieldMappingsAt("intent_metadata.complexity", bleve.NewTextFieldMapping())
	mapping.DefaultMapping.AddFieldMappingsAt("intent_metadata.result_type", bleve.NewTextFieldMapping())

	// Open or create index
	index, err := bleve.Open(blevePath)
	if err == nil {
		log.Printf("Opened existing Bleve index at %s", blevePath)
	} else {
		// Index doesn't exist or is incompatible, create new one
		log.Printf("Creating new Bleve index at %s (error: %v)", blevePath, err)
		index, err = bleve.New(blevePath, mapping)
		if err != nil {
			return nil, fmt.Errorf("failed to create Bleve index: %w", err)
		}
		log.Printf("Created new Bleve index at %s", blevePath)
	}

	return &SearchIndexer{
		index:          index,
		store:          store,
		intentAnalyzer: intent.NewIntentAnalyzer(),
	}, nil
}

// IndexDocument indexes a single document with intent analysis
func (i *SearchIndexer) IndexDocument(doc *SearchDocument) error {
	// Analyze intent if not already present
	if doc.IntentMetadata == nil {
		doc.IntentMetadata = i.intentAnalyzer.AnalyzeContent(doc.Title, doc.Content, doc.MetaDescription)
		log.Printf("Intent extracted for %s: goal=%s, topics=%v", 
			doc.URL, doc.IntentMetadata.PrimaryGoal, doc.IntentMetadata.Topics)
	}
	
	return i.index.Index(doc.ID, doc)
}

// IndexPendingDocuments indexes pending crawled pages with intent analysis
func (i *SearchIndexer) IndexPendingDocuments(batchSize int) (int, error) {
	// Get uncrawled pages from storage
	pages, err := i.store.GetUnindexedPages(batchSize)
	if err != nil {
		return 0, fmt.Errorf("failed to get unindexed pages: %w", err)
	}

	indexed := 0
	for _, page := range pages {
		// Analyze intent from page content
		intentMetadata := i.intentAnalyzer.AnalyzeContent(page.Title, page.Content, page.MetaDescription)
		
		doc := &SearchDocument{
			ID:              page.ID,
			URL:             page.URL,
			Title:           page.Title,
			Content:         page.Content,
			MetaDescription: page.MetaDescription,
			WordCount:       len(page.Content) / 5, // Approximate word count
			PageRank:        page.PageRank,
			IntentMetadata:  intentMetadata,
			IndexedAt:       time.Now(),
		}

		if err := i.IndexDocument(doc); err != nil {
			log.Printf("Failed to index document %s: %v", page.ID, err)
			continue
		}

		// Mark as indexed
		if err := i.store.MarkAsIndexed(page.ID); err != nil {
			log.Printf("Failed to mark page as indexed: %v", err)
		}

		indexed++
		log.Printf("Indexed page with intent: %s (goal=%s, topics=%d)", 
			page.URL, intentMetadata.PrimaryGoal, len(intentMetadata.Topics))
	}

	return indexed, nil
}

// Search performs a search query
func (i *SearchIndexer) Search(query string, limit int) (*SearchResult, error) {
	// Create query
	q := bleve.NewQueryStringQuery(query)

	// Create search request
	req := bleve.NewSearchRequest(q)
	req.Size = limit
	req.Fields = []string{"title", "content", "meta_description", "url"}
	req.Highlight = bleve.NewHighlight()

	// Execute search
	result, err := i.index.Search(req)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Convert to our result format
	searchResult := &SearchResult{
		Total:    int64(result.Total),
		MaxScore: result.MaxScore,
		Hits:     make([]SearchHit, len(result.Hits)),
	}

	for i, hit := range result.Hits {
		doc := &SearchDocument{}
		for field, value := range hit.Fields {
			switch field {
			case "id":
				if s, ok := value.(string); ok {
					doc.ID = s
				}
			case "url":
				if s, ok := value.(string); ok {
					doc.URL = s
				}
			case "title":
				if s, ok := value.(string); ok {
					doc.Title = s
				}
			case "content":
				if s, ok := value.(string); ok {
					doc.Content = s
				}
			case "meta_description":
				if s, ok := value.(string); ok {
					doc.MetaDescription = s
				}
			}
		}

		searchResult.Hits[i] = SearchHit{
			ID:       hit.ID,
			Score:    hit.Score,
			Document: doc,
		}
	}

	return searchResult, nil
}

// Stats returns index statistics
func (i *SearchIndexer) Stats() (*IndexStats, error) {
	docCount, err := i.index.DocCount()
	if err != nil {
		return nil, err
	}

	return &IndexStats{
		DocumentCount: docCount,
	}, nil
}

// IndexStats holds index statistics
type IndexStats struct {
	DocumentCount uint64 `json:"document_count"`
}

// Close closes the index
func (i *SearchIndexer) Close() error {
	return i.index.Close()
}
