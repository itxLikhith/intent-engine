package indexer

import (
	"fmt"
	"log"
	"time"

	"github.com/blevesearch/bleve/v2"
	"github.com/blevesearch/bleve/v2/mapping"
	"github.com/itxLikhith/intent-engine/go-crawler/pkg/intent"
	"github.com/itxLikhith/intent-engine/go-crawler/pkg/models"
)

// IntentIndexer indexes documents with intent metadata for intent-aware retrieval
// This is NOT a typical search engine - it indexes intent signals for later alignment
type IntentIndexer struct {
	index        bleve.Index
	analyzer     *intent.IntentAnalyzer
	indexPath    string
}

// IndexerConfig holds indexer configuration
type IndexerConfig struct {
	IndexPath string `yaml:"index_path"`
}

// NewIntentIndexer creates a new intent-aware indexer
func NewIntentIndexer(config *IndexerConfig) (*IntentIndexer, error) {
	// Create intent analyzer
	analyzer := intent.NewIntentAnalyzer()

	// Open or create Bleve index
	index, err := openOrCreateIndex(config.IndexPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open index: %w", err)
	}

	log.Printf("Intent Indexer initialized at %s", config.IndexPath)

	return &IntentIndexer{
		index:     index,
		analyzer:  analyzer,
		indexPath: config.IndexPath,
	}, nil
}

// openOrCreateIndex opens existing index or creates new one with intent-aware mapping
func openOrCreateIndex(path string) (bleve.Index, error) {
	// Try to open existing index
	index, err := bleve.Open(path)
	if err == nil {
		log.Println("Opened existing Bleve index")
		return index, nil
	}

	// Create new index with custom mapping
	log.Println("Creating new Bleve index with intent-aware mapping")
	mapping := createIntentIndexMapping()

	index, err = bleve.New(path, mapping)
	if err != nil {
		return nil, fmt.Errorf("failed to create index: %w", err)
	}

	return index, nil
}

// createIntentIndexMapping creates Bleve mapping optimized for intent-aware retrieval
func createIntentIndexMapping() mapping.IndexMapping {
	indexMapping := bleve.NewIndexMapping()

	// Document mapping for intent-indexed documents
	docMapping := bleve.NewDocumentMapping()

	// Field: ID (exact match)
	idMapping := bleve.NewTextFieldMapping()
	idMapping.Index = true
	idMapping.DocValues = true
	idMapping.IncludeTermVectors = false
	docMapping.AddFieldMappingsAt("id", idMapping)

	// Field: PageID (exact match for joins)
	pageIDMapping := bleve.NewTextFieldMapping()
	pageIDMapping.Index = true
	pageIDMapping.DocValues = true
	pageIDMapping.IncludeTermVectors = false
	docMapping.AddFieldMappingsAt("page_id", pageIDMapping)

	// Field: URL (exact match)
	urlMapping := bleve.NewTextFieldMapping()
	urlMapping.Index = true
	urlMapping.DocValues = true
	urlMapping.IncludeTermVectors = false
	docMapping.AddFieldMappingsAt("url", urlMapping)

	// Field: Title (searchable with standard analyzer)
	titleMapping := bleve.NewTextFieldMapping()
	titleMapping.Index = true
	titleMapping.IncludeTermVectors = true
	titleMapping.Analyzer = "standard"
	titleMapping.Store = true
	docMapping.AddFieldMappingsAt("title", titleMapping)

	// Field: Content (searchable with standard analyzer)
	contentMapping := bleve.NewTextFieldMapping()
	contentMapping.Index = true
	contentMapping.IncludeTermVectors = true
	contentMapping.Analyzer = "standard"
	contentMapping.Store = true
	docMapping.AddFieldMappingsAt("content", contentMapping)

	// Field: MetaDescription (searchable)
	metaMapping := bleve.NewTextFieldMapping()
	metaMapping.Index = true
	metaMapping.IncludeTermVectors = true
	metaMapping.Analyzer = "standard"
	metaMapping.Store = true
	docMapping.AddFieldMappingsAt("meta_description", metaMapping)

	// === INTENT-SPECIFIC FIELDS (core differentiator from typical search) ===

	// Field: IntentMetadata.PrimaryGoal (keyword match for intent alignment)
	goalMapping := bleve.NewTextFieldMapping()
	goalMapping.Index = true
	goalMapping.DocValues = true
	goalMapping.IncludeTermVectors = false
	goalMapping.Analyzer = "keyword"
	goalMapping.Store = true
	docMapping.AddFieldMappingsAt("intent_metadata.primary_goal", goalMapping)

	// Field: IntentMetadata.UseCases (keyword array for use case matching)
	useCaseMapping := bleve.NewTextFieldMapping()
	useCaseMapping.Index = true
	useCaseMapping.DocValues = true
	useCaseMapping.IncludeTermVectors = false
	useCaseMapping.Analyzer = "keyword"
	useCaseMapping.Store = true
	docMapping.AddFieldMappingsAt("intent_metadata.use_cases", useCaseMapping)

	// Field: IntentMetadata.Complexity (keyword for skill-level matching)
	complexityMapping := bleve.NewTextFieldMapping()
	complexityMapping.Index = true
	complexityMapping.DocValues = true
	complexityMapping.IncludeTermVectors = false
	complexityMapping.Analyzer = "keyword"
	complexityMapping.Store = true
	docMapping.AddFieldMappingsAt("intent_metadata.complexity", complexityMapping)

	// Field: IntentMetadata.ResultType (keyword for result type matching)
	resultTypeMapping := bleve.NewTextFieldMapping()
	resultTypeMapping.Index = true
	resultTypeMapping.DocValues = true
	resultTypeMapping.IncludeTermVectors = false
	resultTypeMapping.Analyzer = "keyword"
	resultTypeMapping.Store = true
	docMapping.AddFieldMappingsAt("intent_metadata.result_type", resultTypeMapping)

	// Field: IntentMetadata.EthicalSignals (for ethical alignment)
	ethicalDimensionMapping := bleve.NewTextFieldMapping()
	ethicalDimensionMapping.Index = true
	ethicalDimensionMapping.DocValues = true
	ethicalDimensionMapping.IncludeTermVectors = false
	ethicalDimensionMapping.Analyzer = "keyword"
	ethicalDimensionMapping.Store = true
	docMapping.AddFieldMappingsAt("intent_metadata.ethical_signals.dimension", ethicalDimensionMapping)

	ethicalPreferenceMapping := bleve.NewTextFieldMapping()
	ethicalPreferenceMapping.Index = true
	ethicalPreferenceMapping.DocValues = true
	ethicalPreferenceMapping.IncludeTermVectors = false
	ethicalPreferenceMapping.Analyzer = "keyword"
	ethicalPreferenceMapping.Store = true
	docMapping.AddFieldMappingsAt("intent_metadata.ethical_signals.preference", ethicalPreferenceMapping)

	// Field: IntentMetadata.Topics (for topic-based filtering)
	topicsMapping := bleve.NewTextFieldMapping()
	topicsMapping.Index = true
	topicsMapping.DocValues = true
	topicsMapping.IncludeTermVectors = false
	topicsMapping.Analyzer = "keyword"
	topicsMapping.Store = true
	docMapping.AddFieldMappingsAt("intent_metadata.topics", topicsMapping)

	// Field: IntentMetadata.KeyPhrases (for phrase matching)
	keyPhrasesMapping := bleve.NewTextFieldMapping()
	keyPhrasesMapping.Index = true
	keyPhrasesMapping.DocValues = true
	keyPhrasesMapping.IncludeTermVectors = false
	keyPhrasesMapping.Analyzer = "keyword"
	keyPhrasesMapping.Store = true
	docMapping.AddFieldMappingsAt("intent_metadata.key_phrases", keyPhrasesMapping)

	// Field: IntentMetadata.TargetSkillLevel
	skillLevelMapping := bleve.NewTextFieldMapping()
	skillLevelMapping.Index = true
	skillLevelMapping.DocValues = true
	skillLevelMapping.IncludeTermVectors = false
	skillLevelMapping.Analyzer = "keyword"
	skillLevelMapping.Store = true
	docMapping.AddFieldMappingsAt("intent_metadata.target_skill_level", skillLevelMapping)

	// Field: IntentMetadata.ExtractionConfidence (numeric for filtering)
	confidenceMapping := bleve.NewNumericFieldMapping()
	confidenceMapping.Index = true
	confidenceMapping.DocValues = true
	confidenceMapping.Store = true
	docMapping.AddFieldMappingsAt("intent_metadata.extraction_confidence", confidenceMapping)

	// Field: PageRank (numeric for quality boosting)
	pagerankMapping := bleve.NewNumericFieldMapping()
	pagerankMapping.Index = true
	pagerankMapping.DocValues = true
	pagerankMapping.Store = true
	docMapping.AddFieldMappingsAt("pagerank", pagerankMapping)

	// Field: QualityScore (numeric for quality filtering)
	qualityMapping := bleve.NewNumericFieldMapping()
	qualityMapping.Index = true
	qualityMapping.DocValues = true
	qualityMapping.Store = true
	docMapping.AddFieldMappingsAt("quality_score", qualityMapping)

	// Field: WordCount (numeric for content depth)
	wordCountMapping := bleve.NewNumericFieldMapping()
	wordCountMapping.Index = true
	wordCountMapping.DocValues = true
	wordCountMapping.Store = true
	docMapping.AddFieldMappingsAt("word_count", wordCountMapping)

	// Field: IndexedAt (timestamp for temporal filtering)
	indexedAtMapping := bleve.NewDateTimeFieldMapping()
	indexedAtMapping.Index = true
	indexedAtMapping.DocValues = true
	indexedAtMapping.Store = true
	docMapping.AddFieldMappingsAt("indexed_at", indexedAtMapping)

	indexMapping.DefaultMapping = docMapping
	indexMapping.DefaultAnalyzer = "standard"

	return indexMapping
}

// IndexDocument indexes a single document with intent metadata
func (i *IntentIndexer) IndexDocument(page *models.CrawledPage) error {
	// Analyze content to extract intent signals
	intentMetadata := i.analyzer.AnalyzeContent(
		page.Title,
		page.Content,
		page.MetaDescription,
	)

	// Create intent-indexed document
	doc := &intent.IntentIndexedDocument{
		ID:              page.ID,
		PageID:          page.ID,
		URL:             page.URL,
		Title:           page.Title,
		Content:         page.Content,
		MetaDescription: page.MetaDescription,
		IntentMetadata:  intentMetadata,
		TermFrequencies: computeTermFrequencies(page.Content),
		WordCount:       countWords(page.Content),
		PageRank:        page.PageRank,
		QualityScore:    computeQualityScore(page, intentMetadata),
		IndexedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	// Index in Bleve
	if err := i.index.Index(page.ID, doc); err != nil {
		return fmt.Errorf("failed to index document: %w", err)
	}

	log.Printf("Indexed document: %s (goal=%s, use_cases=%d, confidence=%.2f)",
		page.ID, intentMetadata.PrimaryGoal, len(intentMetadata.UseCases),
		intentMetadata.ExtractionConfidence)

	return nil
}

// IndexDocuments indexes multiple documents in batch
func (i *IntentIndexer) IndexDocuments(pages []*models.CrawledPage) (int, error) {
	batch := i.index.NewBatch()
	successCount := 0

	for _, page := range pages {
		// Analyze content
		intentMetadata := i.analyzer.AnalyzeContent(
			page.Title,
			page.Content,
			page.MetaDescription,
		)

		// Create document
		doc := &intent.IntentIndexedDocument{
			ID:              page.ID,
			PageID:          page.ID,
			URL:             page.URL,
			Title:           page.Title,
			Content:         page.Content,
			MetaDescription: page.MetaDescription,
			IntentMetadata:  intentMetadata,
			TermFrequencies: computeTermFrequencies(page.Content),
			WordCount:       countWords(page.Content),
			PageRank:        page.PageRank,
			QualityScore:    computeQualityScore(page, intentMetadata),
			IndexedAt:       time.Now(),
			UpdatedAt:       time.Now(),
		}

		// Add to batch
		if err := batch.Index(page.ID, doc); err != nil {
			log.Printf("Warning: failed to add document %s to batch: %v", page.ID, err)
			continue
		}
		successCount++
	}

	// Execute batch
	if err := i.index.Batch(batch); err != nil {
		return successCount, fmt.Errorf("failed to execute batch: %w", err)
	}

	log.Printf("Batch indexed %d/%d documents", successCount, len(pages))
	return successCount, nil
}

// DeleteDocument removes a document from the index
func (i *IntentIndexer) DeleteDocument(pageID string) error {
	return i.index.Delete(pageID)
}

// SearchByIntent performs intent-aware search
// This is the key differentiator from typical search engines
func (i *IntentIndexer) SearchByIntent(query string, queryIntent *intent.DeclaredIntent, limit int) (*SearchResponse, error) {
	// Create a simple match query
	matchQuery := bleve.NewMatchQuery(query)
	matchQuery.SetField("title")
	matchQuery.SetBoost(2.0)
	
	// If intent is provided, add goal filtering
	if queryIntent != nil && queryIntent.Goal != "" {
		goalQuery := bleve.NewTermQuery(string(queryIntent.Goal))
		goalQuery.SetField("intent_metadata.primary_goal")
		conjunctionQuery := bleve.NewConjunctionQuery(matchQuery, goalQuery)
		searchRequest := bleve.NewSearchRequest(conjunctionQuery)
		return i.executeSearch(searchRequest, query, limit, queryIntent)
	}
	
	// Simple search without intent filtering
	searchRequest := bleve.NewSearchRequest(matchQuery)
	return i.executeSearch(searchRequest, query, limit, nil)
}

// executeSearch executes a search request and processes results
func (i *IntentIndexer) executeSearch(searchRequest *bleve.SearchRequest, query string, limit int, queryIntent *intent.DeclaredIntent) (*SearchResponse, error) {
	searchRequest.Size = limit
	searchRequest.Fields = []string{
		"id", "page_id", "url", "title", "content", "meta_description",
		"intent_metadata", "pagerank", "quality_score",
	}
	searchRequest.IncludeLocations = false
	searchRequest.Explain = true
	
	result, err := i.index.Search(searchRequest)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Convert results
	response := &SearchResponse{
		Query:        query,
		TotalHits:    int(result.Total),
		MaxScore:     result.MaxScore,
		SearchResults: make([]IntentSearchResult, 0, len(result.Hits)),
		ProcessingTimeMs: float64(result.Took) / float64(time.Millisecond),
	}

	for _, hit := range result.Hits {
		searchResult := IntentSearchResult{
			PageID:    hit.ID,
			Score:     hit.Score,
			MatchReasons: make([]string, 0),
		}

		// Extract fields
		if fields := hit.Fields; fields != nil {
			searchResult.URL = extractStringField(fields, "url")
			searchResult.Title = extractStringField(fields, "title")
			searchResult.Snippet = extractStringField(fields, "meta_description")

			// Extract intent metadata for alignment scoring
			if intentMeta, ok := fields["intent_metadata"].(map[string]interface{}); ok {
				searchResult.IntentMetadata = extractIntentMetadata(intentMeta)

				// Compute intent alignment if query intent provided
				if queryIntent != nil && searchResult.IntentMetadata != nil {
					doc := &intent.IntentIndexedDocument{
						IntentMetadata: searchResult.IntentMetadata,
					}
					alignment := intent.ComputeIntentAlignment(doc, queryIntent)
					searchResult.IntentAlignment = alignment
					searchResult.MatchReasons = alignment.MatchReasons
					
					// Boost score by intent alignment
					searchResult.FinalScore = hit.Score * (0.5 + 0.5*alignment.TotalScore)
				} else {
					searchResult.FinalScore = hit.Score
				}
			} else {
				searchResult.FinalScore = hit.Score
			}
		}

		response.SearchResults = append(response.SearchResults, searchResult)
	}

	return response, nil
}

// Search performs traditional keyword search (fallback)
func (i *IntentIndexer) Search(query string, limit int) (*SearchResponse, error) {
	return i.SearchByIntent(query, nil, limit)
}

// GetStats returns indexer statistics
func (i *IntentIndexer) GetStats() (*models.IndexStats, error) {
	docCount, err := i.index.DocCount()
	if err != nil {
		return nil, fmt.Errorf("failed to get doc count: %w", err)
	}

	// Get index size (approximate)
	stats := &models.IndexStats{
		TotalDocuments: int64(docCount),
		LastIndexedAt:  time.Now(),
	}

	// Note: Bleve doesn't expose total terms or index size directly
	// In production, would track these metrics separately

	return stats, nil
}

// Close closes the index
func (i *IntentIndexer) Close() error {
	return i.index.Close()
}

// === Helper Functions ===

// computeTermFrequencies computes term frequencies for a document
func computeTermFrequencies(content string) map[string]int {
	terms := make(map[string]int)
	words := splitWords(content)

	for _, word := range words {
		if len(word) > 2 { // Skip very short words
			terms[word]++
		}
	}

	return terms
}

// countWords counts words in content
func countWords(content string) int {
	return len(splitWords(content))
}

// splitWords splits content into words (simple tokenization)
func splitWords(content string) []string {
	// Simple word splitting - in production use proper tokenizer
	words := make([]string, 0)
	current := ""

	for _, r := range content {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			current += string(r)
		} else {
			if current != "" {
				words = append(words, lowercaseWord(current))
				current = ""
			}
		}
	}

	if current != "" {
		words = append(words, lowercaseWord(current))
	}

	return words
}

// lowercaseWord converts a word to lowercase
func lowercaseWord(word string) string {
	result := ""
	for _, r := range word {
		if r >= 'A' && r <= 'Z' {
			result += string(r + 32)
		} else {
			result += string(r)
		}
	}
	return result
}

// computeQualityScore computes overall quality score for a document
func computeQualityScore(page *models.CrawledPage, metadata *intent.IntentExtractionMetadata) float64 {
	score := 0.5 // Base score

	// Factor 1: Content length (longer = more comprehensive, up to a point)
	wordCount := countWords(page.Content)
	if wordCount > 1000 {
		score += 0.15
	} else if wordCount > 500 {
		score += 0.10
	} else if wordCount > 200 {
		score += 0.05
	}

	// Factor 2: Intent extraction confidence
	if metadata.ExtractionConfidence > 0.8 {
		score += 0.15
	} else if metadata.ExtractionConfidence > 0.6 {
		score += 0.10
	} else if metadata.ExtractionConfidence > 0.4 {
		score += 0.05
	}

	// Factor 3: PageRank (if available)
	if page.PageRank > 0.7 {
		score += 0.10
	} else if page.PageRank > 0.4 {
		score += 0.05
	}

	// Factor 4: Ethical signals (bonus for privacy-focused, open-source, etc.)
	if len(metadata.EthicalSignals) > 0 {
		score += 0.05
	}

	// Cap at 1.0
	if score > 1.0 {
		score = 1.0
	}

	return score
}

// buildIntentQuery builds a Bleve query that considers intent signals
// Simplified version using Bleve v2 API
func buildIntentQuery(query string, queryIntent *intent.DeclaredIntent) interface{} {
	// Simple match query across title and content
	matchQuery := bleve.NewMatchQuery(query)
	matchQuery.SetField("title")
	matchQuery.SetBoost(2.0)
	
	// If intent is provided, add additional filtering
	if queryIntent != nil && queryIntent.Goal != "" {
		// Create conjunction with goal filter
		goalQuery := bleve.NewTermQuery(string(queryIntent.Goal))
		goalQuery.SetField("intent_metadata.primary_goal")
		return bleve.NewConjunctionQuery(matchQuery, goalQuery)
	}
	
	return matchQuery
}

// extractStringField extracts a string field from search result fields
func extractStringField(fields map[string]interface{}, fieldName string) string {
	if val, ok := fields[fieldName]; ok {
		if str, ok := val.(string); ok {
			return str
		}
	}
	return ""
}

// extractIntentMetadata extracts intent metadata from search result fields
func extractIntentMetadata(meta map[string]interface{}) *intent.IntentExtractionMetadata {
	if meta == nil {
		return nil
	}

	result := &intent.IntentExtractionMetadata{}

	// Extract primary_goal
	if goal, ok := meta["primary_goal"].(string); ok {
		result.PrimaryGoal = intent.IntentGoal(goal)
	}

	// Extract use_cases
	if useCases, ok := meta["use_cases"].([]interface{}); ok {
		for _, uc := range useCases {
			if ucStr, ok := uc.(string); ok {
				result.UseCases = append(result.UseCases, intent.UseCase(ucStr))
			}
		}
	}

	// Extract complexity
	if complexity, ok := meta["complexity"].(string); ok {
		result.Complexity = intent.Complexity(complexity)
	}

	// Extract result_type
	if resultType, ok := meta["result_type"].(string); ok {
		result.ResultType = intent.ResultType(resultType)
	}

	// Extract target_skill_level
	if skillLevel, ok := meta["target_skill_level"].(string); ok {
		result.TargetSkillLevel = intent.SkillLevel(skillLevel)
	}

	// Extract extraction_confidence
	if confidence, ok := meta["extraction_confidence"].(float64); ok {
		result.ExtractionConfidence = confidence
	}

	// Extract topics
	if topics, ok := meta["topics"].([]interface{}); ok {
		for _, topic := range topics {
			if topicStr, ok := topic.(string); ok {
				result.Topics = append(result.Topics, topicStr)
			}
		}
	}

	// Extract key_phrases
	if keyPhrases, ok := meta["key_phrases"].([]interface{}); ok {
		for _, phrase := range keyPhrases {
			if phraseStr, ok := phrase.(string); ok {
				result.KeyPhrases = append(result.KeyPhrases, phraseStr)
			}
		}
	}

	// Extract ethical_signals
	if signals, ok := meta["ethical_signals"].([]interface{}); ok {
		for _, signal := range signals {
			if signalMap, ok := signal.(map[string]interface{}); ok {
				ethicalSignal := intent.EthicalSignal{}
				if dim, ok := signalMap["dimension"].(string); ok {
					ethicalSignal.Dimension = intent.EthicalDimension(dim)
				}
				if pref, ok := signalMap["preference"].(string); ok {
					ethicalSignal.Preference = pref
				}
				result.EthicalSignals = append(result.EthicalSignals, ethicalSignal)
			}
		}
	}

	return result
}

// IntentSearchResult represents a search result with intent alignment
type IntentSearchResult struct {
	PageID           string                        `json:"page_id"`
	URL              string                        `json:"url"`
	Title            string                        `json:"title"`
	Snippet          string                        `json:"snippet"`
	Score            float64                       `json:"score"`
	FinalScore       float64                       `json:"final_score"`
	IntentMetadata   *intent.IntentExtractionMetadata `json:"intent_metadata"`
	IntentAlignment  *intent.IntentAlignmentScore   `json:"intent_alignment"`
	MatchReasons     []string                      `json:"match_reasons"`
}

// SearchResponse represents a search response with intent-aware results
type SearchResponse struct {
	Query            string               `json:"query"`
	TotalHits        int                  `json:"total_hits"`
	MaxScore         float64              `json:"max_score"`
	SearchResults    []IntentSearchResult `json:"results"`
	ProcessingTimeMs float64              `json:"processing_time_ms"`
}
