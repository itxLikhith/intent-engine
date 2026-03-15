package indexer

import (
	"os"
	"testing"
	"time"

	"github.com/itxLikhith/intent-engine/go-crawler/pkg/intent"
	"github.com/itxLikhith/intent-engine/go-crawler/pkg/models"
)

func TestIntentIndexer_IndexAndSearch(t *testing.T) {
	// Create temporary directory for test index
	tmpDir, err := os.MkdirTemp("", "intent-indexer-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Initialize indexer
	indexer, err := NewIntentIndexer(&IndexerConfig{
		IndexPath: tmpDir + "/bleve",
	})
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer indexer.Close()

	// Create test pages with different intent profiles
	pages := []*models.CrawledPage{
		{
			ID:              "page_1",
			URL:             "https://example.com/beginner-email-setup",
			Title:           "How to Set Up Encrypted Email for Beginners",
			Content:         "This beginner-friendly tutorial will teach you how to set up encrypted email step by step. Perfect for learning about secure communication.",
			MetaDescription: "Beginner's guide to encrypted email",
			PageRank:        0.8,
			CrawledAt:       time.Now(),
			UpdatedAt:       time.Now(),
			NextCrawlAt:     time.Now().Add(24 * time.Hour),
		},
		{
			ID:              "page_2",
			URL:             "https://example.com/advanced-email-config",
			Title:           "Advanced Email Server Configuration",
			Content:         "Expert-level guide for configuring custom email servers with advanced optimization and technical performance tuning.",
			MetaDescription: "Advanced email server configuration",
			PageRank:        0.7,
			CrawledAt:       time.Now(),
			UpdatedAt:       time.Now(),
			NextCrawlAt:     time.Now().Add(24 * time.Hour),
		},
		{
			ID:              "page_3",
			URL:             "https://example.com/email-providers-compare",
			Title:           "Best Email Providers Compared 2024",
			Content:         "Compare different email providers: ProtonMail vs Tutanota vs Gmail. See the differences and choose the best option.",
			MetaDescription: "Email provider comparison",
			PageRank:        0.9,
			CrawledAt:       time.Now(),
			UpdatedAt:       time.Now(),
			NextCrawlAt:     time.Now().Add(24 * time.Hour),
		},
		{
			ID:              "page_4",
			URL:             "https://example.com/privacy-email-review",
			Title:           "Privacy-First Email Services Review",
			Content:         "Review of privacy-focused, open-source email services with end-to-end encryption and no tracking.",
			MetaDescription: "Privacy-focused email review",
			PageRank:        0.85,
			CrawledAt:       time.Now(),
			UpdatedAt:       time.Now(),
			NextCrawlAt:     time.Now().Add(24 * time.Hour),
		},
	}

	// Index pages
	t.Log("Indexing test pages...")
	for _, page := range pages {
		if err := indexer.IndexDocument(page); err != nil {
			t.Errorf("Failed to index page %s: %v", page.ID, err)
		}
	}

	// Test 1: Traditional keyword search
	t.Run("KeywordSearch", func(t *testing.T) {
		response, err := indexer.Search("email setup", 10)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if response.TotalHits == 0 {
			t.Error("Expected results for keyword search")
		}

		t.Logf("Keyword search returned %d results", response.TotalHits)
		for i, result := range response.SearchResults {
			t.Logf("  %d. %s (score: %.2f)", i+1, result.Title, result.Score)
		}
	})

	// Test 2: Intent-aware search (beginner learning)
	t.Run("IntentSearch_Beginner_Learn", func(t *testing.T) {
		queryIntent := &intent.DeclaredIntent{
			Query:      "how to set up encrypted email",
			Goal:       intent.IntentGoalLearn,
			UseCases:   []intent.UseCase{intent.UseCaseLearning},
			SkillLevel: intent.SkillLevelBeginner,
			EthicalSignals: []intent.EthicalSignal{
				{Dimension: intent.EthicalDimensionPrivacy, Preference: "privacy-first"},
			},
		}

		response, err := indexer.SearchByIntent("email setup", queryIntent, 10)
		if err != nil {
			t.Fatalf("Intent search failed: %v", err)
		}

		if response.TotalHits == 0 {
			t.Error("Expected results for intent search")
		}

		// First result should be the beginner tutorial (page_1)
		if len(response.SearchResults) > 0 {
			topResult := response.SearchResults[0]
			t.Logf("Top result: %s", topResult.Title)
			t.Logf("  Intent alignment: %.2f", topResult.IntentAlignment.TotalScore)
			t.Logf("  Match reasons: %v", topResult.MatchReasons)

			// Verify intent alignment is computed
			if topResult.IntentAlignment == nil {
				t.Error("Expected intent alignment to be computed")
			} else if topResult.IntentAlignment.TotalScore < 0.5 {
				t.Errorf("Expected alignment score > 0.5, got %.2f", topResult.IntentAlignment.TotalScore)
			}
		}

		for i, result := range response.SearchResults {
			t.Logf("  %d. %s (final_score: %.2f, alignment: %.2f)",
				i+1, result.Title, result.FinalScore, result.IntentAlignment.TotalScore)
		}
	})

	// Test 3: Intent-aware search (advanced)
	t.Run("IntentSearch_Advanced", func(t *testing.T) {
		queryIntent := &intent.DeclaredIntent{
			Query:      "advanced email configuration",
			Goal:       intent.IntentGoalTroubleshooting,
			SkillLevel: intent.SkillLevelAdvanced,
		}

		response, err := indexer.SearchByIntent("email configuration", queryIntent, 10)
		if err != nil {
			t.Fatalf("Intent search failed: %v", err)
		}

		if len(response.SearchResults) > 0 {
			// Should rank advanced content higher
			topResult := response.SearchResults[0]
			t.Logf("Top result for advanced intent: %s", topResult.Title)
			t.Logf("  Complexity match: %.2f", topResult.IntentAlignment.ComplexityMatch)
		}
	})

	// Test 4: Intent-aware search (comparison)
	t.Run("IntentSearch_Comparison", func(t *testing.T) {
		queryIntent := &intent.DeclaredIntent{
			Query:    "compare email providers",
			Goal:     intent.IntentGoalComparison,
			UseCases: []intent.UseCase{intent.UseCaseComparison},
		}

		response, err := indexer.SearchByIntent("email providers", queryIntent, 10)
		if err != nil {
			t.Fatalf("Intent search failed: %v", err)
		}

		if len(response.SearchResults) > 0 {
			// Should rank comparison content higher
			topResult := response.SearchResults[0]
			t.Logf("Top result for comparison intent: %s", topResult.Title)
			t.Logf("  Goal match: %.2f", topResult.IntentAlignment.GoalMatch)
		}
	})

	// Test 5: Get stats
	t.Run("GetStats", func(t *testing.T) {
		stats, err := indexer.GetStats()
		if err != nil {
			t.Fatalf("Failed to get stats: %v", err)
		}

		if stats.TotalDocuments != 4 {
			t.Errorf("Expected 4 documents, got %d", stats.TotalDocuments)
		}

		t.Logf("Index stats: %+v", stats)
	})
}

func TestIntentIndexer_BatchIndexing(t *testing.T) {
	// Create temporary directory for test index
	tmpDir, err := os.MkdirTemp("", "intent-indexer-batch-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Initialize indexer
	indexer, err := NewIntentIndexer(&IndexerConfig{
		IndexPath: tmpDir + "/bleve",
	})
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer indexer.Close()

	// Create batch of test pages
	pages := make([]*models.CrawledPage, 20)
	for i := 0; i < 20; i++ {
		pages[i] = &models.CrawledPage{
			ID:              string(rune('a' + i)),
			URL:             "https://example.com/page-" + string(rune('a'+i)),
			Title:           "Test Page " + string(rune('A'+i)),
			Content:         "This is test content for page " + string(rune('a'+i)),
			MetaDescription: "Test description",
			PageRank:        0.5,
			CrawledAt:       time.Now(),
			UpdatedAt:       time.Now(),
			NextCrawlAt:     time.Now().Add(24 * time.Hour),
		}
	}

	// Index in batch
	successCount, err := indexer.IndexDocuments(pages)
	if err != nil {
		t.Fatalf("Batch indexing failed: %v", err)
	}

	if successCount != 20 {
		t.Errorf("Expected 20 pages indexed, got %d", successCount)
	}

	// Verify count
	stats, err := indexer.GetStats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.TotalDocuments != 20 {
		t.Errorf("Expected 20 documents in index, got %d", stats.TotalDocuments)
	}
}

func TestIntentAnalyzer_ContentAnalysis(t *testing.T) {
	analyzer := intent.NewIntentAnalyzer()

	tests := []struct {
		name          string
		title         string
		content       string
		metaDesc      string
		expectGoal    intent.IntentGoal
		expectPrivacy bool
	}{
		{
			name:          "Privacy tutorial",
			title:         "How to Set Up Privacy-First Email",
			content:       "Learn to configure encrypted email with this step-by-step tutorial",
			metaDesc:      "Privacy-focused email setup guide",
			expectGoal:    intent.IntentGoalLearn,
			expectPrivacy: true,
		},
		{
			name:          "Product comparison",
			title:         "Best Email Services Compared",
			content:       "Compare top email providers and see the differences",
			metaDesc:      "Email provider comparison",
			expectGoal:    intent.IntentGoalComparison,
			expectPrivacy: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metadata := analyzer.AnalyzeContent(tt.title, tt.content, tt.metaDesc)

			if metadata.PrimaryGoal != tt.expectGoal {
				t.Errorf("Expected goal %s, got %s", tt.expectGoal, metadata.PrimaryGoal)
			}

			hasPrivacy := false
			for _, signal := range metadata.EthicalSignals {
				if signal.Dimension == intent.EthicalDimensionPrivacy {
					hasPrivacy = true
					break
				}
			}

			if hasPrivacy != tt.expectPrivacy {
				t.Errorf("Expected privacy=%v, got privacy signals=%v", tt.expectPrivacy, metadata.EthicalSignals)
			}

			t.Logf("Extracted metadata: %+v", metadata)
		})
	}
}
