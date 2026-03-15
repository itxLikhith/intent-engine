package intent

import (
	"testing"
)

func TestIntentAnalyzer_GoalExtraction(t *testing.T) {
	analyzer := NewIntentAnalyzer()

	tests := []struct {
		name     string
		title    string
		content  string
		expected IntentGoal
	}{
		{
			name:     "Learn goal",
			title:    "How to Set Up Encrypted Email",
			content:  "This tutorial will guide you through the setup process step by step",
			expected: IntentGoalLearn,
		},
		{
			name:     "Comparison goal",
			title:    "Best Email Providers Compared",
			content:  "We compare ProtonMail, Tutanota, and Gmail to help you choose",
			expected: IntentGoalComparison,
		},
		{
			name:     "Troubleshooting goal",
			title:    "Fix Email Not Working",
			content:  "Having problems with your email? Here's how to troubleshoot common issues",
			expected: IntentGoalTroubleshooting,
		},
		{
			name:     "Purchase goal",
			title:    "Buy ProtonMail Premium",
			content:  "Get the best deal on ProtonMail premium plans with our discounts",
			expected: IntentGoalPurchase,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metadata := analyzer.AnalyzeContent(tt.title, tt.content, "")
			if metadata.PrimaryGoal != tt.expected {
				t.Errorf("Expected goal %s, got %s", tt.expected, metadata.PrimaryGoal)
			}
		})
	}
}

func TestIntentAnalyzer_UseCaseExtraction(t *testing.T) {
	analyzer := NewIntentAnalyzer()

	tests := []struct {
		name    string
		content string
		expect  []UseCase
	}{
		{
			name:    "Learning use case",
			content: "Learn how to configure email with this comprehensive guide",
			expect:  []UseCase{UseCaseLearning},
		},
		{
			name:    "Troubleshooting use case",
			content: "Fix and troubleshoot common email problems",
			expect:  []UseCase{UseCaseTroubleshooting},
		},
		{
			name:    "Comparison use case",
			content: "Compare different providers and see the differences",
			expect:  []UseCase{UseCaseComparison},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metadata := analyzer.AnalyzeContent("", tt.content, "")
			
			found := false
			for _, uc := range metadata.UseCases {
				for _, expected := range tt.expect {
					if uc == expected {
						found = true
						break
					}
				}
			}
			
			if !found && len(tt.expect) > 0 {
				t.Errorf("Expected use cases %v, got %v", tt.expect, metadata.UseCases)
			}
		})
	}
}

func TestIntentAnalyzer_ComplexityExtraction(t *testing.T) {
	analyzer := NewIntentAnalyzer()

	tests := []struct {
		name     string
		content  string
		expected Complexity
	}{
		{
			name:     "Beginner complexity",
			content:  "Beginner's guide: easy setup for newcomers",
			expected: ComplexitySimple,
		},
		{
			name:     "Advanced complexity",
			content:  "Advanced optimization techniques for experts",
			expected: ComplexityAdvanced,
		},
		{
			name:     "Moderate complexity (default)",
			content:  "Standard configuration guide",
			expected: ComplexityModerate,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metadata := analyzer.AnalyzeContent("", tt.content, "")
			if metadata.Complexity != tt.expected {
				t.Errorf("Expected complexity %s, got %s", tt.expected, metadata.Complexity)
			}
		})
	}
}

func TestIntentAnalyzer_EthicalSignals(t *testing.T) {
	analyzer := NewIntentAnalyzer()

	tests := []struct {
		name    string
		content string
		expect  int // minimum number of ethical signals
	}{
		{
			name:    "Privacy-focused",
			content: "Privacy-first email with end-to-end encryption and no tracking",
			expect:  1,
		},
		{
			name:    "Open-source",
			content: "Open-source software with community-driven development",
			expect:  1,
		},
		{
			name:    "No ethical signals",
			content: "Regular email service",
			expect:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metadata := analyzer.AnalyzeContent("", tt.content, "")
			
			if len(metadata.EthicalSignals) < tt.expect {
				t.Errorf("Expected at least %d ethical signals, got %d", tt.expect, len(metadata.EthicalSignals))
			}
		})
	}
}

func TestIntentAnalyzer_FullAnalysis(t *testing.T) {
	analyzer := NewIntentAnalyzer()

	title := "How to Set Up Encrypted Email for Beginners"
	content := `
		This comprehensive tutorial will teach you how to set up 
		privacy-first encrypted email. Perfect for beginners who want 
		to learn about secure communication. Step-by-step guide with 
		easy instructions.
	`
	metaDescription := "Beginner's guide to encrypted email setup"

	metadata := analyzer.AnalyzeContent(title, content, metaDescription)

	// Verify primary goal
	if metadata.PrimaryGoal != IntentGoalLearn {
		t.Errorf("Expected goal %s, got %s", IntentGoalLearn, metadata.PrimaryGoal)
	}

	// Verify use cases
	if len(metadata.UseCases) == 0 {
		t.Error("Expected use cases to be extracted")
	}

	// Verify complexity
	if metadata.Complexity != ComplexitySimple {
		t.Errorf("Expected beginner complexity, got %s", metadata.Complexity)
	}

	// Verify ethical signals (privacy)
	hasPrivacy := false
	for _, signal := range metadata.EthicalSignals {
		if signal.Dimension == EthicalDimensionPrivacy {
			hasPrivacy = true
			break
		}
	}
	if !hasPrivacy {
		t.Error("Expected privacy ethical signal")
	}

	// Verify confidence is reasonable
	if metadata.ExtractionConfidence < 0.5 {
		t.Errorf("Expected confidence > 0.5, got %f", metadata.ExtractionConfidence)
	}

	t.Logf("Full analysis result: %+v", metadata)
}

func TestComputeIntentAlignment(t *testing.T) {
	tests := []struct {
		name          string
		docMetadata   *IntentExtractionMetadata
		queryIntent   *DeclaredIntent
		expectedScore float64 // minimum expected score
	}{
		{
			name: "Perfect goal match",
			docMetadata: &IntentExtractionMetadata{
				PrimaryGoal: IntentGoalLearn,
				UseCases:    []UseCase{UseCaseLearning},
				Complexity:  ComplexitySimple,
			},
			queryIntent: &DeclaredIntent{
				Goal:       IntentGoalLearn,
				UseCases:   []UseCase{UseCaseLearning},
				SkillLevel: SkillLevelBeginner,
			},
			expectedScore: 0.7, // High score expected
		},
		{
			name: "Goal mismatch",
			docMetadata: &IntentExtractionMetadata{
				PrimaryGoal: IntentGoalPurchase,
				UseCases:    []UseCase{UseCaseMarketResearch},
				Complexity:  ComplexityAdvanced,
			},
			queryIntent: &DeclaredIntent{
				Goal:       IntentGoalLearn,
				UseCases:   []UseCase{UseCaseLearning},
				SkillLevel: SkillLevelBeginner,
			},
			expectedScore: 0.3, // Low score expected
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc := &IntentIndexedDocument{
				IntentMetadata: tt.docMetadata,
			}
			
			alignment := ComputeIntentAlignment(doc, tt.queryIntent)
			
			if alignment.TotalScore < tt.expectedScore {
				t.Errorf("Expected score >= %.2f, got %.2f", tt.expectedScore, alignment.TotalScore)
			}
			
			t.Logf("Alignment score: %.2f (goal=%.2f, use_case=%.2f, complexity=%.2f, ethical=%.2f)",
				alignment.TotalScore, alignment.GoalMatch, alignment.UseCaseMatch,
				alignment.ComplexityMatch, alignment.EthicalMatch)
		})
	}
}
