package intent

import (
	"fmt"
	"regexp"
	"strings"
)

// IntentAnalyzer extracts intent signals from content
// This is a rule-based analyzer (matching Python's approach)
// In production, this would use ML models via ONNX runtime
type IntentAnalyzer struct {
	goalPatterns       map[*regexp.Regexp]IntentGoal
	useCasePatterns    map[*regexp.Regexp]UseCase
	complexityPatterns map[*regexp.Regexp]Complexity
	resultTypePatterns map[*regexp.Regexp]ResultType
	ethicalPatterns    map[*regexp.Regexp]EthicalSignal
}

// NewIntentAnalyzer creates a new intent analyzer
func NewIntentAnalyzer() *IntentAnalyzer {
	analyzer := &IntentAnalyzer{
		goalPatterns:       make(map[*regexp.Regexp]IntentGoal),
		useCasePatterns:    make(map[*regexp.Regexp]UseCase),
		complexityPatterns: make(map[*regexp.Regexp]Complexity),
		resultTypePatterns: make(map[*regexp.Regexp]ResultType),
		ethicalPatterns:    make(map[*regexp.Regexp]EthicalSignal),
	}

	analyzer.compilePatterns()
	return analyzer
}

// compilePatterns compiles regex patterns for intent extraction
func (a *IntentAnalyzer) compilePatterns() {
	// Goal patterns (matching Python's GoalClassifier)
	goalPatterns := map[string]IntentGoal{
		`(?i)\b(how to|how do i|guide|tutorial|setup|configure|install|learn|teach me|explain|what is)\b`:        IntentGoalLearn,
		`(?i)\b(setup|configur|install|learn|tutorial|guide|manual|instructions?)\b`:                             IntentGoalLearn,
		`(?i)\b(compare|comparing|versus|vs\.?|difference|better|best|alternative|alternatives?)\b`:              IntentGoalComparison,
		`(?i)\b(which is|should i use|recommend|top|best|vs|versus)\b`:                                           IntentGoalComparison,
		`(?i)\b(fix|broken|not working|error|problem|issue|trouble|debug|can't|won't|help)\b`:                    IntentGoalTroubleshooting,
		`(?i)\b(why is|not working|fix|troubleshoot|solve|resolve)\b`:                                            IntentGoalTroubleshooting,
		`(?i)\b(buy|purchase|get|order|shop|price|cost|deal|discount|cheapest|where to buy)\b`:                   IntentGoalPurchase,
		`(?i)\b(price|cost|buy|purchase|order|where to get)\b`:                                                   IntentGoalPurchase,
		`(?i)\b(find|search|locate|where|when|who|what|information|details|about)\b`:                             IntentGoalFindInformation,
		`(?i)\b(near me|nearby|local|around here|closest|find near|find nearby)\b`:                               IntentGoalLocalService,
		`(?i)\b(go to|navigate to|directions|route|map|find location|address)\b`:                                 IntentGoalNavigation,
	}

	for pattern, goal := range goalPatterns {
		if re, err := regexp.Compile(pattern); err == nil {
			a.goalPatterns[re] = goal
		}
	}

	// Use case patterns
	useCasePatterns := map[string]UseCase{
		`(?i)\b(compare|difference|versus|vs|better|alternative)\b`:                                UseCaseComparison,
		`(?i)\b(how to|tutorial|guide|learn|setup|configure)\b`:                                    UseCaseLearning,
		`(?i)\b(fix|troubleshoot|error|problem|issue|debug|solve)\b`:                               UseCaseTroubleshooting,
		`(?i)\b(verify|confirm|validate|check|authentication|security)\b`:                          UseCaseVerification,
		`(?i)\b(learn|skills|career|professional|development|training)\b`:                          UseCaseProfessionalDevelopment,
		`(?i)\b(market|research|analysis|competitive|industry)\b`:                                  UseCaseMarketResearch,
	}

	for pattern, useCase := range useCasePatterns {
		if re, err := regexp.Compile(pattern); err == nil {
			a.useCasePatterns[re] = useCase
		}
	}

	// Complexity patterns
	complexityPatterns := map[string]Complexity{
		`(?i)\b(beginner|novice|newbie|basic|fundamental|simple|easy|introduction)\b`: ComplexitySimple,
		`(?i)\b(advanced|expert|mastery|technical|optimization|performance|custom)\b`: ComplexityAdvanced,
		`(?i)\b(intermediate|moderate|regular|standard|typical)\b`:                    ComplexityModerate,
	}

	for pattern, complexity := range complexityPatterns {
		if re, err := regexp.Compile(pattern); err == nil {
			a.complexityPatterns[re] = complexity
		}
	}

	// Result type patterns
	resultTypePatterns := map[string]ResultType{
		`(?i)\b(how to|tutorial|guide|step[- ]?by[- ]?step|walkthrough)\b`: ResultTypeTutorial,
		`(?i)\b(compare|versus|vs|difference|alternative|review)\b`:        ResultTypeCommunity,
		`(?i)\b(buy|purchase|price|deal|discount|shop)\b`:                  ResultTypeMarketplace,
		`(?i)\b(what is|explain|define|describe|meaning)\b`:                ResultTypeAnswer,
		`(?i)\b(tool|calculator|generator|converter|utility)\b`:            ResultTypeTool,
	}

	for pattern, resultType := range resultTypePatterns {
		if re, err := regexp.Compile(pattern); err == nil {
			a.resultTypePatterns[re] = resultType
		}
	}

	// Ethical signal patterns
	ethicalPatterns := map[string]EthicalSignal{
		`(?i)\b(privacy[- ]?first|privacy[- ]?focused|data protection|no tracking|secure)\b`: {
			Dimension:  EthicalDimensionPrivacy,
			Preference: "privacy-first",
		},
		`(?i)\b(open[- ]?source|foss|free software|libre|community driven)\b`: {
			Dimension:  EthicalDimensionOpenness,
			Preference: "open-source",
		},
		`(?i)\b(sustainable|eco[- ]?friendly|carbon[- ]?neutral|green|environmental)\b`: {
			Dimension:  EthicalDimensionSustainability,
			Preference: "sustainable",
		},
		`(?i)\b(ethical|fair trade|social responsibility|human rights)\b`: {
			Dimension:  EthicalDimensionEthics,
			Preference: "ethical",
		},
		`(?i)\b(accessible|accessibility|inclusive|wcag|disability)\b`: {
			Dimension:  EthicalDimensionAccessibility,
			Preference: "accessible",
		},
	}

	for pattern, signal := range ethicalPatterns {
		if re, err := regexp.Compile(pattern); err == nil {
			a.ethicalPatterns[re] = signal
		}
	}
}

// AnalyzeContent extracts intent signals from content
// This is the main entry point for intent analysis
func (a *IntentAnalyzer) AnalyzeContent(title, content, metaDescription string) *IntentExtractionMetadata {
	// Combine all text for analysis
	text := a.combineText(title, content, metaDescription)
	textLower := strings.ToLower(text)

	metadata := &IntentExtractionMetadata{
		UseCases:           make([]UseCase, 0),
		EthicalSignals:     make([]EthicalSignal, 0),
		Topics:             make([]string, 0),
		KeyPhrases:         make([]string, 0),
		ExtractionConfidence: 0.0,
	}

	// Extract primary goal
	metadata.PrimaryGoal = a.extractPrimaryGoal(textLower)

	// Extract use cases
	metadata.UseCases = a.extractUseCases(textLower)

	// Extract complexity
	metadata.Complexity = a.extractComplexity(textLower)

	// Extract result type
	metadata.ResultType = a.extractResultType(textLower)

	// Extract ethical signals
	metadata.EthicalSignals = a.extractEthicalSignals(textLower)

	// Extract topics and key phrases
	metadata.Topics = a.extractTopics(text)
	metadata.KeyPhrases = a.extractKeyPhrases(text)

	// Determine target skill level
	metadata.TargetSkillLevel = a.extractSkillLevel(textLower)

	// Calculate extraction confidence
	metadata.ExtractionConfidence = a.calculateConfidence(metadata)

	return metadata
}

// combineText combines title, content, and meta description
func (a *IntentAnalyzer) combineText(title, content, metaDescription string) string {
	parts := make([]string, 0, 3)
	
	if title != "" {
		parts = append(parts, title)
	}
	if content != "" {
		// Limit content to first 5000 chars for efficiency
		if len(content) > 5000 {
			content = content[:5000]
		}
		parts = append(parts, content)
	}
	if metaDescription != "" {
		parts = append(parts, metaDescription)
	}

	return strings.Join(parts, " ")
}

// extractPrimaryGoal extracts the primary intent goal
func (a *IntentAnalyzer) extractPrimaryGoal(textLower string) IntentGoal {
	goalScores := make(map[IntentGoal]int)

	for pattern, goal := range a.goalPatterns {
		if pattern.MatchString(textLower) {
			goalScores[goal]++
		}
	}

	if len(goalScores) == 0 {
		return IntentGoalLearn // Default goal
	}

	// Return goal with highest score
	var primaryGoal IntentGoal
	maxScore := 0
	for goal, score := range goalScores {
		if score > maxScore {
			maxScore = score
			primaryGoal = goal
		}
	}

	return primaryGoal
}

// extractUseCases extracts all matching use cases
func (a *IntentAnalyzer) extractUseCases(textLower string) []UseCase {
	useCaseSet := make(map[UseCase]bool)

	for pattern, useCase := range a.useCasePatterns {
		if pattern.MatchString(textLower) {
			useCaseSet[useCase] = true
		}
	}

	useCases := make([]UseCase, 0, len(useCaseSet))
	for useCase := range useCaseSet {
		useCases = append(useCases, useCase)
	}

	if len(useCases) == 0 {
		useCases = append(useCases, UseCaseLearning) // Default
	}

	return useCases
}

// extractComplexity extracts complexity level
func (a *IntentAnalyzer) extractComplexity(textLower string) Complexity {
	scores := make(map[Complexity]int)

	for pattern, complexity := range a.complexityPatterns {
		if pattern.MatchString(textLower) {
			scores[complexity]++
		}
	}

	if len(scores) == 0 {
		return ComplexityModerate // Default
	}

	var primaryComplexity Complexity
	maxScore := 0
	for complexity, score := range scores {
		if score > maxScore {
			maxScore = score
			primaryComplexity = complexity
		}
	}

	return primaryComplexity
}

// extractResultType extracts expected result type
func (a *IntentAnalyzer) extractResultType(textLower string) ResultType {
	typeScores := make(map[ResultType]int)

	for pattern, resultType := range a.resultTypePatterns {
		if pattern.MatchString(textLower) {
			typeScores[resultType]++
		}
	}

	if len(typeScores) == 0 {
		return ResultTypeTutorial // Default
	}

	var primaryType ResultType
	maxScore := 0
	for resultType, score := range typeScores {
		if score > maxScore {
			maxScore = score
			primaryType = resultType
		}
	}

	return primaryType
}

// extractEthicalSignals extracts ethical preferences
func (a *IntentAnalyzer) extractEthicalSignals(textLower string) []EthicalSignal {
	signalSet := make(map[string]EthicalSignal)

	for pattern, signal := range a.ethicalPatterns {
		if pattern.MatchString(textLower) {
			signalSet[string(signal.Dimension)] = signal
		}
	}

	signals := make([]EthicalSignal, 0, len(signalSet))
	for _, signal := range signalSet {
		signals = append(signals, signal)
	}

	return signals
}

// extractSkillLevel determines target skill level from content
func (a *IntentAnalyzer) extractSkillLevel(textLower string) SkillLevel {
	// Count indicators for each skill level
	beginnerIndicators := len(regexp.MustCompile(`(?i)\b(beginner|novice|newbie|basic|easy|simple|introduction|getting started)\b`).FindAllString(textLower, -1))
	advancedIndicators := len(regexp.MustCompile(`(?i)\b(advanced|expert|mastery|technical|complex|sophisticated)\b`).FindAllString(textLower, -1))

	if advancedIndicators > beginnerIndicators && advancedIndicators > 0 {
		return SkillLevelAdvanced
	} else if beginnerIndicators > 0 {
		return SkillLevelBeginner
	}

	return SkillLevelIntermediate // Default
}

// extractTopics extracts main topics from content
func (a *IntentAnalyzer) extractTopics(text string) []string {
	// Simple topic extraction based on noun phrases
	// In production, would use NLP library or ML model
	
	// Remove common stop words and extract significant words
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true,
		"but": true, "in": true, "on": true, "at": true, "to": true,
		"for": true, "of": true, "with": true, "by": true, "from": true,
	}

	words := strings.Fields(text)
	topicMap := make(map[string]int)

	for _, word := range words {
		word = strings.Trim(strings.ToLower(word), ".,!?;:\"'()[]{}")
		if len(word) > 4 && !stopWords[word] {
			topicMap[word]++
		}
	}

	// Return top 10 topics by frequency
	topics := make([]string, 0, 10)
	for topic := range topicMap {
		topics = append(topics, topic)
		if len(topics) >= 10 {
			break
		}
	}

	return topics
}

// extractKeyPhrases extracts key phrases from content
func (a *IntentAnalyzer) extractKeyPhrases(text string) []string {
	// Extract common technical phrases
	phrases := []string{
		"step-by-step", "how to", "best practices", "getting started",
		"quick start", "tutorial", "user guide", "api reference",
		"code example", "source code", "open source", "free software",
	}

	var keyPhrases []string
	textLower := strings.ToLower(text)

	for _, phrase := range phrases {
		if strings.Contains(textLower, phrase) {
			keyPhrases = append(keyPhrases, phrase)
		}
	}

	return keyPhrases
}

// calculateConfidence calculates confidence score for extraction
func (a *IntentAnalyzer) calculateConfidence(metadata *IntentExtractionMetadata) float64 {
	confidence := 0.5 // Base confidence

	// Boost for multiple signals
	if len(metadata.UseCases) > 0 {
		confidence += 0.1
	}
	if len(metadata.EthicalSignals) > 0 {
		confidence += 0.1
	}
	if metadata.PrimaryGoal != "" {
		confidence += 0.1
	}
	if len(metadata.Topics) > 5 {
		confidence += 0.1
	}

	// Cap at 0.95
	if confidence > 0.95 {
		confidence = 0.95
	}

	return confidence
}

// ComputeIntentAlignment computes how well a document matches a query intent
func ComputeIntentAlignment(doc *IntentIndexedDocument, queryIntent *DeclaredIntent) *IntentAlignmentScore {
	if doc.IntentMetadata == nil || queryIntent == nil {
		return &IntentAlignmentScore{
			TotalScore: 0.5,
			MatchReasons: []string{"insufficient-data"},
		}
	}

	score := &IntentAlignmentScore{
		GoalMatch:       0.0,
		UseCaseMatch:    0.0,
		ComplexityMatch: 0.0,
		EthicalMatch:    0.0,
		TemporalMatch:   0.5, // Neutral if not specified
		MatchReasons:    make([]string, 0),
	}

	// 1. Goal alignment (35% weight)
	if doc.IntentMetadata.PrimaryGoal == queryIntent.Goal {
		score.GoalMatch = 1.0
		score.MatchReasons = append(score.MatchReasons, fmt.Sprintf("matches-%s-intent", queryIntent.Goal))
	} else if doc.IntentMetadata.PrimaryGoal != "" && queryIntent.Goal != "" {
		// Partial match for related goals
		relatedGoals := getRelatedGoals(queryIntent.Goal)
		for _, related := range relatedGoals {
			if doc.IntentMetadata.PrimaryGoal == related {
				score.GoalMatch = 0.6
				score.MatchReasons = append(score.MatchReasons, fmt.Sprintf("related-to-%s-intent", queryIntent.Goal))
				break
			}
		}
	}

	// 2. Use case alignment (25% weight)
	queryUseCases := make(map[UseCase]bool)
	for _, uc := range queryIntent.UseCases {
		queryUseCases[uc] = true
	}

	if len(queryUseCases) > 0 {
		matches := 0
		for _, docUC := range doc.IntentMetadata.UseCases {
			if queryUseCases[docUC] {
				matches++
			}
		}
		score.UseCaseMatch = float64(matches) / float64(len(queryUseCases))
		if score.UseCaseMatch > 0 {
			score.MatchReasons = append(score.MatchReasons, "use-case-alignment")
		}
	}

	// 3. Complexity/Skill level alignment (15% weight)
	// Map complexity to skill level for comparison
	complexityToSkill := map[Complexity]SkillLevel{
		ComplexitySimple:   SkillLevelBeginner,
		ComplexityModerate: SkillLevelIntermediate,
		ComplexityAdvanced: SkillLevelAdvanced,
	}
	docSkillLevel := complexityToSkill[doc.IntentMetadata.Complexity]
	
	if docSkillLevel == queryIntent.SkillLevel {
		score.ComplexityMatch = 1.0
		score.MatchReasons = append(score.MatchReasons, fmt.Sprintf("skill-level-%s", queryIntent.SkillLevel))
	} else {
		// Adjacent skill levels get partial credit
		complexityDiff := getComplexityDifference(doc.IntentMetadata.Complexity, queryIntent.SkillLevel)
		if complexityDiff == 1 {
			score.ComplexityMatch = 0.7
		} else {
			score.ComplexityMatch = 0.3
		}
	}

	// 4. Ethical alignment (15% weight)
	queryEthics := make(map[EthicalDimension]bool)
	for _, signal := range queryIntent.EthicalSignals {
		queryEthics[signal.Dimension] = true
	}

	if len(queryEthics) > 0 {
		matches := 0
		for _, docSignal := range doc.IntentMetadata.EthicalSignals {
			if queryEthics[docSignal.Dimension] {
				matches++
			}
		}
		score.EthicalMatch = float64(matches) / float64(len(queryEthics))
		if score.EthicalMatch > 0 {
			score.MatchReasons = append(score.MatchReasons, "ethical-alignment")
		}
	}

	// 5. Temporal alignment (10% weight) - optional
	if queryIntent.TemporalIntent != nil && doc.IntentMetadata.TemporalRelevance != nil {
		// Compare temporal preferences
		if queryIntent.TemporalIntent.Recency == doc.IntentMetadata.TemporalRelevance.Recency {
			score.TemporalMatch = 1.0
		} else {
			score.TemporalMatch = 0.5
		}
	}

	// Calculate weighted total
	score.TotalScore = score.GoalMatch*0.35 +
		score.UseCaseMatch*0.25 +
		score.ComplexityMatch*0.15 +
		score.EthicalMatch*0.15 +
		score.TemporalMatch*0.10

	return score
}

// getRelatedGoals returns goals related to the given goal
func getRelatedGoals(goal IntentGoal) []IntentGoal {
	related := map[IntentGoal][]IntentGoal{
		IntentGoalLearn:           {IntentGoalFindInformation, IntentGoalTroubleshooting},
		IntentGoalComparison:      {IntentGoalPurchase, IntentGoalFindInformation},
		IntentGoalTroubleshooting: {IntentGoalLearn, IntentGoalFindInformation},
		IntentGoalPurchase:        {IntentGoalComparison, IntentGoalLocalService},
		IntentGoalLocalService:    {IntentGoalNavigation, IntentGoalPurchase},
		IntentGoalNavigation:      {IntentGoalLocalService, IntentGoalFindInformation},
	}

	if related, ok := related[goal]; ok {
		return related
	}
	return []IntentGoal{}
}

// getComplexityDifference returns the absolute difference between complexity levels
// Accepts Complexity for c1 and SkillLevel for c2 (maps SkillLevel to Complexity for comparison)
func getComplexityDifference(c1 Complexity, c2 SkillLevel) int {
	complexityLevelMap := map[Complexity]int{
		ComplexitySimple:   0,
		ComplexityModerate: 1,
		ComplexityAdvanced: 2,
	}
	
	skillToComplexity := map[SkillLevel]Complexity{
		SkillLevelBeginner:   ComplexitySimple,
		SkillLevelIntermediate: ComplexityModerate,
		SkillLevelAdvanced:   ComplexityAdvanced,
		SkillLevelExpert:     ComplexityAdvanced,
	}
	
	l1, ok1 := complexityLevelMap[c1]
	c2Mapped, ok2 := skillToComplexity[c2]
	l2 := 0
	if ok2 {
		l2, ok1 = complexityLevelMap[c2Mapped]
	}

	if !ok1 || !ok2 {
		return 2 // Max difference if unknown
	}

	diff := l1 - l2
	if diff < 0 {
		diff = -diff
	}
	return diff
}
