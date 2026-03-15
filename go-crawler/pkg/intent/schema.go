package intent

import "time"

// IntentGoal represents the purpose of a query
type IntentGoal string

const (
	IntentGoalFindInformation IntentGoal = "find_information"
	IntentGoalComparison      IntentGoal = "comparison"
	IntentGoalTroubleshooting IntentGoal = "troubleshooting"
	IntentGoalPurchase        IntentGoal = "purchase"
	IntentGoalLocalService    IntentGoal = "local_service"
	IntentGoalNavigation      IntentGoal = "navigation"
	IntentGoalLearn           IntentGoal = "learn"
	IntentGoalCreate          IntentGoal = "create"
	IntentGoalReflect         IntentGoal = "reflect"
)

// UseCase represents specific use cases
type UseCase string

const (
	UseCaseComparison         UseCase = "comparison"
	UseCaseLearning           UseCase = "learning"
	UseCaseTroubleshooting    UseCase = "troubleshooting"
	UseCaseVerification       UseCase = "verification"
	UseCaseEntertainment      UseCase = "entertainment"
	UseCaseCommunityEngagement UseCase = "community_engagement"
	UseCaseProfessionalDevelopment UseCase = "professional_development"
	UseCaseMarketResearch     UseCase = "market_research"
)

// ConstraintType represents types of constraints
type ConstraintType string

const (
	ConstraintTypeInclusion ConstraintType = "inclusion"
	ConstraintTypeExclusion ConstraintType = "exclusion"
	ConstraintTypeRange     ConstraintType = "range"
	ConstraintTypeDatatype  ConstraintType = "datatype"
)

// Urgency represents time sensitivity
type Urgency string

const (
	UrgencyImmediate  Urgency = "immediate"
	UrgencySoon       Urgency = "soon"
	UrgencyFlexible   Urgency = "flexible"
	UrgencyExploratory Urgency = "exploratory"
)

// SkillLevel represents user expertise
type SkillLevel string

const (
	SkillLevelBeginner   SkillLevel = "beginner"
	SkillLevelIntermediate SkillLevel = "intermediate"
	SkillLevelAdvanced   SkillLevel = "advanced"
	SkillLevelExpert     SkillLevel = "expert"
)

// TemporalHorizon represents time horizon
type TemporalHorizon string

const (
	TemporalHorizonImmediate TemporalHorizon = "immediate"
	TemporalHorizonToday     TemporalHorizon = "today"
	TemporalHorizonWeek      TemporalHorizon = "week"
	TemporalHorizonMonth     TemporalHorizon = "month"
	TemporalHorizonLongterm  TemporalHorizon = "longterm"
	TemporalHorizonFlexible  TemporalHorizon = "flexible"
)

// Recency represents content recency preference
type Recency string

const (
	RecencyBreaking  Recency = "breaking"
	RecencyRecent    Recency = "recent"
	RecencyEvergreen Recency = "evergreen"
	RecencyHistorical Recency = "historical"
)

// Frequency represents how often something occurs
type Frequency string

const (
	FrequencyOneOff      Frequency = "oneoff"
	FrequencyRecurring   Frequency = "recurring"
	FrequencyExploratory Frequency = "exploratory"
	FrequencyFlexible    Frequency = "flexible"
)

// EthicalDimension represents ethical considerations
type EthicalDimension string

const (
	EthicalDimensionPrivacy       EthicalDimension = "privacy"
	EthicalDimensionSustainability EthicalDimension = "sustainability"
	EthicalDimensionEthics        EthicalDimension = "ethics"
	EthicalDimensionAccessibility EthicalDimension = "accessibility"
	EthicalDimensionOpenness      EthicalDimension = "openness"
)

// ResultType represents expected result type
type ResultType string

const (
	ResultTypeAnswer     ResultType = "answer"
	ResultTypeTutorial   ResultType = "tutorial"
	ResultTypeTool       ResultType = "tool"
	ResultTypeMarketplace ResultType = "marketplace"
	ResultTypeCommunity  ResultType = "community"
)

// Complexity represents query complexity
type Complexity string

const (
	ComplexitySimple   Complexity = "simple"
	ComplexityModerate Complexity = "moderate"
	ComplexityAdvanced Complexity = "advanced"
)

// Constraint represents a constraint extracted from user input
type Constraint struct {
	Type      ConstraintType `json:"type"`
	Dimension string         `json:"dimension"` // 'language', 'region', 'price', 'license', 'format', 'recency'
	Value     interface{}    `json:"value"`     // Single value, range, or list
	HardFilter bool          `json:"hardFilter"` // Must exclude results violating this
}

// TemporalIntent represents temporal aspects of user intent
type TemporalIntent struct {
	Horizon   TemporalHorizon `json:"horizon"`
	Recency   Recency         `json:"recency"`
	Frequency Frequency       `json:"frequency"`
}

// EthicalSignal represents ethical preferences
type EthicalSignal struct {
	Dimension  EthicalDimension `json:"dimension"`
	Preference string           `json:"preference"` // "privacy-first", "open-source", "carbon-neutral", etc.
}

// DeclaredIntent represents user-declared intent components
type DeclaredIntent struct {
	Query              string       `json:"query"`
	Goal               IntentGoal   `json:"goal,omitempty"`
	Constraints        []Constraint `json:"constraints"`
	NegativePreferences []string    `json:"negativePreferences"`
	Urgency            Urgency      `json:"urgency"`
	Budget             string       `json:"budget,omitempty"`
	SkillLevel         SkillLevel   `json:"skillLevel"`
	// Additional fields for intent-aware search (simplified combined model)
	UseCases          []UseCase       `json:"useCases,omitempty"`
	EthicalSignals    []EthicalSignal `json:"ethicalSignals,omitempty"`
	TemporalIntent    *TemporalIntent `json:"temporalIntent,omitempty"`
}

// InferredIntent represents inferred intent components
type InferredIntent struct {
	UseCases       []UseCase       `json:"useCases"`
	TemporalIntent *TemporalIntent `json:"temporalIntent,omitempty"`
	ResultType     ResultType      `json:"resultType,omitempty"`
	Complexity     Complexity      `json:"complexity"`
	EthicalSignals []EthicalSignal `json:"ethicalSignals"`
}

// IntentExtractionMetadata represents extracted intent from content
// This is stored with indexed documents for intent-aligned retrieval
type IntentExtractionMetadata struct {
	// Primary intent signals extracted from content
	PrimaryGoal    IntentGoal   `json:"primary_goal"`
	UseCases       []UseCase    `json:"use_cases"`
	Complexity     Complexity   `json:"complexity"`
	ResultType     ResultType   `json:"result_type"`
	EthicalSignals []EthicalSignal `json:"ethical_signals"`

	// Content characteristics for intent matching
	Topics           []string `json:"topics"`
	KeyPhrases       []string `json:"key_phrases"`
	TargetSkillLevel SkillLevel `json:"target_skill_level"`

	// Temporal relevance
	TemporalRelevance *TemporalIntent `json:"temporal_relevance,omitempty"`

	// Confidence scores
	ExtractionConfidence float64 `json:"extraction_confidence"`
}

// IntentIndexedDocument represents a document indexed with intent signals
// This is the core structure for intent-aware retrieval (NOT typical search)
type IntentIndexedDocument struct {
	ID              string                 `json:"id"`
	PageID          string                 `json:"page_id"`
	URL             string                 `json:"url"`
	Title           string                 `json:"title"`
	Content         string                 `json:"content"`
	MetaDescription string                 `json:"meta_description"`
	
	// Intent-specific fields (NOT in typical search engines)
	IntentMetadata  *IntentExtractionMetadata `json:"intent_metadata"`
	
	// Traditional search fields (secondary to intent)
	TermFrequencies map[string]int `json:"term_frequencies"`
	WordCount       int            `json:"word_count"`
	
	// Quality signals
	PageRank        float64 `json:"pagerank"`
	QualityScore    float64 `json:"quality_score"`
	
	// Timestamps
	IndexedAt       time.Time `json:"indexed_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

// IntentAlignmentScore represents how well a document matches an intent
type IntentAlignmentScore struct {
	TotalScore      float64  `json:"total_score"`
	GoalMatch       float64  `json:"goal_match"`
	UseCaseMatch    float64  `json:"use_case_match"`
	ComplexityMatch float64  `json:"complexity_match"`
	EthicalMatch    float64  `json:"ethical_match"`
	TemporalMatch   float64  `json:"temporal_match"`
	MatchReasons    []string `json:"match_reasons"`
}
