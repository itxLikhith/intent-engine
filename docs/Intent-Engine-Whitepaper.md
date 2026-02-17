# INTENT ENGINE: Foundational Architecture for Privacy-First Workspace Ecosystems

**Version:** 2.0
**Date:** February 17, 2026
**Target Audience:** Founders, Principal Engineers, System Architects
**Classification:** Technical Design Document

---

## EXECUTIVE SUMMARY

This whitepaper defines the **Intent Engine**—a foundational, privacy-preserving system that extracts, normalizes, ranks, and matches user intent across a distributed workspace ecosystem. Unlike surveillance-based systems (Google, Meta), the Intent Engine operates on **ephemeral, declared, and locally-inferred signals** without persistent behavioral profiling.

### Core Principles

1. **Intent-First**: All product decisions derive from structured intent, not user identity or history
2. **Privacy Native**: No persistent tracking; intent signals decay on session boundary
3. **Open Architecture**: Intent schema is language-agnostic, composable, and extensible
4. **Non-Discriminatory**: Matching algorithms are constraint-based, never using sensitive attributes or proxy discrimination
5. **Transparent**: Intent extraction rules are inspectable; users control intent signals

### Key Deliverables

- **Universal Intent Schema**: Extensible data model capturing constraints, goals, temporal context, skill level, ethical signals
- **Intent Extraction Algorithm**: Parses text, metadata, and UI events without behavioral profiling
- **Cross-Product Intent Normalization**: Maps Search, Docs, Mail, Calendar, Meet, Forms, Sites, Diary intents to common schema
- **Ranking & Matching Logic**: Service selection, result ranking, ad matching based purely on declared intent
- **Privacy & Threat Model**: Formal analysis of tracking risks and mitigations

---

## SYSTEM OVERVIEW

### Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION LAYER                    │
│         (Search | Docs | Mail | Calendar | Meet | Sites)     │
└────────────┬──────────────────────────────────┬──────────────┘
             │                                  │
             ▼                                  ▼
    ┌──────────────────┐          ┌──────────────────┐
    │ Event Capture    │          │  Form/Fields     │
    │ (Local)          │          │  Parser          │
    └──────────┬───────┘          └────────┬─────────┘
               │                           │
               └───────────┬───────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   INTENT EXTRACTION ENGINE       │
        │  (Parse & Normalize Intent)      │
        │  - Constraint extraction         │
        │  - Goal/use case inference       │
        │  - Temporal intent detection     │
        │  - Skill level assessment        │
        │  - Ethical signal recognition    │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │   INTENT SCHEMA (Normalized)     │
        │  - Structured constraints        │
        │  - Semantic goals                │
        │  - Temporal context              │
        │  - Skill signals                 │
        │  - Privacy preferences           │
        └──────────────┬───────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌────────┐  ┌──────────┐  ┌──────────┐
    │ SEARCH │  │   DOCS   │  │   ADS    │
    │RANKING │  │ RANKING  │  │MATCHING  │
    └────────┘  └──────────┘  └──────────┘
         │             │             │
         └─────────────┼─────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │   RESULT/SERVICE RENDERING       │
        │  (Intent-matched outputs)        │
        └──────────────────────────────────┘
```

### Why Intent-First Instead of Identity-Based?

| Signal Type | Google/Meta | Intent Engine |
|---|---|---|
| **Data Collection** | Behavioral profiling, cross-site tracking | Declared intent, local context |
| **Signal Lifetime** | Persistent (years) | Ephemeral (session-scoped) |
| **Tracking** | Cookies, fingerprinting, login linking | None—opt-in on-device inference |
| **Bias Risk** | High (proxy discrimination via history) | Lower (constraints are explicit) |
| **User Control** | Limited (settings buried) | Native (intent signals are user-visible) |
| **Inference** | Behavioral: "users like you bought X" | Contextual: "this query seeks comparisons" |
| **GDPR/Privacy Law** | Requires consent, tracking infrastructure | Non-tracking by design |

---

## UNIVERSAL INTENT SCHEMA

### Design Principles

The schema must be:
1. **Language-agnostic**: Work across natural language, structured fields, UI events
2. **Composable**: Combine intent signals from multiple sources (query, document state, calendar)
3. **Extensible**: New intent categories added without breaking existing systems
4. **Actionable**: Enable ranking, matching, and service selection without side channels
5. **Inspectable**: Users can view which intent signals were extracted

### Core Data Model

> **Note:** The following TypeScript representation is for illustrative purposes. The canonical data models are defined in Python in the `core/schema.py` file.

```typescript
interface UniversalIntent {
  // Unique session-scoped ID (not persistent)
  intentId: string;

  // Product context (which service generated this)
  context: {
    product: 'search' | 'docs' | 'mail' | 'calendar' | 'meet' | 'forms' | 'sites' | 'diary';
    timestamp: ISO8601;
    sessionId: string; // Session-scoped only
    userLocale: string; // e.g., 'en-IN'
  };

  // Declared intent (user-supplied constraints and goals)
  declared: {
    query?: string; // Free-form text
    goal: IntentGoal; // Structured goal
    constraints: Constraint[]; // Hard filters
    negativePreferences: string[]; // "not X", "no Y"
    urgency: 'immediate' | 'soon' | 'flexible' | 'exploratory';
    budget?: string; // "under 1000", "premium", null
    skillLevel: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  };

  // Inferred intent (derived from context without tracking)
  inferred: {
    useCases: UseCase[]; // [comparison, learning, troubleshooting, ...]
    temporalIntent: TemporalIntent;
    documentContext?: DocumentContext; // From open docs/emails
    meetingContext?: MeetingContext; // From calendar/Meet
    resultType: 'answer' | 'tutorial' | 'tool' | 'marketplace' | 'community';
    complexity: 'simple' | 'moderate' | 'advanced';
    ethicalSignals: EthicalSignal[]; // Privacy, sustainability, etc.
  };

  // Feedback (captured in current session only)
  sessionFeedback: {
    clicked?: string[]; // URLs clicked
    dwell?: number; // Seconds on result
    reformulated?: boolean; // User refined query
    bounced?: boolean; // Left immediately
  };

  // TTL: Auto-delete after session ends
  expiresAt: ISO8601;
}

enum IntentGoal {
  // Search-specific
  FIND_INFORMATION = 'find_information',
  COMPARISON = 'comparison',
  TROUBLESHOOTING = 'troubleshooting',
  PURCHASE = 'purchase',
  LOCAL_SERVICE = 'local_service',
  NAVIGATION = 'navigation',

  // Docs/Mail-specific
  DRAFT_DOCUMENT = 'draft_document',
  COLLABORATE = 'collaborate',
  ORGANIZE = 'organize',
  ANALYZE = 'analyze',
  SCHEDULE = 'schedule',

  // Cross-product
  LEARN = 'learn',
  CREATE = 'create',
  REFLECT = 'reflect', // Diary
}

enum UseCase {
  COMPARISON = 'comparison',
  LEARNING = 'learning',
  TROUBLESHOOTING = 'troubleshooting',
  VERIFICATION = 'verification',
  ENTERTAINMENT = 'entertainment',
  COMMUNITY_ENGAGEMENT = 'community_engagement',
  PROFESSIONAL_DEVELOPMENT = 'professional_development',
  MARKET_RESEARCH = 'market_research',
}

interface Constraint {
  type: 'inclusion' | 'exclusion' | 'range' | 'datatype';
  dimension: string; // 'language', 'region', 'price', 'license', 'format', 'recency'
  value: string | number | [number, number]; // Single value, range, or list
  hardFilter: boolean; // Must exclude results violating this
}

interface TemporalIntent {
  horizon: 'immediate' | 'today' | 'week' | 'month' | 'longterm';
  recency: 'breaking' | 'recent' | 'evergreen' | 'historical';
  frequency: 'oneoff' | 'recurring' | 'exploratory';
}

interface DocumentContext {
  docId?: string;
  content?: string; // First 1000 chars only, not persisted
  lastEditTime?: ISO8601;
  collaborators?: number; // Count only, not names
  contentType: 'text' | 'spreadsheet' | 'presentation' | 'form';
}

interface MeetingContext {
  meetingId?: string;
  subject?: string;
  participantCount?: number;
  isRecurring?: boolean;
  timeZone?: string;
}

interface EthicalSignal {
  dimension: 'privacy' | 'sustainability' | 'ethics' | 'accessibility' | 'openness';
  preference: string; // "privacy-first", "open-source", "carbon-neutral", etc.
}
```

### Schema Instantiation Examples

#### Example 1: Search Query

**User Query:** *"How to set up end-to-end encrypted email on Android, no big tech solutions"*

```json
{
  "intentId": "sess_abc123_1",
  "context": {
    "product": "search",
    "timestamp": "2026-01-19T12:34:56Z",
    "sessionId": "sess_abc123",
    "userLocale": "en-IN"
  },
  "declared": {
    "query": "How to set up end-to-end encrypted email on Android, no big tech solutions",
    "goal": "LEARN",
    "constraints": [
      {
        "type": "inclusion",
        "dimension": "platform",
        "value": "Android",
        "hardFilter": true
      },
      {
        "type": "inclusion",
        "dimension": "feature",
        "value": "end-to-end_encryption",
        "hardFilter": true
      },
      {
        "type": "exclusion",
        "dimension": "provider",
        "value": ["Google", "Microsoft", "Apple"],
        "hardFilter": true
      }
    ],
    "negativePreferences": ["no big tech", "no proprietary"],
    "urgency": "soon",
    "skillLevel": "intermediate"
  },
  "inferred": {
    "useCases": ["learning", "troubleshooting"],
    "temporalIntent": {
      "horizon": "today",
      "recency": "recent",
      "frequency": "oneoff"
    },
    "resultType": "tutorial",
    "complexity": "moderate",
    "ethicalSignals": [
      {
        "dimension": "privacy",
        "preference": "privacy-first"
      },
      {
        "dimension": "openness",
        "preference": "open-source_preferred"
      }
    ]
  },
  "sessionFeedback": {},
  "expiresAt": "2026-01-19T20:34:56Z"
}
```

#### Example 2: Docs Collaboration Intent

**User Action:** Opens a shared document and begins editing a section on "Q1 Budget Forecast"

```json
{
  "intentId": "sess_def456_1",
  "context": {
    "product": "docs",
    "timestamp": "2026-01-19T09:15:30Z",
    "sessionId": "sess_def456",
    "userLocale": "en-IN"
  },
  "declared": {
    "goal": "COLLABORATE",
    "constraints": [
      {
        "type": "inclusion",
        "dimension": "document_type",
        "value": "spreadsheet",
        "hardFilter": false
      }
    ],
    "urgency": "soon",
    "skillLevel": "advanced"
  },
  "inferred": {
    "useCases": ["collaboration", "analysis"],
    "documentContext": {
      "docId": "doc_q1_budget",
      "content": "Q1 Budget Forecast for APAC region...",
      "lastEditTime": "2026-01-19T08:00:00Z",
      "collaborators": 5,
      "contentType": "spreadsheet"
    },
    "temporalIntent": {
      "horizon": "week",
      "recency": "recent",
      "frequency": "recurring"
    },
    "resultType": "tool",
    "complexity": "advanced",
    "ethicalSignals": []
  },
  "sessionFeedback": {},
  "expiresAt": "2026-01-19T17:15:30Z"
}
```

#### Example 3: Calendar/Diary Intent

**User Action:** Creates a calendar event for "Weekly 1:1 with mentor" and adds a private diary entry: *"Feeling uncertain about career direction, need guidance"*

```json
{
  "intentId": "sess_ghi789_2",
  "context": {
    "product": "diary",
    "timestamp": "2026-01-19T18:45:00Z",
    "sessionId": "sess_ghi789",
    "userLocale": "en-IN"
  },
  "declared": {
    "goal": "REFLECT",
    "constraints": [],
    "urgency": "soon",
    "skillLevel": "intermediate"
  },
  "inferred": {
    "useCases": ["learning", "professional_development"],
    "meetingContext": {
      "subject": "Weekly 1:1 with mentor",
      "participantCount": 2,
      "isRecurring": true,
      "timeZone": "Asia/Kolkata"
    },
    "temporalIntent": {
      "horizon": "week",
      "recency": "evergreen",
      "frequency": "recurring"
    },
    "resultType": "community", // Suggests mentor resources/articles
    "complexity": "moderate",
    "ethicalSignals": [
      {
        "dimension": "privacy",
        "preference": "private_entry_only"
      }
    ]
  },
  "sessionFeedback": {},
  "expiresAt": "2026-01-19T23:45:00Z"
}
```

---

## INTENT EXTRACTION ALGORITHM

### Overview

The extraction algorithm operates in **three parallel phases**:
1. **Parsing**: Decompose input (query, form fields, document state) into atomic signals
2. **Inference**: Derive unstated intent from context and linguistic patterns
3. **Normalization**: Map signals to universal schema without tracking side effects

### Phase 1: Parsing

#### 1A. Query/Text Parsing

```python
def parse_text_intent(text: str, context: Context) -> ParsedIntent:
    """
    Extract explicit constraints, goals, and signals from free-form text.
    Uses regex, NLP, and rule-based patterns—no ML models that require user data.
    """

    parsed = ParsedIntent()

    # 1. Constraint extraction (hard filters)
    constraints = extract_constraints(text)
    # Examples:
    #   "Android" → constraint(platform = Android)
    #   "under 500" → constraint(price <= 500)
    #   "not Microsoft" → constraint(provider != Microsoft)
    #   "2024" → constraint(year = 2024)
    #   "open source" → constraint(license = open_source)
    parsed.constraints = constraints

    # 2. Goal inference from question/imperative patterns
    goal = classify_goal(text)
    # Patterns:
    #   "How to..." → LEARN
    #   "Compare X vs Y" → COMPARISON
    #   "What's wrong with..." → TROUBLESHOOTING
    #   "Where can I..." → LOCAL_SERVICE or NAVIGATION
    #   "I need to draft..." → DRAFT_DOCUMENT
    parsed.goal = goal

    # 3. Use case detection
    use_cases = detect_use_cases(text)
    # Keywords map:
    #   "vs", "compare", "better", "differ" → COMPARISON
    #   "tutorial", "learn", "explain", "basics" → LEARNING
    #   "broken", "error", "not working" → TROUBLESHOOTING
    parsed.use_cases = use_cases

    # 4. Temporal intent extraction
    temporal = extract_temporal_intent(text)
    # Patterns:
    #   "today", "now", "urgent" → horizon = immediate
    #   "this week" → horizon = week
    #   "evergreen tutorial" → recency = evergreen
    parsed.temporal_intent = temporal

    # 5. Skill level inference
    skill_level = infer_skill_level(text)
    # Heuristics:
    #   "For beginners" → BEGINNER
    #   "Advanced setup" → ADVANCED
    #   "API documentation" → EXPERT
    parsed.skill_level = skill_level

    # 6. Ethical signal extraction
    ethical_signals = extract_ethical_signals(text)
    # Keywords:
    #   "privacy", "encrypted", "no tracking" → privacy
    #   "open source" → openness
    #   "sustainable", "carbon neutral" → sustainability
    parsed.ethical_signals = ethical_signals

    return parsed

def extract_constraints(text: str) -> List[Constraint]:
    """
    Rule-based constraint extraction without ML.
    """
    constraints = []

    # Dimension-specific regex patterns
    patterns = {
        'platform': [r'\b(Android|iOS|Windows|macOS|Linux)\b'],
        'price': [
            r'under\s+(\d+)',
            r'(\d+)\s*(?:rupees|dollars|EUR)',
            r'free|premium|enterprise'
        ],
        'provider': [
            r'\b(Google|Microsoft|Apple|Amazon|Meta|Facebook)\b',
            r'big\s+tech|FAANG'
        ],
        'license': [
            r'\b(MIT|GPL|Apache|BSD|Creative Commons)\b',
            r'open\s+source|proprietary|closed'
        ],
        'region': [r'\b(India|US|EU|APAC|Singapore)\b'],
        'language': [r'\b(Python|JavaScript|Java|Rust|Go|C\+\+)\b'],
        'format': [r'\b(PDF|CSV|JSON|XML|HTML)\b'],
        'recency': [
            r'(2024|2025|this year|recent)',
            r'(2020|2021|older|historical)'
        ],
    }

    for dimension, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Detect inclusion vs exclusion
                before_text = text[max(0, match.start()-50):match.start()]
                is_excluded = bool(re.search(r'\b(no|not|without|avoid|exclude)\s+', before_text, re.IGNORECASE))

                constraint = Constraint(
                    type='exclusion' if is_excluded else 'inclusion',
                    dimension=dimension,
                    value=match.group(0),
                    hardFilter=True
                )
                constraints.append(constraint)

    return constraints

def classify_goal(text: str) -> IntentGoal:
    """
    Map text patterns to enumerated IntentGoal.
    """
    goal_patterns = {
        IntentGoal.LEARN: [
            r'\bhow\s+to\b',
            r'\blearn\s+',
            r'\btutorial\b',
            r'\bexplain\b',
            r'\bbasics\b',
        ],
        IntentGoal.TROUBLESHOOTING: [
            r'\berror\b',
            r'\bnot\s+work',
            r'\bbroken\b',
            r'\bfix\b',
            r'\bdebug\b',
        ],
        IntentGoal.COMPARISON: [
            r'\bvs\.?\b',
            r'\bcompare\b',
            r'\bdifference\b',
            r'\bbetter\b',
            r'\balternative\b',
        ],
        IntentGoal.PURCHASE: [
            r'\bbuy\b',
            r'\bprice\b',
            r'\bwhere\s+to\s+get\b',
            r'\bcost\b',
        ],
        IntentGoal.LOCAL_SERVICE: [
            r'\bnear\s+me\b',
            r'\blocal\b',
            r'\bin\s+\w+\s+(?:city|area)\b',
        ],
    }

    for goal, patterns in goal_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return goal

    # Default: FIND_INFORMATION if no pattern matched
    return IntentGoal.FIND_INFORMATION
```

#### 1B. Form & Structured Field Parsing

When a user fills a form or interacts with structured fields (e.g., search filters):

```python
def parse_form_intent(form_data: Dict[str, Any], context: Context) -> ParsedIntent:
    """
    Extract intent from form submissions (no text).
    Example: User fills "Skills: Python, Rust" | "Budget: <5000" | "Company Size: Startup"
    """

    parsed = ParsedIntent()

    # Direct constraint mapping
    for field_name, field_value in form_data.items():
        if field_value is not None and field_value != "":
            constraint = map_field_to_constraint(field_name, field_value)
            parsed.constraints.append(constraint)

    # Infer goal from form context
    # Example: resume builder form → goal = CREATE
    form_name = context.get('form_name', '')
    parsed.goal = infer_goal_from_form(form_name)

    return parsed

def map_field_to_constraint(field_name: str, field_value: Any) -> Constraint:
    """
    Example mappings:
    - field "skills" with value "Python" → constraint(skill=Python)
    - field "max_budget" with value 5000 → constraint(price <= 5000)
    - field "company_size" with value "startup" → constraint(company_size=startup)
    """
    dimension_map = {
        'platform': 'platform',
        'budget': 'price',
        'price': 'price',
        'skills': 'skill',
        'language': 'language',
        'experience': 'skill_level',
        'location': 'region',
    }

    dimension = dimension_map.get(field_name.lower(), field_name.lower())

    return Constraint(
        type='inclusion',
        dimension=dimension,
        value=field_value,
        hardFilter=True
    )
```

#### 1C. Document State Parsing

```python
def parse_document_context(doc_id: str, doc_state: DocumentState) -> DocumentContext:
    """
    Extract intent signals from an open document without reading all content.
    Only read: title, first 1000 chars, metadata (NOT persisted).
    """

    doc_context = DocumentContext()
    doc_context.docId = doc_id

    # Read minimal content
    doc_context.content = doc_state.content[:1000]  # First 1KB only
    doc_context.lastEditTime = doc_state.last_edit_time
    doc_context.collaborators = len(doc_state.collaborator_list)
    doc_context.contentType = infer_content_type(doc_state.mime_type)

    # Goal inference from document state
    # If doc has numerical data and user opens "Data" → analyze
    # If multiple collaborators and user leaves comment → collaborate

    return doc_context
```

### Phase 2: Inference (No Behavioral Profiling)

#### 2A. Use Case Inference from Linguistic Patterns

```python
def detect_use_cases(text: str, context: Context) -> List[UseCase]:
    """
    Infer use cases from question structure and keywords.
    NO user history, NO profile data—only current context.
    """

    use_cases = []

    # Linguistic pattern detection
    patterns = {
        UseCase.COMPARISON: [
            r'\b(?:vs\.|versus|compare|compared\s+to|difference|better)\b',
            r'\b(?:which|what\'s the difference)\b',
        ],
        UseCase.LEARNING: [
            r'\b(?:how\s+to|tutorial|guide|explain|learn|introduction)\b',
            r'\b(?:what\s+is|basics|beginner|fundamentals)\b',
        ],
        UseCase.TROUBLESHOOTING: [
            r'\b(?:error|not\s+working|broken|fix|debug|why\s+is)\b',
            r'\b(?:issue|problem|what\s+went\s+wrong)\b',
        ],
        UseCase.VERIFICATION: [
            r'\b(?:is\s+it\s+true|fact.*check|verify|confirm|source)\b',
            r'\b(?:recent|news|latest|breaking)\b',
        ],
        UseCase.COMMUNITY_ENGAGEMENT: [
            r'\b(?:ask|question|opinion|thoughts|anyone|community)\b',
            r'\b(?:forum|reddit|discussion)\b',
        ],
    }

    for use_case, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, text, re.IGNORECASE):
                use_cases.append(use_case)
                break  # Avoid duplicates per use case

    # Fallback
    if not use_cases:
        use_cases.append(UseCase.LEARNING)

    return list(set(use_cases))  # Deduplicate
```

#### 2B. Temporal Intent Inference

```python
def extract_temporal_intent(text: str) -> TemporalIntent:
    """
    Infer time horizon, recency preference, and frequency from textual and contextual clues.
    """

    temporal = TemporalIntent()

    # Horizon detection
    horizon_patterns = {
        'immediate': [r'\b(?:now|today|urgent|asap|right\s+away)\b'],
        'today': [r'\b(?:today|this\s+morning|tonight)\b'],
        'week': [r'\b(?:this\s+week|next\s+week|few\s+days)\b'],
        'month': [r'\b(?:this\s+month|next\s+month|a\s+month)\b'],
        'longterm': [r'\b(?:next\s+year|long[\s-]?term|future)\b'],
    }

    for horizon_type, patterns in horizon_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                temporal.horizon = horizon_type
                break

    # Default based on goal
    if not temporal.horizon:
        temporal.horizon = 'flexible'

    # Recency preference
    if re.search(r'\b(?:latest|recent|breaking|2024|2025)\b', text, re.IGNORECASE):
        temporal.recency = 'recent'
    elif re.search(r'\b(?:evergreen|classic|fundamentals|basics)\b', text, re.IGNORECASE):
        temporal.recency = 'evergreen'
    else:
        temporal.recency = 'recent'  # Default

    # Frequency
    if re.search(r'\b(?:every|weekly|daily|recurring|repeating)\b', text, re.IGNORECASE):
        temporal.frequency = 'recurring'
    else:
        temporal.frequency = 'oneoff'

    return temporal
```

#### 2C. Complexity & Skill Level Matching

```python
def infer_skill_level(text: str) -> str:
    """
    Infer user's implied technical proficiency from language patterns.
    """

    advanced_indicators = [
        r'\b(?:API|CLI|regex|algorithm|optimization|memory\s+leak)\b',
        r'\b(?:database|SQL|concurrency|threading|async)\b',
        r'\b(?:architecture|design\s+pattern|microservices)\b',
    ]

    beginner_indicators = [
        r'\b(?:what\s+is|basics|beginner|simple|easy)\b',
        r'\b(?:first\s+time|never\s+used|newbie|explain\s+like)\b',
        r'\b(?:step\s+by\s+step|walkthrough)\b',
    ]

    if any(re.search(pattern, text, re.IGNORECASE) for pattern in advanced_indicators):
        return 'advanced'
    elif any(re.search(pattern, text, re.IGNORECASE) for pattern in beginner_indicators):
        return 'beginner'
    else:
        return 'intermediate'
```

#### 2D. Ethical Signal Extraction

```python
def extract_ethical_signals(text: str) -> List[EthicalSignal]:
    """
    Detect privacy, sustainability, openness preferences from explicit mentions.
    """

    signals = []

    ethical_patterns = {
        'privacy': {
            'keywords': [r'\b(?:privacy|encrypted|no\s+tracking|anonymous|confidential)\b'],
            'preference': 'privacy-first'
        },
        'openness': {
            'keywords': [r'\b(?:open\s+source|open\s+access|transparent|no\s+proprietary)\b'],
            'preference': 'open-source_preferred'
        },
        'sustainability': {
            'keywords': [r'\b(?:carbon\s+neutral|green|sustainable|eco|environmental)\b'],
            'preference': 'carbon-aware'
        },
        'accessibility': {
            'keywords': [r'\b(?:accessible|inclusive|blind|deaf|wcag|a11y)\b'],
            'preference': 'accessibility-first'
        },
        'ethics': {
            'keywords': [r'\b(?:ethical|fair|bias|discrimination|human\s+rights)\b'],
            'preference': 'ethical'
        },
    }

    for dimension, config in ethical_patterns.items():
        for keyword_pattern in config['keywords']:
            if re.search(keyword_pattern, text, re.IGNORECASE):
                signals.append(EthicalSignal(
                    dimension=dimension,
                    preference=config['preference']
                ))
                break

    return signals
```

### Phase 3: Normalization & Schema Assignment

```python
def normalize_to_schema(parsed: ParsedIntent, context: Context) -> UniversalIntent:
    """
    Combine parsed and inferred signals into the UniversalIntent schema.
    """

    import uuid
    from datetime import datetime, timedelta

    now = datetime.utcnow()
    session_ttl = timedelta(hours=8)  # Intent expires at session end

    intent = UniversalIntent()

    # Identifiers (session-scoped only)
    intent.intentId = f"intent_{uuid.uuid4().hex[:12]}"
    intent.context = IntentContext(
        product=context.product,
        timestamp=now.isoformat(),
        sessionId=context.session_id,
        userLocale=context.locale,
    )

    # Declared intent
    intent.declared = DeclaredIntent(
        query=parsed.query or None,
        goal=parsed.goal,
        constraints=parsed.constraints,
        negativePreferences=parsed.negative_preferences or [],
        urgency=map_urgency(parsed.temporal_intent.horizon),
        budget=parsed.budget or None,
        skillLevel=parsed.skill_level,
    )

    # Inferred intent
    intent.inferred = InferredIntent(
        useCases=parsed.use_cases,
        temporalIntent=parsed.temporal_intent,
        resultType=map_result_type(parsed.goal, parsed.use_cases),
        complexity=map_complexity(parsed.skill_level),
        ethicalSignals=parsed.ethical_signals,
    )

    # Session feedback (empty at creation, populated on interaction)
    intent.sessionFeedback = SessionFeedback()

    # TTL
    intent.expiresAt = (now + session_ttl).isoformat()

    return intent

def map_urgency(horizon: str) -> str:
    """Map temporal horizon to urgency enum."""
    urgency_map = {
        'immediate': 'immediate',
        'today': 'immediate',
        'week': 'soon',
        'month': 'flexible',
        'longterm': 'flexible',
    }
    return urgency_map.get(horizon, 'flexible')

def map_result_type(goal: IntentGoal, use_cases: List[UseCase]) -> str:
    """Infer result type from goal and use cases."""

    result_type_map = {
        IntentGoal.LEARN: 'tutorial',
        IntentGoal.COMPARISON: 'comparison',
        IntentGoal.TROUBLESHOOTING: 'tutorial',
        IntentGoal.PURCHASE: 'marketplace',
        IntentGoal.LOCAL_SERVICE: 'service_listing',
        IntentGoal.DRAFT_DOCUMENT: 'tool',
        IntentGoal.COLLABORATE: 'tool',
        IntentGoal.CREATE: 'tool',
        IntentGoal.REFLECT: 'community',
    }

    return result_type_map.get(goal, 'answer')

def map_complexity(skill_level: str) -> str:
    """Map skill level to complexity."""
    return skill_level  # Simple 1-to-1 map
```

---

## RANKING & MATCHING LOGIC

### Search Ranking Algorithm

**Core Principle**: Rank results by **intent-to-content alignment**, not historical engagement or user identity.

```python
def rank_search_results(
    intent: UniversalIntent,
    candidate_results: List[SearchResult],
    index: SearchIndex
) -> List[RankedResult]:
    """
    Rank search results based on:
    1. Constraint satisfaction (hard filters)
    2. Intent alignment (goal, use case, result type)
    3. Quality signals (freshness, authority, accessibility)
    4. Ethical alignment
    """

    ranked = []

    for result in candidate_results:

        # Step 1: Apply hard filters (constraints)
        if not satisfies_constraints(result, intent.declared.constraints):
            continue  # Skip this result

        # Step 2: Compute intent alignment score
        alignment_score = compute_intent_alignment(result, intent)

        # Step 3: Quality signals (no tracking)
        quality_score = compute_quality_score(result, intent)

        # Step 4: Ethical alignment
        ethical_score = compute_ethical_alignment(result, intent.inferred.ethicalSignals)

        # Step 5: Combine scores
        final_score = (
            0.50 * alignment_score +
            0.30 * quality_score +
            0.20 * ethical_score
        )

        ranked.append(RankedResult(
            result=result,
            score=final_score,
            reasons=[...]  # Explainability
        ))

    # Sort descending by score
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked

def satisfies_constraints(result: SearchResult, constraints: List[Constraint]) -> bool:
    """
    Check if result satisfies ALL hard constraints.
    """

    for constraint in constraints:
        if not satisfies_constraint(result, constraint):
            return False

    return True

def satisfies_constraint(result: SearchResult, constraint: Constraint) -> bool:
    """
    Check one constraint.
    Extract result metadata (title, snippet, domain, structured data) and compare.
    """

    if constraint.type == 'inclusion':
        # Result must contain the value in its metadata
        result_text = f"{result.title} {result.snippet} {result.domain}"
        return constraint.value.lower() in result_text.lower()

    elif constraint.type == 'exclusion':
        # Result must NOT contain the value
        result_text = f"{result.title} {result.snippet} {result.domain}"
        return constraint.value.lower() not in result_text.lower()

    elif constraint.type == 'range':
        # Numeric comparison (e.g., price)
        result_value = extract_numeric_value(result, constraint.dimension)
        if result_value is None:
            return False
        min_val, max_val = constraint.value
        return min_val <= result_value <= max_val

    return True

def compute_intent_alignment(result: SearchResult, intent: UniversalIntent) -> float:
    """
    Score how well the result aligns with user's declared intent.
    Uses semantic similarity (no tracking, no user history).
    """

    score = 0.0

    # 1. Goal alignment (result type vs expected type)
    result_type = classify_result_type(result)  # Infer from URL structure, content
    if result_type == intent.inferred.resultType:
        score += 0.25

    # 2. Use case alignment
    for use_case in intent.inferred.useCases:
        if result_contains_use_case(result, use_case):
            score += 0.15 / len(intent.inferred.useCases)

    # 3. Skill level match (avoid beginner results for experts and vice versa)
    if matches_skill_level(result, intent.declared.skillLevel):
        score += 0.20

    # 4. Temporal alignment
    if matches_temporal_preference(result, intent.inferred.temporalIntent):
        score += 0.15

    # 5. Recency preference
    if matches_recency(result, intent.inferred.temporalIntent.recency):
        score += 0.10

    return min(score, 1.0)  # Clamp to [0, 1]

def classify_result_type(result: SearchResult) -> str:
    """
    Infer result type from URL patterns and content snippets.
    Examples:
    - github.com/* → code/documentation
    - youtube.com/* → video
    - reddit.com/* → community
    - amazon.com/* → marketplace
    - wikipedia.org/* → reference
    """

    domain = extract_domain(result.url)

    type_map = {
        'github.com': 'code',
        'stackoverflow.com': 'community',
        'reddit.com': 'community',
        'medium.com': 'article',
        'arxiv.org': 'research',
        'youtube.com': 'video',
        'amazon.com': 'marketplace',
        'ebay.com': 'marketplace',
        'wikipedia.org': 'reference',
        'pytorch.org': 'documentation',
    }

    for domain_keyword, type_label in type_map.items():
        if domain_keyword in domain:
            return type_label

    # Default heuristic: analyze URL structure
    if '/docs/' in result.url or '/documentation/' in result.url:
        return 'documentation'
    elif '/tutorial' in result.url or '/guide/' in result.url:
        return 'tutorial'
    else:
        return 'article'

def compute_quality_score(result: SearchResult, intent: UniversalIntent) -> float:
    """
    Score based on non-tracking quality signals:
    - Domain authority (public page rank, .edu, .gov)
    - Recency (last crawl date vs query temporal intent)
    - Content completeness (length, structure)
    - Accessibility (alt text, captions, language)
    """

    score = 0.0

    # Domain authority (public signals only)
    domain_authority = get_domain_authority(result.domain)  # Public data
    if domain_authority > 50:
        score += 0.25
    elif domain_authority > 30:
        score += 0.15
    else:
        score += 0.05

    # Recency
    days_old = (datetime.utcnow() - result.last_crawl).days
    if days_old < 7:
        score += 0.25  # Fresh
    elif days_old < 30:
        score += 0.15
    elif days_old < 365:
        score += 0.05

    # Content completeness
    if len(result.snippet) > 150:  # Substantial content
        score += 0.25

    # Accessibility (has alt text, captions, etc.)
    if result.accessibility_signals:
        score += 0.20

    return min(score, 1.0)

def compute_ethical_alignment(result: SearchResult, ethical_signals: List[EthicalSignal]) -> float:
    """
    Score based on ethical signal matching.
    Example: User declares "privacy-first" → rank privacy-respecting sites higher.
    """

    score = 0.5  # Neutral baseline

    if not ethical_signals:
        return score  # No ethical preferences declared

    for signal in ethical_signals:
        if signal.dimension == 'privacy':
            # Check if domain is privacy-respecting (public reputation signals)
            if is_privacy_respecting_domain(result.domain):
                score += 0.25

        elif signal.dimension == 'openness':
            # Preference for open-source projects
            if 'github.com' in result.domain or 'opensource' in result.url.lower():
                score += 0.25

        elif signal.dimension == 'accessibility':
            # Preference for accessible content
            if result.accessibility_signals:
                score += 0.25

    return min(score, 1.0)
```

### Cross-Product Ranking: Service Selection

When a user has ambiguous intent, the Intent Engine recommends which service(s) are most relevant.

```python
def recommend_services(
    intent: UniversalIntent,
    available_services: Dict[str, Service]
) -> List[ServiceRecommendation]:
    """
    Given a user's intent, recommend the best workspace services to use.
    Example: If intent suggests "collaboration" → recommend Docs > Meet > Mail
    """

    recommendations = []

    for service_name, service in available_services.items():

        # Score intent match for this service
        match_score = compute_service_match(intent, service)

        # Add to recommendations
        if match_score > 0.3:  # Threshold
            recommendations.append(ServiceRecommendation(
                service=service,
                score=match_score,
                reason=explain_match(intent, service)
            ))

    # Sort by score
    recommendations.sort(key=lambda x: x.score, reverse=True)
    return recommendations

def compute_service_match(intent: UniversalIntent, service: Service) -> float:
    """
    Score service relevance based on intent goal and use cases.
    """

    # Mapping of intent goals to services
    service_affinity = {
        'search': {
            IntentGoal.FIND_INFORMATION: 0.9,
            IntentGoal.COMPARISON: 0.8,
            IntentGoal.TROUBLESHOOTING: 0.8,
            IntentGoal.LEARN: 0.7,
        },
        'docs': {
            IntentGoal.DRAFT_DOCUMENT: 0.95,
            IntentGoal.COLLABORATE: 0.90,
            IntentGoal.CREATE: 0.85,
            IntentGoal.ANALYZE: 0.75,
        },
        'mail': {
            IntentGoal.ORGANIZE: 0.85,
            IntentGoal.COLLABORATE: 0.70,
        },
        'calendar': {
            IntentGoal.SCHEDULE: 0.95,
            IntentGoal.COLLABORATE: 0.75,
        },
        'meet': {
            IntentGoal.COLLABORATE: 0.90,
            IntentGoal.LEARN: 0.75,
        },
        'forms': {
            IntentGoal.COLLECT: 0.95,
            IntentGoal.ANALYZE: 0.80,
        },
        'diary': {
            IntentGoal.REFLECT: 0.95,
        },
    }

    # Base score from goal affinity
    base_score = service_affinity.get(service.name, {}).get(intent.declared.goal, 0.3)

    # Boost for use cases
    use_case_boost = 0.0
    for use_case in intent.inferred.useCases:
        if use_case in service.supported_use_cases:
            use_case_boost += 0.1

    return min(base_score + use_case_boost, 1.0)
```

### Ad Matching Algorithm

**Core Principle**: Match ads based on **declared intent only**, never on user identity or history.

```python
def match_ads(
    intent: UniversalIntent,
    ad_inventory: List[Ad],
    constraints: MatchingConstraints
) -> List[MatchedAd]:
    """
    Match relevant ads to user intent.

    Ad matching respects:
    1. User's hard constraints (exclusions, budget, region, etc.)
    2. Advertiser's hard constraints (target audience demographics NOT allowed)
    3. Relevance to declared goal + use case
    4. User's ethical preferences

    NO: User history, behavioral profiles, third-party data
    NO: Proxy discrimination via sensitive attributes
    """

    matched = []

    for ad in ad_inventory:

        # Filter 1: User's hard constraints
        if not satisfies_user_constraints(ad, intent.declared.constraints):
            continue

        # Filter 2: Advertiser's hard constraints (without discriminating)
        if not satisfies_advertiser_constraints(ad, intent):
            continue

        # Filter 3: Relevance score
        relevance_score = compute_ad_relevance(ad, intent)

        if relevance_score > 0.4:  # Minimum relevance threshold
            matched.append(MatchedAd(
                ad=ad,
                relevance_score=relevance_score,
                matchedIntentDimensions=[...]  # Transparency
            ))

    # Sort by relevance
    matched.sort(key=lambda x: x.relevance_score, reverse=True)
    return matched[:5]  # Top 5 ads max

def satisfies_user_constraints(ad: Ad, constraints: List[Constraint]) -> bool:
    """
    Ensure ad respects user's explicit filters.
    """
    for constraint in constraints:
        if constraint.type == 'exclusion':
            # User excluded this category
            if ad.category == constraint.value or ad.brand == constraint.value:
                return False

    return True

def satisfies_advertiser_constraints(ad: Ad, intent: UniversalIntent) -> bool:
    """
    Validate advertiser's constraints don't violate fairness.

    ALLOWED constraints (non-discriminatory):
    - Geographic region (India, EU, US)
    - Language (English, Hindi, Spanish)
    - Device type (mobile, desktop)
    - Declared intent (users searching for "running shoes")

    FORBIDDEN constraints (discriminatory):
    - Age, gender, race, religion, sexuality
    - Income, credit score, parental status
    - Health conditions, political affiliation
    - Behavioral profiling ("users who viewed X")
    """

    # Check for forbidden constraints
    forbidden_attrs = [
        'age', 'gender', 'race', 'religion', 'sexuality',
        'income', 'parental_status', 'health_condition',
        'behavioral_segment', 'lookalike_audience'
    ]

    for constraint in ad.advertiser_constraints:
        if constraint.dimension in forbidden_attrs:
            return False  # Reject ad with forbidden targeting

    # Allowed: geographic, device, intent-based targeting
    return True

def compute_ad_relevance(ad: Ad, intent: UniversalIntent) -> float:
    """
    Score ad relevance based on intent match.
    """

    score = 0.0

    # 1. Goal match
    if ad.primary_goal == intent.declared.goal:
        score += 0.40

    # 2. Use case match
    for use_case in intent.inferred.useCases:
        if use_case in ad.target_use_cases:
            score += 0.15 / len(ad.target_use_cases)

    # 3. Keyword relevance
    query_words = intent.declared.query.split() if intent.declared.query else []
    ad_keywords = set(ad.keywords)
    matching_keywords = len(set(query_words) & ad_keywords)
    if matching_keywords > 0:
        score += 0.20 * (matching_keywords / len(query_words))

    # 4. Ethical alignment
    if intent.inferred.ethicalSignals:
        # Ad respects ethical preferences
        if is_ad_ethical(ad, intent.inferred.ethicalSignals):
            score += 0.10

    # 5. Skill level match
    if matches_skill_level(ad, intent.declared.skillLevel):
        score += 0.10

    return min(score, 1.0)
```

---

## PRIVACY & THREAT MODEL

### Data Retention & TTL

| Data | Retention | Location | Deletion |
|---|---|---|---|
| **Intent objects** | Session duration (8 hours max) | In-memory only | Auto-delete at session end |
| **Session feedback** (clicks, dwell) | Session duration | In-memory only | Auto-delete at session end |
| **Ranking logs** (for system improvement) | 7 days | Encrypted server-side | Auto-purge after 7 days |
| **Ad impressions** | 30 days (aggregated only) | Encrypted server-side | Auto-purge after 30 days |
| **Search index** | Indefinite | Searx/self-hosted index | User can request deletion |
| **User configuration** | Indefinite | Encrypted at-rest | User can request deletion |

### Threat Model

#### Threat 1: Intent Inference Side Channels

**Attack**: Adversary reconstructs user behavior from repeated intent queries.

**Example**: If user queries "budgeting for IVF treatment" weekly, repeat requests leak health intent.

**Mitigation**:
- Intent TTL expires intent after session (no historical queries stored)
- Aggregate statistics only (no individual query logs)
- Session IDs are randomized (not linkable across time)
- No cross-session intent correlation

#### Threat 2: Ad Matching Privacy Leakage

**Attack**: Advertiser deduces user intent from ad metrics (impressions, clicks, conversions).

**Example**: "Users searching for antidepressants" audience → inference of user health.

**Mitigation**:
- No persistent audience segments (one-time matching per session)
- Advertiser sees only: ad relevance score, aggregate conversion rate
- No feedback on why specific user saw ad (only aggregate patterns)
- Conversion reporting uses differential privacy (noise added to counts)

#### Threat 3: Document Context Leakage

**Attack**: System reads open document titles/content to infer intent, creating surveillance.

**Example**: Read "Cancer Research Proposal" from open Doc → infer user health status.

**Mitigation**:
- Read document titles only for intent inference (not content)
- Read first 1000 chars of content for context; don't persist
- Don't profile individual documents
- User can opt-out: "Don't infer intent from open documents"

#### Threat 4: Cross-Product Intent Linking

**Attack**: Combine intents across Services (Search + Mail + Docs) to profile users.

**Example**: Search for "rental apartments" + Mail from realty.com + Docs named "Housing Plans" → infer housing market intent and sell to real estate ads.

**Mitigation**:
- Intent objects are product-scoped (Search intent ≠ Docs intent)
- No cross-product intent aggregation
- Each product has independent session
- Users can't be profiled across services

#### Threat 5: Temporal Inference

**Attack**: Correlate timing of intents across sessions to infer behavior.

**Example**: User creates Docs titled "Resignation Letter" → infer job transition → sell executive recruiter ads.

**Mitigation**:
- Session IDs expire after 8 hours
- No persistent session links
- Each new session is a fresh intent context
- No temporal correlation across sessions

### Privacy Compliance Checklist

| Regulation | Requirement | Implementation |
|---|---|---|
| **GDPR** | Right to deletion, consent | No persistent profiles; users control intent signals |
| **GDPR** | Data portability | Export intent schema as JSON |
| **CCPA** | Opt-out of "sale" | Users can disable ad personalization (intent matching only) |
| **CCPA** | Non-discrimination | No price/service discrimination based on age, income, etc. |
| **India DPDP** | Processing purpose limitation | Intent used only for matching/ranking, not profiling |
| **India DPDP** | Data minimization | Read minimal doc content; don't persist |
| **India DPDP** | Anonymization | Intent objects not linked to persistent user ID |

---

## DATA MODELS & TABLES

### Intent Storage (In-Memory)

```sql
-- Session-scoped, ephemeral table
CREATE TABLE intent (
    intent_id STRING PRIMARY KEY,
    session_id STRING NOT NULL,
    product STRING NOT NULL, -- search, docs, mail, calendar, meet, forms, diary
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,

    -- Declared intent (JSON)
    declared_intent JSON NOT NULL, -- goal, constraints, urgency, skill_level, etc.

    -- Inferred intent (JSON)
    inferred_intent JSON NOT NULL, -- use_cases, temporal_intent, ethical_signals, etc.

    -- Session feedback (mutable)
    session_feedback JSON, -- clicks, dwell_time, bounced, reformulated

    -- Index for cleanup
    CREATE INDEX idx_expires_at ON intent(expires_at)
);

-- Auto-delete expired intents
CREATE PROCEDURE cleanup_expired_intents AS
    DELETE FROM intent WHERE expires_at < NOW();

CALL cleanup_expired_intents(); -- Scheduled every 1 hour
```

### Ranking Logs (Server-Side, Encrypted, 7-day TTL)

```sql
CREATE TABLE ranking_log (
    log_id STRING PRIMARY KEY,
    session_id STRING NOT NULL,
    intent_id STRING NOT NULL, -- Reference to intent object
    query_text STRING, -- Only if search intent
    product STRING,

    -- Anonymized result metadata
    result_urls STRING[], -- URLs only, no titles or snippets
    result_scores FLOAT64[], -- Relevance scores
    ranking_reasons JSON, -- Why result ranked high (for system improvement)

    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL, -- 7 days

    CREATE INDEX idx_expires_at ON ranking_log(expires_at)
);
```

### Ad Impression & Conversion (Aggregated, Differential Privacy)

```sql
CREATE TABLE ad_impressions_aggregated (
    ad_id STRING NOT NULL,
    date DATE NOT NULL,

    -- Intent dimensions (aggregated, no user identity)
    intent_goal STRING, -- e.g., 'PURCHASE'
    intent_use_case STRING, -- e.g., 'COMPARISON'
    advertiser_id STRING NOT NULL,

    -- Aggregated metrics (differential privacy applied)
    impression_count INT64, -- With noise added
    click_count INT64, -- With noise added
    conversion_count INT64, -- With noise added

    PRIMARY KEY (ad_id, date, intent_goal, intent_use_case)
);
```

### Service Recommendation (Stateless, Computed Real-Time)

```sql
-- No persistent storage—computed on-demand from intent schema
-- Example pseudocode:
SELECT
    service_name,
    COMPUTE_SERVICE_MATCH(intent, service) AS match_score
FROM services
WHERE COMPUTE_SERVICE_MATCH(intent, service) > 0.3
ORDER BY match_score DESC;
```

---

## SYSTEM ARCHITECTURE DIAGRAM (TEXT)

### High-Level Interaction Flow

```
User Interaction
    │
    ├─→ [Search Query]
    ├─→ [Docs Edit]
    ├─→ [Mail Compose]
    ├─→ [Calendar Event]
    ├─→ [Meet Invite]
    └─→ [Diary Entry]
    │
    ▼
Intent Extraction Engine (on-device or local)
    │
    ├─→ Parse (Regex, NLP, form parsing)
    ├─→ Infer (Linguistic patterns, temporal, skill level, ethics)
    └─→ Normalize (Map to UniversalIntent schema)
    │
    ▼
UniversalIntent Object
    {
      intentId, context,
      declared {goal, constraints, urgency, skill_level},
      inferred {use_cases, temporal, ethical_signals},
      sessionFeedback {},
      expiresAt
    }
    │
    ├─────────────┬──────────────┬─────────────┐
    │             │              │             │
    ▼             ▼              ▼             ▼
  Search      Docs          Ads          Service
  Ranking    Ranking      Matching    Recommendation
    │             │              │             │
    ├─→ Apply constraints
    ├─→ Score intent alignment
    ├─→ Score quality (no tracking)
    ├─→ Score ethical alignment
    └─→ Rank / Match / Recommend
    │
    ▼
Result Rendering
    - Ranked search results
    - Document suggestions
    - Matched ads (intent-based)
    - Recommended services
    │
    ▼
User Session Feedback (captured locally)
    - Clicks, dwell time, bounce
    - Query reformulation
    [Deleted at session end]
```

### Data Flow (Privacy-Preserving)

```
┌──────────────────────────────────────────────────────────┐
│                    USER DEVICE                           │
│  (Local: No persistent identity, no external tracking)   │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Intent Extraction (On-Device)                      │ │
│  │ - Parse query/form/doc/meeting                     │ │
│  │ - Infer use case, temporal, skill level            │ │
│  │ - Output: UniversalIntent object                   │ │
│  │ - TTL: Session-scoped                              │ │
│  └────────────────────────────────────────────────────┘ │
│                       │                                 │
│                       ▼                                 │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Ranking/Matching (On-Device)                       │ │
│  │ - Apply user constraints (hard filters)            │ │
│  │ - Score result alignment (no tracking)             │ │
│  │ - Determine ad relevance (declared intent only)    │ │
│  │ - Recommend services (cross-product routing)       │ │
│  └────────────────────────────────────────────────────┘ │
│                       │                                 │
│                       ▼                                 │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Session Feedback (In-Memory)                       │ │
│  │ - User clicks, dwell time, bounces                 │ │
│  │ - Deleted at session end (no history)              │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
         │                                    │
         │ [Aggregated & Encrypted]           │
         ▼                                    ▼
    ┌─────────────┐              ┌──────────────────┐
    │  Ranking    │              │  Ad Metrics      │
    │  Logs       │              │  (Aggregate)     │
    │  (7-day TTL)│              │  (30-day TTL)    │
    └─────────────┘              └──────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ Differential Privacy │
                              │ (Noise for privacy)  │
                              └──────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ Advertiser Reports   │
                              │ (Aggregate patterns) │
                              └──────────────────────┘
```

---

## IMPLEMENTATION ROADMAP

> **Note:** The following roadmap is a historical document outlining the original implementation plan.

### Phase 1: Core Intent Schema & Extraction (Months 1–2)

**Deliverables**:
- UniversalIntent schema (TypeScript types)
- Intent extraction algorithm (Python reference implementation)
- Unit tests for extraction accuracy
- Schema validation library

**Key Tasks**:
1. Implement text parsing (constraints, goals, use cases)
2. Implement linguistic inference (temporal, skill level, ethics)
3. Integrate with Search product (MVP)
4. Build intent schema visualization tool (transparency)

### Phase 2: Cross-Product Integration (Months 3–4)

**Deliverables**:
- Intent extraction for Docs, Mail, Calendar, Meet
- Service recommendation algorithm
- Cross-product demos

**Key Tasks**:
1. Parse Docs edit events → intent
2. Parse Mail compose state → intent
3. Parse Calendar events → intent
4. Implement service ranking logic
5. Build product routing demo

### Phase 3: Ranking & Ad Matching (Months 5–6)

**Deliverables**:
- Search ranking algorithm
- Ad matching algorithm
- Fairness constraints (no discrimination)
- Performance benchmarks

**Key Tasks**:
1. Implement constraint satisfaction
2. Implement intent alignment scoring
3. Implement ad matching logic
4. Design fairness tests (no proxy discrimination)
5. Benchmark ranking speed

### Phase 4: Privacy & Security (Months 7–8)

**Deliverables**:
- TTL enforcement
- Data deletion pipeline
- Threat model validation
- Privacy audit

**Key Tasks**:
1. Implement session-scoped storage
2. Implement auto-deletion at session end
3. Encrypt server-side logs
4. Differential privacy for ad metrics
5. Security review by external auditors

### Phase 5: Production & Monitoring (Months 9–12)

**Deliverables**:
- Production deployment
- Monitoring dashboards
- User education materials
- Transparency reports

**Key Tasks**:
1. Deploy to production clusters
2. Build monitoring (extraction accuracy, ranking quality)
3. A/B test vs. historical ranking
4. Publish transparency reports (quarterly)
5. User feedback loops

---

## COMPARISON: Intent-First vs. Identity-Based Systems

| Dimension | Google (Identity-Based) | Intent Engine (Intent-First) |
|---|---|---|
| **Data Collection** | Behavioral profiling, cross-site tracking, login linking | Declared intent, local inference (no tracking) |
| **Signal Lifetime** | Persistent (years, cookies) | Ephemeral (session-scoped, <8 hours) |
| **User Control** | Limited (settings buried in 40-page settings) | Native (intent signals visible, editable) |
| **Discrimination Risk** | High (proxy discrimination via inferred attributes) | Lower (constraints are explicit, auditable) |
| **GDPR Compliance** | Requires consent, tracking infrastructure | Non-tracking by design, minimal consent |
| **Advertiser Surveillance** | High (detailed audience insights) | Low (aggregate metrics only, no audience insights) |
| **User Privacy** | ✓ No persistent tracking | ✓ Session-scoped intent only |
| **Transparency** | Limited (algorithmic opacity) | High (intent extraction rules inspectable) |
| **Bias Auditing** | Difficult (opaque behavioral signals) | Easier (explicit constraints auditable) |
| **Inference Accuracy** | High (uses user history) | Moderate (uses current context only) |
| **Ranking Quality** | Good (personalized via history) | Good (intent-matched, no history needed) |
| **Ad Relevance** | Good (behavior-based) | Good (intent-based, fewer irrelevant ads) |

---

## CONCLUSION

The **Intent Engine** is a foundational system that enables privacy-first ranking, ads, and cross-product service discovery without behavioral profiling or user tracking. By operating on **ephemeral, declared, and locally-inferred intent signals**, the system achieves:

1. **Privacy**: No persistent user profiles; intent signals expire at session end
2. **Fairness**: Explicit constraints prevent proxy discrimination
3. **Transparency**: Intent extraction rules are inspectable; users control signals
4. **Quality**: Intent-based ranking and matching are competitive with history-based systems
5. **Compliance**: Naturally satisfies GDPR, CCPA, India DPDP without invasive tracking

This architecture forms the foundation for building a **Google-like workspace ecosystem that is fundamentally more ethical and privacy-preserving** than incumbent solutions.

---

## REFERENCES & FURTHER READING

1. **Differential Privacy in Advertising**: Erlingsson et al., "Learning Differentially Private Recurrent Language Models" (ICLR 2020)
2. **Privacy-First Ad Matching**: https://github.com/topics/privacy-preserving-ads
3. **Searx/SearxNG**: https://github.com/searxng/searxng (privacy-first metasearch)
4. **Fair ML**: https://fairmlbook.org
5. **Schema Design**: https://schema.org (structured data standards)
6. **GDPR/CCPA Compliance**: https://gdpr-info.eu, https://cpra.ca.gov

---

**Document Version**: 1.0
**Last Updated**: January 19, 2026
**Status**: Ready for Implementation
**Author**: Generalist AI Architect
