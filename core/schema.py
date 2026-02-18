"""
Intent Engine - Core Schema Definitions

This module defines the universal intent schema and related data structures
used across all components of the Intent Engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# Define enums matching the TypeScript definitions from the docs
class IntentGoal(Enum):
    # Search-specific
    FIND_INFORMATION = "find_information"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"
    PURCHASE = "purchase"
    LOCAL_SERVICE = "local_service"
    NAVIGATION = "navigation"

    # Docs/Mail-specific
    DRAFT_DOCUMENT = "draft_document"
    COLLABORATE = "collaborate"
    ORGANIZE = "organize"
    ANALYZE = "analyze"
    SCHEDULE = "schedule"

    # Cross-product
    LEARN = "learn"
    CREATE = "create"
    REFLECT = "reflect"  # Diary


class UseCase(Enum):
    COMPARISON = "comparison"
    LEARNING = "learning"
    TROUBLESHOOTING = "troubleshooting"
    VERIFICATION = "verification"
    ENTERTAINMENT = "entertainment"
    COMMUNITY_ENGAGEMENT = "community_engagement"
    PROFESSIONAL_DEVELOPMENT = "professional_development"
    MARKET_RESEARCH = "market_research"


class ConstraintType(Enum):
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"
    RANGE = "range"
    DATATYPE = "datatype"


class Urgency(Enum):
    IMMEDIATE = "immediate"
    SOON = "soon"
    FLEXIBLE = "flexible"
    EXPLORATORY = "exploratory"


class SkillLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TemporalHorizon(Enum):
    IMMEDIATE = "immediate"
    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    LONGTERM = "longterm"
    FLEXIBLE = "flexible"


class Recency(Enum):
    BREAKING = "breaking"
    RECENT = "recent"
    EVERGREEN = "evergreen"
    HISTORICAL = "historical"


class Frequency(Enum):
    ONEOFF = "oneoff"
    RECURRING = "recurring"
    EXPLORATORY = "exploratory"
    FLEXIBLE = "flexible"


class EthicalDimension(Enum):
    PRIVACY = "privacy"
    SUSTAINABILITY = "sustainability"
    ETHICS = "ethics"
    ACCESSIBILITY = "accessibility"
    OPENNESS = "openness"


class ResultType(Enum):
    ANSWER = "answer"
    TUTORIAL = "tutorial"
    TOOL = "tool"
    MARKETPLACE = "marketplace"
    COMMUNITY = "community"


class Complexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    ADVANCED = "advanced"


class ContentType(Enum):
    TEXT = "text"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    FORM = "form"


@dataclass
class Constraint:
    """Represents a constraint extracted from user input"""

    type: ConstraintType
    dimension: str  # 'language', 'region', 'price', 'license', 'format', 'recency'
    value: (
        str | int | float | list[str | int | float] | list[int]
    )  # Single value, range, or list
    hardFilter: bool  # Must exclude results violating this


@dataclass
class TemporalIntent:
    """Temporal aspects of user intent"""

    horizon: TemporalHorizon
    recency: Recency
    frequency: Frequency


@dataclass
class DocumentContext:
    """Context from open documents"""

    docId: str | None = None
    content: str | None = None  # First 1000 chars only, not persisted
    lastEditTime: str | None = None
    collaborators: int | None = None  # Count only, not names
    contentType: ContentType | None = None


@dataclass
class MeetingContext:
    """Context from calendar/meetings"""

    meetingId: str | None = None
    subject: str | None = None
    participantCount: int | None = None
    isRecurring: bool | None = None
    timeZone: str | None = None


@dataclass
class EthicalSignal:
    """Ethical preferences extracted from intent"""

    dimension: EthicalDimension
    preference: str  # "privacy-first", "open-source", "carbon-neutral", etc.


@dataclass
class DeclaredIntent:
    """User-declared intent components"""

    query: str | None = None  # Free-form text
    goal: IntentGoal | None = None  # Structured goal
    constraints: list[Constraint] = field(default_factory=list)  # Hard filters
    negativePreferences: list[str] = field(default_factory=list)  # "not X", "no Y"
    urgency: Urgency = Urgency.FLEXIBLE
    budget: str | None = None  # "under 1000", "premium", null
    skillLevel: SkillLevel = SkillLevel.INTERMEDIATE


@dataclass
class InferredIntent:
    """Inferred intent components"""

    useCases: list[UseCase] = field(
        default_factory=list
    )  # [comparison, learning, troubleshooting, ...]
    temporalIntent: TemporalIntent | None = None
    documentContext: DocumentContext | None = None  # From open docs/emails
    meetingContext: MeetingContext | None = None  # From calendar/Meet
    resultType: ResultType | None = None
    complexity: Complexity = Complexity.MODERATE
    ethicalSignals: list[EthicalSignal] = field(
        default_factory=list
    )  # Privacy, sustainability, etc.


@dataclass
class SessionFeedback:
    """Feedback captured during the session"""

    clicked: list[str] | None = None  # URLs clicked
    dwell: int | None = None  # Seconds on result
    reformulated: bool | None = None  # User refined query
    bounced: bool | None = None  # Left immediately


@dataclass
class UniversalIntent:
    """Main intent object matching the schema from the whitepaper"""

    # Unique session-scoped ID (not persistent)
    intentId: str

    # Product context (which service generated this)
    context: dict[str, Any]

    # Declared intent (user-supplied constraints and goals)
    declared: DeclaredIntent

    # Inferred intent (derived from context without tracking)
    inferred: InferredIntent

    # Feedback (captured in current session only)
    sessionFeedback: SessionFeedback = field(default_factory=SessionFeedback)

    # TTL: Auto-delete after session ends
    expiresAt: str = ""


@dataclass
class IntentExtractionRequest:
    """Request object for intent extraction API"""

    product: str  # 'search' | 'docs' | 'mail' | 'calendar' | 'meet' | 'forms' | 'diary' | 'sites'
    input: dict[str, str]  # TextInput | FormInput | DocumentInput | EventInput
    context: dict[str, Any]  # ExtractionContext
    options: dict[str, Any] | None = None  # ExtractionOptions


@dataclass
class IntentExtractionResponse:
    """Response object for intent extraction API"""

    intent: UniversalIntent
    extractionMetrics: dict[str, Any]  # confidence, extractedDimensions, warnings
