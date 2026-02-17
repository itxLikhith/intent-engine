"""
Intent Engine - Core Schema Definitions

This module defines the universal intent schema and related data structures
used across all components of the Intent Engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


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
    value: Union[str, int, float, List[Union[str, int, float]], List[int]]  # Single value, range, or list
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

    docId: Optional[str] = None
    content: Optional[str] = None  # First 1000 chars only, not persisted
    lastEditTime: Optional[str] = None
    collaborators: Optional[int] = None  # Count only, not names
    contentType: Optional[ContentType] = None


@dataclass
class MeetingContext:
    """Context from calendar/meetings"""

    meetingId: Optional[str] = None
    subject: Optional[str] = None
    participantCount: Optional[int] = None
    isRecurring: Optional[bool] = None
    timeZone: Optional[str] = None


@dataclass
class EthicalSignal:
    """Ethical preferences extracted from intent"""

    dimension: EthicalDimension
    preference: str  # "privacy-first", "open-source", "carbon-neutral", etc.


@dataclass
class DeclaredIntent:
    """User-declared intent components"""

    query: Optional[str] = None  # Free-form text
    goal: Optional[IntentGoal] = None  # Structured goal
    constraints: List[Constraint] = field(default_factory=list)  # Hard filters
    negativePreferences: List[str] = field(default_factory=list)  # "not X", "no Y"
    urgency: Urgency = Urgency.FLEXIBLE
    budget: Optional[str] = None  # "under 1000", "premium", null
    skillLevel: SkillLevel = SkillLevel.INTERMEDIATE


@dataclass
class InferredIntent:
    """Inferred intent components"""

    useCases: List[UseCase] = field(default_factory=list)  # [comparison, learning, troubleshooting, ...]
    temporalIntent: Optional[TemporalIntent] = None
    documentContext: Optional[DocumentContext] = None  # From open docs/emails
    meetingContext: Optional[MeetingContext] = None  # From calendar/Meet
    resultType: Optional[ResultType] = None
    complexity: Complexity = Complexity.MODERATE
    ethicalSignals: List[EthicalSignal] = field(default_factory=list)  # Privacy, sustainability, etc.


@dataclass
class SessionFeedback:
    """Feedback captured during the session"""

    clicked: Optional[List[str]] = None  # URLs clicked
    dwell: Optional[int] = None  # Seconds on result
    reformulated: Optional[bool] = None  # User refined query
    bounced: Optional[bool] = None  # Left immediately


@dataclass
class UniversalIntent:
    """Main intent object matching the schema from the whitepaper"""

    # Unique session-scoped ID (not persistent)
    intentId: str

    # Product context (which service generated this)
    context: Dict[str, Any]

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
    input: Dict[str, str]  # TextInput | FormInput | DocumentInput | EventInput
    context: Dict[str, Any]  # ExtractionContext
    options: Optional[Dict[str, Any]] = None  # ExtractionOptions


@dataclass
class IntentExtractionResponse:
    """Response object for intent extraction API"""

    intent: UniversalIntent
    extractionMetrics: Dict[str, Any]  # confidence, extractedDimensions, warnings
