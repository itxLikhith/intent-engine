"""
Intent Engine - Phase 1: Intent Extractor Module

This module implements the core intent extraction functionality as described in the technical reference.
It extracts structured intent from free-form user queries using hybrid parsing (rule-based + semantic inference).
"""

import re
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
import logging
import numpy as np
from functools import lru_cache

from core.schema import (
    IntentGoal, UseCase, ConstraintType, Urgency, SkillLevel, TemporalHorizon,
    Recency, Frequency, EthicalDimension, ResultType, Complexity, ContentType,
    Constraint, TemporalIntent, DocumentContext, MeetingContext, EthicalSignal,
    DeclaredIntent, InferredIntent, SessionFeedback, UniversalIntent,
    IntentExtractionRequest, IntentExtractionResponse
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstraintExtractor:
    """
    Extracts constraints from user queries using rule-based regex patterns
    """
    
    def __init__(self):
        # Platform constraints
        self.platform_patterns = {
            r'\b(android|mobile|phone)\b': ('platform', 'Android'),
            r'\b(ios|iphone|ipad|apple)\b': ('platform', 'iOS'),
            r'\b(windows|pc|desktop)\b': ('platform', 'Windows'),
            r'\b(mac|macos|macbook)\b': ('platform', 'macOS'),
            r'\b(linux|ubuntu|debian|fedora)\b': ('platform', 'Linux'),
            r'\b(web|browser|chrome|firefox|safari)\b': ('platform', 'Web'),
        }
        
        # Provider exclusion constraints
        self.exclusion_patterns = {
            r'\b(no\s+(google|gmail|android))\b': ('provider', 'Google'),
            r'\b(no\s+(microsoft|outlook|windows))\b': ('provider', 'Microsoft'),
            r'\b(no\s+(apple|ios|iphone|ipad))\b': ('provider', 'Apple'),
            r'\b(no\s+(big\s+tech|big\s+corporations?))\b': ('provider', ['Google', 'Microsoft', 'Apple', 'Amazon', 'Meta']),
            r'\b(not?\s+google|avoid\s+google)\b': ('provider', 'Google'),
            r'\b(not?\s+microsoft|avoid\s+microsoft)\b': ('provider', 'Microsoft'),
            r'\b(proprietary|closed\s+source)\b': ('license', 'proprietary'),
            r'\b(open\s+source|oss|free\s+software)\b': ('license', 'open-source'),
        }
        
        # Price constraints
        self.price_patterns = {
            r'\b(under|less than|below)\s*(\d+)\s*(rupees|rs|₹|dollars?|usd)\b': ('price', '<={value}'),
            r'\b(over|more than|above)\s*(\d+)\s*(rupees|rs|₹|dollars?|usd)\b': ('price', '>={value}'),
            r'\b(free|gratis|no cost|zero cost)\b': ('price', '0'),
            r'\b(budget|cheap|affordable|low cost)\b': ('price', 'budget'),
        }
        
        # Feature constraints
        self.feature_patterns = {
            r'\b(end[-\s]*to[-\s]*end[-\s]*encrypt|e2e[-\s]*encrypt|end[-\s]*to[-\s]*end[-\s]*encrypted|e2e[-\s]*encrypted)\b': ('feature', 'end-to-end_encryption'),
            r'\b(end[-\s]*to[-\s]*end|e2e)\s+(encrypt|encryption)\b': ('feature', 'end-to-end_encryption'),
            r'\b(encrypted|secure|private)\s+(email|mail)\b': ('feature', 'encrypted_email'),
            r'\b(real[-\s]*time|instant)\s+(sync|collaboration)\b': ('feature', 'real-time_collaboration'),
            r'\b(offline|local|on[-\s]*device)\s+(storage|sync)\b': ('feature', 'offline_capability'),
            r'\b(ad[-\s]*free|no\s+ads|ad[-\s]*less)\b': ('feature', 'ad-free'),
        }
    
    def extract_constraints(self, text: str) -> List[Constraint]:
        """
        Extract constraints from the input text using regex patterns
        """
        constraints = []
        seen_constraints = set()  # To avoid duplicates
        text_lower = text.lower()
        
        # Extract platform constraints
        for pattern, (dimension, value) in self.platform_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                constraint_key = (ConstraintType.INCLUSION.value, dimension, str(value))
                if constraint_key not in seen_constraints:
                    constraints.append(Constraint(
                        type=ConstraintType.INCLUSION,
                        dimension=dimension,
                        value=value,
                        hardFilter=True
                    ))
                    seen_constraints.add(constraint_key)
        
        # Extract exclusion constraints
        for pattern, (dimension, value) in self.exclusion_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                constraint_key = (ConstraintType.EXCLUSION.value, dimension, str(value))
                if constraint_key not in seen_constraints:
                    constraints.append(Constraint(
                        type=ConstraintType.EXCLUSION,
                        dimension=dimension,
                        value=value,
                        hardFilter=True
                    ))
                    seen_constraints.add(constraint_key)
        
        # Extract price constraints
        for pattern, (dimension, template) in self.price_patterns.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    value_str = match[1]  # The numeric value
                    actual_template = template.format(value=value_str)
                    constraint_key = (ConstraintType.RANGE.value, dimension, actual_template)
                    if constraint_key not in seen_constraints:
                        constraints.append(Constraint(
                            type=ConstraintType.RANGE,
                            dimension=dimension,
                            value=actual_template,
                            hardFilter=True
                        ))
                        seen_constraints.add(constraint_key)
        
        # Extract feature constraints - need to handle overlaps carefully
        # Only allow one feature constraint per feature TYPE to avoid duplicates
        # FIX: Track by value (feature type) not dimension to properly deduplicate
        feature_types_seen = set()
        for pattern, (dimension, value) in self.feature_patterns.items():
            matches = re.findall(pattern, text_lower)
            # FIX: Check if this specific feature value was already extracted
            if matches and value not in feature_types_seen:
                constraint_key = (ConstraintType.INCLUSION.value, dimension, str(value))
                if constraint_key not in seen_constraints:
                    constraints.append(Constraint(
                        type=ConstraintType.INCLUSION,
                        dimension=dimension,
                        value=value,
                        hardFilter=True
                    ))
                    seen_constraints.add(constraint_key)
                    # FIX: Track by value, not dimension (was always "feature")
                    feature_types_seen.add(value)

        return constraints
    
    def extract_negative_preferences(self, text: str) -> List[str]:
        """
        Extract negative preferences from the text
        """
        text_lower = text.lower()
        negative_prefs = []
        
        # Look for common negative preference patterns
        if re.search(r'\b(no\s+big\s+tech|no\s+big\s+corporations?)\b', text_lower):
            negative_prefs.append("no big tech")
        if re.search(r'\b(no\s+proprietary|no\s+closed\s+source|open\s+source)\b', text_lower):
            negative_prefs.append("no proprietary")
        if re.search(r'\b(privacy[-\s]*first|privacy\s+focused)\b', text_lower):
            negative_prefs.append("privacy-focused")
        
        return negative_prefs


class GoalClassifier:
    """
    Classifies user intent goals using regex patterns
    """
    
    def __init__(self):
        # Define patterns for different goals
        self.goal_patterns = {
            IntentGoal.LEARN: [
                r'\b(how to|how do i|guide|tutorial|setup|configure|install|learn|teach me|explain|what is|tell me about)\b',
                r'\b(setup|configur|install|learn|tutorial|guide|manual|instructions?)\b',
                r'\b(explain|understand|know about|find out about)\b'
            ],
            IntentGoal.COMPARISON: [
                r'\b(compare|comparing|versus|vs\.?|difference|better|best|alternative|alternatives?)\b',
                r'\b(which is|should i use|recommend|top|best|vs|versus)\b'
            ],
            IntentGoal.TROUBLESHOOTING: [
                r'\b(fix|broken|not working|error|problem|issue|trouble|debug|cant|can\'t|won\'t|help)\b',
                r'\b(why is|not working|fix|troubleshoot|solve|resolve)\b'
            ],
            IntentGoal.PURCHASE: [
                r'\b(buy|purchase|get|order|shop|price|cost|deal|discount|cheapest|where to buy)\b',
                r'\b(price|cost|buy|purchase|order|where to get)\b'
            ],
            IntentGoal.FIND_INFORMATION: [
                r'\b(find|search|locate|where|when|who|what|information|details|about)\b',
                r'\b(what is|where is|when|who|information|details)\b'
            ],
            IntentGoal.LOCAL_SERVICE: [
                r'\b(near me|nearby|local|around here|closest|find near|find nearby)\b',
                r'\b(service|repair|mechanic|doctor|dentist|restaurant) near me\b'
            ],
            IntentGoal.NAVIGATION: [
                r'\b(go to|navigate to|directions|route|map|find location|address)\b',
                r'\b(directions|route|navigate|location|address|find the way)\b'
            ]
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for goal, patterns in self.goal_patterns.items():
            compiled_list = [re.compile(p, re.IGNORECASE) for p in patterns]
            self.compiled_patterns[goal] = compiled_list
    
    def classify_goal(self, text: str) -> Optional[IntentGoal]:
        """
        Classify the goal from the input text using regex patterns
        """
        text_lower = text.lower()
        
        for goal, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    return goal
        
        return None  # No goal detected


class SkillLevelDetector:
    """
    Detects user skill level from query text
    """
    
    def detect_skill_level(self, text: str) -> SkillLevel:
        """
        Detect skill level based on keywords in the text
        """
        text_lower = text.lower()
        
        # Beginner indicators
        beginner_keywords = [
            r'\b(beginner|novice|newbie|new to|first time|just starting|basic|fundamental|simple)\b',
            r'\b(help me|need help|how do|what is|explain|for dummies|easy|simple)\b'
        ]
        
        # Advanced indicators
        advanced_keywords = [
            r'\b(advanced|expert|power user|custom|optimize|performance|technical|configuration|advanced settings)\b',
            r'\b(technical|configuration|advanced|expert|performance|optimization)\b'
        ]
        
        # Count matches for each category
        beginner_count = sum(len(re.findall(kw, text_lower)) for kw in beginner_keywords)
        advanced_count = sum(len(re.findall(kw, text_lower)) for kw in advanced_keywords)
        
        # Determine skill level based on counts
        if advanced_count > beginner_count:
            return SkillLevel.ADVANCED
        elif beginner_count > 0:
            return SkillLevel.BEGINNER
        else:
            return SkillLevel.INTERMEDIATE


class SemanticInferenceEngine:
    """
    Performs semantic inference using sentence transformers (mock implementation for now)
    In production, this would use ONNX runtime or quantized GGML for CPU efficiency
    """
    
    def __init__(self):
        # Define use case examples for semantic matching
        self.use_case_examples = {
            UseCase.COMPARISON: [
                "compare different email providers",
                "difference between services",
                "which option is better",
                "evaluate alternatives"
            ],
            UseCase.LEARNING: [
                "how to set up email",
                "tutorial for beginners",
                "learn about encryption",
                "guide to configure"
            ],
            UseCase.TROUBLESHOOTING: [
                "fix email not working",
                "troubleshoot connection issues",
                "solve error messages",
                "debug problems"
            ],
            UseCase.VERIFICATION: [
                "verify account",
                "confirm email address",
                "validate credentials",
                "check security"
            ],
            UseCase.PROFESSIONAL_DEVELOPMENT: [
                "advance career skills",
                "learn new technology",
                "professional growth",
                "career development"
            ]
        }
        
        # Define ethical signal examples
        self.ethical_signal_examples = {
            "privacy-first": [
                "privacy protection",
                "data security",
                "no tracking",
                "private browsing",
                "secure communication",
                "privacy focused",
                "privacy-first approach",
                "protecting user privacy"
            ],
            "open-source_preferred": [
                "open source software",
                "free and open",
                "community driven",
                "transparent code",
                "open development",
                "open source solution"
            ],
            "ethical_company": [
                "ethical business",
                "social responsibility",
                "environmental impact",
                "fair labor practices",
                "sustainable operations"
            ]
        }
    
    def infer_use_cases(self, query: str) -> List[UseCase]:
        """
        Infer use cases based on semantic similarity to example texts
        This is a simplified version - in production would use embeddings
        """
        query_lower = query.lower()
        inferred_use_cases = []
        
        for use_case, examples in self.use_case_examples.items():
            # Simple keyword matching for now - in production would use embeddings
            for example in examples:
                if any(word in query_lower for word in example.split()[:3]):  # Match first few words
                    inferred_use_cases.append(use_case)
                    break
        
        return inferred_use_cases
    
    def infer_ethical_signals(self, query: str) -> List[EthicalSignal]:
        """
        Infer ethical signals based on semantic similarity to example texts
        This is a simplified version - in production would use embeddings
        """
        query_lower = query.lower()
        inferred_signals = []
        
        for preference, examples in self.ethical_signal_examples.items():
            # Simple keyword matching for now - in production would use embeddings
            for example in examples:
                if any(word in query_lower for word in example.split()[:3]):  # Match first few words
                    # Map preference string back to EthicalDimension enum
                    if "privacy" in preference:
                        dimension = EthicalDimension.PRIVACY
                    elif "open" in preference:
                        dimension = EthicalDimension.OPENNESS
                    elif "sustain" in preference:
                        dimension = EthicalDimension.SUSTAINABILITY
                    else:
                        dimension = EthicalDimension.ETHICS
                    
                    inferred_signals.append(EthicalSignal(dimension=dimension, preference=preference))
                    break
        
        return inferred_signals
    
    def infer_result_type(self, query: str) -> Optional[ResultType]:
        """
        Infer the expected result type based on query content
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how to', 'tutorial', 'guide', 'setup', 'configure']):
            return ResultType.TUTORIAL
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'alternative']):
            return ResultType.COMMUNITY  # Community discussions often compare products
        elif any(word in query_lower for word in ['buy', 'purchase', 'price', 'cost', 'deal']):
            return ResultType.MARKETPLACE
        elif any(word in query_lower for word in ['what is', 'explain', 'define', 'describe']):
            return ResultType.ANSWER
        else:
            return ResultType.TOOL  # Default to tool for most technical queries
    
    def infer_complexity(self, query: str) -> Complexity:
        """
        Infer the complexity level of the query
        """
        # Simple heuristics based on query characteristics
        if len(query.split()) <= 5:
            return Complexity.SIMPLE
        elif any(word in query.lower() for word in ['advanced', 'expert', 'technical', 'configuration', 'performance']):
            return Complexity.ADVANCED
        else:
            return Complexity.MODERATE


class IntentExtractor:
    """
    Main class that orchestrates intent extraction using all components
    Implements Algorithm 1: Text-to-Intent Extraction from the technical reference
    """
    
    def __init__(self):
        self.constraint_extractor = ConstraintExtractor()
        self.goal_classifier = GoalClassifier()
        self.skill_detector = SkillLevelDetector()
        self.semantic_engine = SemanticInferenceEngine()
    
    def extract_intent_from_request(self, request: IntentExtractionRequest) -> IntentExtractionResponse:
        """
        Main function to extract intent from request following Algorithm 1
        """
        text = request.input.get('text', '') if isinstance(request.input, dict) else str(request.input)
        session_id = request.context.get('sessionId', f"sess_{uuid.uuid4().hex}")
        user_locale = request.context.get('userLocale', 'en-US')
        
        # Phase 1: Constraint Extraction (Regex patterns)
        constraints = self.constraint_extractor.extract_constraints(text)
        negative_preferences = self.constraint_extractor.extract_negative_preferences(text)
        
        # Phase 2: Goal Classification (Keyword matching)
        goal = self.goal_classifier.classify_goal(text)
        
        # Phase 3: Skill Level Detection
        skill_level = self.skill_detector.detect_skill_level(text)
        
        # Phase 4: Semantic Inference (Embedding-based - simplified for now)
        use_cases = self.semantic_engine.infer_use_cases(text)
        ethical_signals = self.semantic_engine.infer_ethical_signals(text)
        result_type = self.semantic_engine.infer_result_type(text)
        complexity = self.semantic_engine.infer_complexity(text)
        
        # Determine urgency based on keywords
        urgency = self._infer_urgency(text)
        
        # Create temporal intent
        temporal_intent = self._infer_temporal_intent(text)
        
        # Create the UniversalIntent object
        intent = self._create_universal_intent(
            product=request.product,
            query=text,
            session_id=session_id,
            user_locale=user_locale,
            goal=goal,
            constraints=constraints,
            negative_preferences=negative_preferences,
            urgency=urgency,
            skill_level=skill_level,
            use_cases=use_cases,
            temporal_intent=temporal_intent,
            result_type=result_type,
            complexity=complexity,
            ethical_signals=ethical_signals
        )
        
        # Create response with metrics
        response = IntentExtractionResponse(
            intent=intent,
            extractionMetrics={
                'confidence': 0.8,  # Placeholder confidence
                'extractedDimensions': [c.dimension for c in constraints] + (['goal'] if goal else []),
                'warnings': []  # Add warnings if needed
            }
        )
        
        return response
    
    def _create_universal_intent(self, 
                                product: str, 
                                query: str, 
                                session_id: str, 
                                user_locale: str,
                                goal: Optional[IntentGoal] = None,
                                constraints: Optional[List[Constraint]] = None,
                                negative_preferences: Optional[List[str]] = None,
                                urgency: Urgency = Urgency.FLEXIBLE,
                                skill_level: SkillLevel = SkillLevel.INTERMEDIATE,
                                use_cases: Optional[List[UseCase]] = None,
                                temporal_intent: Optional[TemporalIntent] = None,
                                result_type: Optional[ResultType] = None,
                                complexity: Complexity = Complexity.MODERATE,
                                ethical_signals: Optional[List[EthicalSignal]] = None) -> UniversalIntent:
        """
        Create a UniversalIntent object with proper structure
        """
        intent_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
        
        # Calculate expiration time (8 hours from now)
        expires_at = (datetime.utcnow() + timedelta(hours=8)).isoformat() + "Z"
        
        context_dict = {
            "product": product,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sessionId": session_id,
            "userLocale": user_locale
        }
        
        return UniversalIntent(
            intentId=intent_id,
            context=context_dict,
            declared=DeclaredIntent(
                query=query,
                goal=goal,
                constraints=constraints or [],
                negativePreferences=negative_preferences or [],
                urgency=urgency,
                skillLevel=skill_level
            ),
            inferred=InferredIntent(
                useCases=use_cases or [],
                temporalIntent=temporal_intent,
                resultType=result_type,
                complexity=complexity,
                ethicalSignals=ethical_signals or []
            ),
            sessionFeedback=SessionFeedback(),
            expiresAt=expires_at
        )
    
    def _infer_urgency(self, text: str) -> Urgency:
        """
        Infer urgency level from text
        """
        text_lower = text.lower()
        
        immediate_keywords = ['urgent', 'now', 'immediately', 'right now', 'asap', 'emergency']
        soon_keywords = ['today', 'this week', 'soon', 'within days', 'quick']
        
        if any(word in text_lower for word in immediate_keywords):
            return Urgency.IMMEDIATE
        elif any(word in text_lower for word in soon_keywords):
            return Urgency.SOON
        else:
            return Urgency.FLEXIBLE
    
    def _infer_temporal_intent(self, text: str) -> TemporalIntent:
        """
        Infer temporal intent from text
        """
        text_lower = text.lower()
        
        # Determine horizon
        if any(word in text_lower for word in ['now', 'today', 'immediate', 'urgent']):
            horizon = TemporalHorizon.TODAY
        elif any(word in text_lower for word in ['this week', 'weekly', 'week']):
            horizon = TemporalHorizon.WEEK
        elif any(word in text_lower for word in ['monthly', 'month']):
            horizon = TemporalHorizon.MONTH
        else:
            horizon = TemporalHorizon.FLEXIBLE
        
        # Determine recency
        if any(word in text_lower for word in ['latest', 'new', 'recent', 'updated', 'current']):
            recency = Recency.RECENT
        elif any(word in text_lower for word in ['old', 'historical', 'past', 'archive']):
            recency = Recency.HISTORICAL
        else:
            recency = Recency.EVERGREEN
        
        # Determine frequency
        if any(word in text_lower for word in ['every', 'daily', 'weekly', 'monthly', 'recurring', 'repeat']):
            frequency = Frequency.RECURRING
        elif any(word in text_lower for word in ['once', 'single', 'one time', 'one-time']):
            frequency = Frequency.ONEOFF
        else:
            frequency = Frequency.EXPLORATORY
        
        return TemporalIntent(horizon=horizon, recency=recency, frequency=frequency)


# Global instance for caching
_intent_extractor_instance = None


def get_intent_extractor() -> IntentExtractor:
    """
    Get singleton instance of IntentExtractor with cached model
    """
    global _intent_extractor_instance
    if _intent_extractor_instance is None:
        _intent_extractor_instance = IntentExtractor()
    return _intent_extractor_instance


def extract_intent(request: IntentExtractionRequest) -> IntentExtractionResponse:
    """
    Main entry point for extracting intent from any input
    Matches the IntentExtractionRequest interface from the technical reference
    """
    extractor = get_intent_extractor()
    return extractor.extract_intent_from_request(request)