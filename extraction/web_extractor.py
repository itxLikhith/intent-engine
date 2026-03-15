"""
Intent Engine - Web Content Intent Extraction

Extracts intent metadata from web pages for indexing.
Used by the Go crawler to tag indexed content with intent categories.

Features:
- Goal detection (LEARN, COMPARISON, TROUBLESHOOTING, etc.)
- Use case extraction
- Skill level assessment
- Topic extraction
- Complexity analysis
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from core.schema import (
    Complexity,
    IntentGoal,
    ResultType,
    UseCase,
)

logger = logging.getLogger(__name__)


@dataclass
class WebIntent:
    """Intent metadata for a web page"""

    url: str
    primary_goal: IntentGoal = IntentGoal.FIND_INFORMATION
    use_cases: list[UseCase] = field(default_factory=list)
    result_type: ResultType = ResultType.ANSWER
    complexity: Complexity = Complexity.MODERATE
    skill_level: str = "intermediate"  # beginner, intermediate, advanced
    topics: list[str] = field(default_factory=list)
    confidence: float = 0.5
    title: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "url": self.url,
            "primary_goal": self.primary_goal.value,
            "use_cases": [uc.value for uc in self.use_cases],
            "result_type": self.result_type.value,
            "complexity": self.complexity.value,
            "skill_level": self.skill_level,
            "topics": self.topics,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
        }


class WebIntentExtractor:
    """
    Extracts intent metadata from web page content.

    Usage:
        extractor = WebIntentExtractor()
        intent = await extractor.extract_from_url(url)
        # or
        intent = extractor.extract_from_content(url, html_content)

    Features:
    - Rule-based goal classification
    - Keyword-based use case detection
    - Readability-based complexity analysis
    - TF-based topic extraction
    """

    def __init__(self):
        # Intent keyword dictionaries for goal detection
        self.goal_keywords = {
            IntentGoal.LEARN: [
                "tutorial",
                "guide",
                "how to",
                "howto",
                "learn",
                "introduction",
                "basics",
                "fundamentals",
                "course",
                "lesson",
                "step by step",
                "walkthrough",
                "beginner",
                "getting started",
            ],
            IntentGoal.COMPARISON: [
                "vs",
                "versus",
                "compare",
                "comparison",
                "better",
                "alternative",
                "review",
                "top 10",
                "best",
                "difference between",
                "which is better",
                "head to head",
            ],
            IntentGoal.TROUBLESHOOTING: [
                "fix",
                "error",
                "problem",
                "issue",
                "troubleshoot",
                "debug",
                "solve",
                "not working",
                "broken",
                "help",
                "won't",
                "cant",
                "can't",
            ],
            IntentGoal.PURCHASE: [
                "buy",
                "price",
                "purchase",
                "deal",
                "discount",
                "cheap",
                "where to buy",
                "cost",
                "pricing",
            ],
            IntentGoal.FIND_INFORMATION: [
                "what is",
                "definition",
                "explain",
                "overview",
                "introduction",
                "about",
                "meaning",
                "examples",
            ],
        }

        # Use case keywords
        self.use_case_keywords = {
            UseCase.LEARNING: [
                "tutorial",
                "learn",
                "study",
                "course",
                "education",
                "training",
            ],
            UseCase.COMPARISON: [
                "compare",
                "review",
                "best",
                "top",
                "versus",
                "alternative",
            ],
            UseCase.TROUBLESHOOTING: [
                "fix",
                "error",
                "problem",
                "issue",
                "solution",
            ],
            UseCase.PROFESSIONAL_DEVELOPMENT: [
                "career",
                "professional",
                "skills",
                "certification",
                "job",
            ],
            UseCase.MARKET_RESEARCH: [
                "market",
                "industry",
                "trends",
                "analysis",
                "report",
            ],
        }

        # Result type indicators
        self.result_type_indicators = {
            ResultType.TUTORIAL: ["tutorial", "guide", "steps", "walkthrough"],
            ResultType.TOOL: ["tool", "software", "app", "application", "calculator"],
            ResultType.MARKETPLACE: [
                "buy",
                "price",
                "purchase",
                "shop",
                "store",
                "deal",
            ],
            ResultType.ANSWER: ["what is", "definition", "answer", "meaning"],
            ResultType.COMMUNITY: [
                "forum",
                "discussion",
                "reddit",
                "community",
                "ask",
            ],
        }

        # Skill level indicators
        self.beginner_keywords = [
            "beginner",
            "introduction",
            "basics",
            "fundamentals",
            "101",
            "for dummies",
            "getting started",
            "easy",
            "simple",
        ]

        self.advanced_keywords = [
            "advanced",
            "expert",
            "deep dive",
            "master",
            "professional",
            "enterprise",
            "complex",
            "optimization",
            "architecture",
        ]

        # Stopwords for topic extraction
        self.stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "as",
            "if",
            "then",
            "than",
            "so",
            "such",
            "not",
            "no",
            "yes",
            "you",
            "your",
            "we",
            "our",
            "they",
            "their",
            "he",
            "she",
            "his",
            "her",
        }

    async def extract_from_url(self, url: str) -> Optional[WebIntent]:
        """
        Extract intent from a web page.

        Args:
            url: URL of the web page

        Returns:
            WebIntent with extracted metadata, or None if extraction failed
        """
        try:
            # Fetch content
            content = await self._fetch_content(url)
            if not content:
                logger.warning(f"Failed to fetch content from {url}")
                return None

            # Extract intent
            return self.extract_from_content(url, content)

        except Exception as e:
            logger.error(f"Error extracting intent from {url}: {e}")
            return None

    def extract_from_content(self, url: str, content: str) -> WebIntent:
        """
        Extract intent from page content.

        Args:
            url: URL of the page
            content: HTML or text content

        Returns:
            WebIntent with extracted metadata
        """
        # Clean content (remove HTML tags if present)
        text_content = self._clean_html(content)

        # Extract title
        title = self._extract_title(content)

        # Extract description (first 200 chars)
        description = text_content[:200] if len(text_content) > 200 else text_content

        # Analyze content (use first 5000 chars for efficiency)
        content_lower = text_content[:5000].lower()

        # Detect primary goal
        primary_goal = self._detect_goal(content_lower)

        # Detect use cases
        use_cases = self._detect_use_cases(content_lower)

        # Detect result type
        result_type = self._detect_result_type(content_lower)

        # Detect complexity
        complexity = self._detect_complexity(text_content)

        # Detect skill level
        skill_level = self._detect_skill_level(content_lower)

        # Extract topics (simple TF-based)
        topics = self._extract_topics(content_lower)

        # Calculate confidence
        confidence = self._calculate_confidence(primary_goal, use_cases, topics)

        return WebIntent(
            url=url,
            primary_goal=primary_goal,
            use_cases=use_cases,
            result_type=result_type,
            complexity=complexity,
            skill_level=skill_level,
            topics=topics,
            confidence=confidence,
            title=title,
            description=description,
        )

    def _detect_goal(self, content: str) -> IntentGoal:
        """Detect primary intent goal from content"""
        goal_scores = {}

        for goal, keywords in self.goal_keywords.items():
            score = sum(1 for kw in keywords if kw in content)
            goal_scores[goal] = score

        # Return highest scoring goal
        if goal_scores:
            best_goal = max(goal_scores, key=goal_scores.get)
            # Only return if we have a minimum score
            if goal_scores[best_goal] >= 2:
                return best_goal

        return IntentGoal.FIND_INFORMATION

    def _detect_use_cases(self, content: str) -> list[UseCase]:
        """Detect use cases from content"""
        detected = []

        for use_case, keywords in self.use_case_keywords.items():
            if any(kw in content for kw in keywords):
                detected.append(use_case)

        return detected if detected else [UseCase.LEARNING]

    def _detect_result_type(self, content: str) -> ResultType:
        """Detect expected result type from content"""
        for result_type, indicators in self.result_type_indicators.items():
            if any(kw in content for kw in indicators):
                return result_type

        return ResultType.ANSWER

    def _detect_complexity(self, content: str) -> Complexity:
        """Detect content complexity based on length and structure"""
        # Word count
        words = content.split()
        word_count = len(words)

        # Sentence count (approximate)
        sentence_count = len(re.findall(r"[.!?]+", content))

        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else word_count

        # Simple heuristics
        if word_count < 500 or avg_sentence_length < 10:
            return Complexity.SIMPLE
        elif word_count > 2000 or avg_sentence_length > 25:
            return Complexity.ADVANCED
        else:
            return Complexity.MODERATE

    def _detect_skill_level(self, content: str) -> str:
        """Detect target skill level from content"""
        beginner_score = sum(1 for kw in self.beginner_keywords if kw in content)
        advanced_score = sum(1 for kw in self.advanced_keywords if kw in content)

        if advanced_score > beginner_score:
            return "advanced"
        elif beginner_score > 0:
            return "beginner"
        else:
            return "intermediate"

    def _extract_topics(self, content: str) -> list[str]:
        """
        Extract main topics using simple term frequency.

        Args:
            content: Lowercased text content

        Returns:
            List of top topics (keywords)
        """
        # Tokenize
        words = re.findall(r"\b[a-z]{3,}\b", content)

        # Count frequency (excluding stopwords)
        word_freq = {}
        for word in words:
            if word not in self.stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # Return top 5 topics
        return [topic for topic, _ in sorted_topics[:5]]

    def _calculate_confidence(
        self, goal: IntentGoal, use_cases: list[UseCase], topics: list[str]
    ) -> float:
        """Calculate extraction confidence score"""
        base_confidence = 0.5

        # Boost for clear goal detection (not default)
        if goal != IntentGoal.FIND_INFORMATION:
            base_confidence += 0.15

        # Boost for multiple use cases
        if len(use_cases) > 1:
            base_confidence += 0.1 * min(len(use_cases), 3)

        # Boost for clear topics
        if len(topics) >= 3:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _clean_html(self, content: str) -> str:
        """Remove HTML tags from content"""
        # Simple HTML tag removal
        text = re.sub(r"<[^>]+>", " ", content)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_title(self, content: str) -> Optional[str]:
        """Extract page title from HTML"""
        match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    async def _fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch web page content.

        Args:
            url: URL to fetch

        Returns:
            HTML content or None if fetch failed
        """
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                    allow_redirects=True,
                ) as response:
                    if response.status == 200:
                        return await response.text(errors="ignore")
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")

        return None


# Singleton instance
_extractor_instance: Optional[WebIntentExtractor] = None


def get_web_intent_extractor() -> WebIntentExtractor:
    """Get or create web intent extractor singleton"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = WebIntentExtractor()
    return _extractor_instance
