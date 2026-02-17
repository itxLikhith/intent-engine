# Intent Engine - Project Context

## Overview
The Intent Engine is a privacy-first, intent-driven system for search, service recommendation, and ad matching. It processes user queries to extract structured intent while respecting privacy and ethical considerations. The system is organized into a modular architecture with four main phases:

1. **Intent Extraction** - Converts free-form queries into structured intent objects
2. **Constraint Satisfaction & Ranking** - Filters and ranks results based on user intent
3. **Service Recommendation** - Routes users to the most appropriate service
4. **Ad Matching** - Matches ads without discriminatory targeting

## Architecture
The project follows a clean, modular structure:
```
intent_engine/
├── core/                   # Shared schema and utilities
│   ├── schema.py           # UniversalIntent class + enums
│   └── utils.py            # Shared helpers (caching, logging)
├── extraction/             # Intent extraction module
│   ├── extractor.py        # Main IntentExtraction logic
│   └── constraints.py      # Constraint parsing logic
├── ranking/                # Ranking module
│   ├── ranker.py           # Main Ranking logic
│   └── scoring.py          # Alignment/quality/ethical scoring
├── services/               # Service recommendation module
│   └── recommender.py      # Service recommendation logic
├── ads/                    # Ad matching module
│   └── matcher.py          # Ad matching logic
├── tests/                  # Unit tests
├── demos/                  # Demo scripts
├── perf_tests/             # Performance tests
├── config/                 # Configuration
│   └── model_cache.py      # Model loading/caching logic
├── main.py                 # Entry point for CLI or server
```

## Key Components

### Core Schema (`core/schema.py`)
Defines the `UniversalIntent` dataclass and associated enums that form the foundation of the system:
- `IntentGoal` - Purpose of the query (LEARN, COMPARE, TROUBLESHOOT, etc.)
- `UseCase` - Specific use cases (LEARNING, TROUBLESHOOTING, etc.)
- `ConstraintType` - Types of constraints (INCLUSION, EXCLUSION, RANGE)
- `EthicalDimension` - Privacy, sustainability, ethics considerations
- `UniversalIntent` - Main intent object with declared and inferred components

### Intent Extraction (`extraction/extractor.py`)
Converts free-form queries into structured intent objects using:
- Rule-based constraint extraction with regex patterns
- Goal classification based on keyword matching
- Skill level detection
- Semantic inference for use cases and ethical signals

### Ranking (`ranking/ranker.py`)
Implements constraint satisfaction and intent-aligned ranking:
- Filters results based on user constraints
- Computes alignment scores between results and user intent
- Applies weighted scoring for semantic similarity, ethical alignment, etc.

### Service Recommendation (`services/recommender.py`)
Routes users to the most appropriate service based on intent matching:
- Matches intent goals to service capabilities
- Considers use cases, temporal patterns, and ethical alignment

### Ad Matching (`ads/matcher.py`)
Performs ethical ad matching without tracking:
- Validates ads against fairness rules (no discriminatory targeting)
- Filters ads based on user constraints
- Scores ads for relevance and ethical alignment

## Building and Running

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the System
```bash
# Run all demos
python main.py demo

# Run specific demo
python main.py demo-search      # Intent extraction demo
python main.py demo-service     # Service recommendation demo
python main.py demo-ad          # Ad matching demo
python main.py demo-ranking     # Ranking demo

# Run tests
python main.py test

# Run performance tests
python main.py perf-test

# Extract intent from a query
python main.py extract --query "How to set up encrypted email on Android?"
```

### Programmatic Usage
```python
from extraction.extractor import extract_intent, IntentExtractionRequest

# Create extraction request
request = IntentExtractionRequest(
    product='search',
    input={'text': 'How to set up E2E encrypted email on Android, no big tech'},
    context={'sessionId': 'session_123', 'userLocale': 'en-US'}
)

# Extract intent
response = extract_intent(request)
intent = response.intent

print(f"Goal: {intent.declared.goal}")
print(f"Constraints: {intent.declared.constraints}")
print(f"Use Cases: {[uc.value for uc in intent.inferred.useCases]}")
```

## Dependencies
Key dependencies defined in `requirements.txt`:
- `transformers` and `sentence-transformers` for NLP
- `torch` for neural network operations
- `numpy` for numerical computations
- `pytest` for testing
- `fastapi` and `uvicorn` for serving (optional)

## Performance Characteristics
- **Warm-up time**: <100ms after initial load
- **Processing time**: <50ms per query after warm-up
- **Memory footprint**: <500MB RAM
- **CPU optimized**: Runs efficiently on standard hardware

## Privacy & Ethics Features
- **No user tracking**: No persistent user profiles or behavioral tracking
- **Local processing**: All processing happens on the user's device
- **Ethical ad matching**: No discriminatory targeting based on protected attributes
- **Data minimization**: Only processes what's necessary for the current session
- **Automatic cleanup**: Session data auto-deletes after 8 hours

## Testing
The system includes comprehensive unit tests covering all four phases:
- `tests/test_extraction.py` - Tests for intent extraction
- `tests/test_ranking.py` - Tests for ranking functionality
- `tests/test_services.py` - Tests for service recommendation
- `tests/test_ads.py` - Tests for ad matching

Run tests with: `python -m pytest tests/ -v`

## Development Conventions
- Follows Python PEP 8 style guidelines
- Uses type hints throughout for better code clarity
- Implements dataclasses for structured data representation
- Uses enums for categorical values to ensure consistency
- Includes comprehensive logging for debugging
- Maintains privacy-first design principles throughout