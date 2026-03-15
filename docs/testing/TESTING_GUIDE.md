# Testing Guide

Quick reference for running tests in the Intent Engine project.

## Quick Start

```bash
# Run all tests
python -m pytest tests/

# Run all tests with verbose output
python -m pytest tests/ -v

# Run all tests with coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

## Running Specific Tests

```bash
# Run specific test file
python -m pytest tests/test_extraction.py -v

# Run specific test class
python -m pytest tests/test_extraction.py::TestIntentExtractor -v

# Run specific test function
python -m pytest tests/test_extraction.py::TestIntentExtractor::test_goal_classification -v

# Run tests matching a pattern
python -m pytest tests/ -k "test_constraint" -v
```

## Test Categories

### Core Module Tests
```bash
# Intent Extraction
python -m pytest tests/test_extraction.py -v

# Ranking
python -m pytest tests/test_ranking.py -v

# Service Recommendation
python -m pytest tests/test_services.py -v

# Ad Matching
python -m pytest tests/test_ads.py -v
```

### API Tests
```bash
# Advertising API
python -m pytest tests/test_advertising_api.py -v

# URL Ranking API
python -m pytest tests/test_url_ranking_api.py -v

# Integration Tests
python -m pytest tests/comprehensive_test.py -v
```

### URL Ranking Tests
```bash
# URL Ranking (Unit Tests)
python -m pytest tests/test_url_ranking.py -v
```

## Advanced Options

```bash
# Stop after first failure
python -m pytest tests/ -x

# Show local variables on failure
python -m pytest tests/ -l

# Show print statements during tests
python -m pytest tests/ -s

# Run tests in parallel (requires pytest-xdist)
python -m pytest tests/ -n auto

# Rerun failed tests first
python -m pytest tests/ --lf

# Run tests that failed last time, then all others
python -m pytest tests/ --ff
```

## Coverage Reports

```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=. --cov-report=html

# Open coverage report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS
xdg-open htmlcov/index.html  # Linux

# Generate terminal coverage report
python -m pytest tests/ --cov=. --cov-report=term-missing
```

## Debugging Tests

```bash
# Start debugger on failures
python -m pytest tests/ --pdb

# Run with verbose logging
python -m pytest tests/ -v --log-cli-level=INFO

# Show capture output on failures
python -m pytest tests/ -rP
```

## Performance Testing

```bash
# Show slowest 10 tests
python -m pytest tests/ --durations=10

# Show all test durations
python -m pytest tests/ --durations=0

# Timeout tests after 60 seconds
python -m pytest tests/ --timeout=60
```

## Common Issues

### Import Errors
If you see import errors, ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Asyncio Errors
All async tests now use `asyncio.run()` - ensure you're using Python 3.11+

### Database Lock Errors
If using SQLite and encountering lock errors:
```bash
# Close any running API servers
# Delete the database file (default: intent_engine.db)
rm intent_engine.db  # or del intent_engine.db on Windows
# Re-run tests
```

## Test Structure

```
tests/
├── __init__.py
├── api_integration_test.py     # Additional integration tests
├── comprehensive_test.py       # API integration tests
├── test_ads.py                 # Ad matching unit tests
├── test_advertising_api.py     # Advertising API tests
├── test_comprehensive_queries.py  # Additional tests
├── test_extraction.py          # Intent extraction tests
├── test_ranking.py             # Ranking algorithm tests
├── test_services.py            # Service recommendation tests
├── test_url_ranking.py         # URL ranking unit tests
├── test_url_ranking_api.py     # URL ranking API tests
└── __pycache__/                # Pytest cache
```

## Configuration

Test configuration is in `pyproject.toml`:
- Test discovery patterns
- Warning filters
- Default pytest options

## Best Practices

1. **Run tests frequently** during development
2. **Write tests for new features** before implementation
3. **Keep tests isolated** - each test should run independently
4. **Use descriptive test names** that explain what's being tested
5. **Assert specific conditions** rather than using generic checks
6. **Clean up resources** after tests (databases, files, etc.)

## CI/CD Integration

For GitHub Actions or other CI systems:
```yaml
- name: Run tests
  run: python -m pytest tests/ -v --tb=short --cov=. --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Support

For issues or questions about tests:
1. Review the test file for examples
2. Run with `-v` flag for detailed output
3. Check pytest documentation: https://docs.pytest.org/
