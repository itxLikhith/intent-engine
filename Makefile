# Intent Engine - Development Makefile
# Common development tasks simplified

.PHONY: help install dev test lint format clean docker-build docker-run migrations

# Default target
help:
	@echo "Intent Engine - Available Commands"
	@echo "=================================="
	@echo ""
	@echo "Development:"
	@echo "  make install        - Install production dependencies"
	@echo "  make dev            - Install development dependencies"
	@echo "  make pre-commit     - Install pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make test-fast      - Run tests in parallel"
	@echo "  make test-watch     - Run tests on file changes"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code"
	@echo "  make check          - Run all checks"
	@echo "  make security       - Run security scans"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Start all services"
	@echo "  make docker-stop    - Stop all services"
	@echo "  make docker-logs    - View logs"
	@echo "  make docker-clean   - Remove containers and volumes"
	@echo ""
	@echo "Database:"
	@echo "  make migrations     - Run database migrations"
	@echo "  make seed           - Seed database with sample data"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           - Build documentation"
	@echo "  make docs-serve     - Serve documentation locally"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make distclean      - Remove everything including venv"

# ============================================================================
# Installation
# ============================================================================

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev: install
	pip install -r requirements-dev.txt
	pre-commit install

pre-commit:
	pre-commit install

# ============================================================================
# Testing
# ============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term-missing tests/

test-fast:
	pytest -n auto --cov=. tests/

test-watch:
	ptw --now . -- -v tests/

# ============================================================================
# Code Quality
# ============================================================================

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

check: lint test

security:
	bandit -r . -f json -o bandit-report.json || true
	safety check -r requirements.txt --output json || true

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 30
	@echo "Checking API health..."
	@curl -f http://localhost:8000/ || echo "API not ready yet"

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker system prune -f

# ============================================================================
# Database
# ============================================================================

migrations:
	@for migration in migrations/*.sql; do \
		if [ -f "$$migration" ]; then \
			echo "Running $$migration"; \
			docker exec intent-engine-postgres psql -U intent_user -d intent_engine -f "/tmp/$$(basename $$migration)" || \
			psql -U intent_user -d intent_engine -f "$$migration"; \
		fi \
	done

seed:
	python seed_sample_data.py

# ============================================================================
# Documentation
# ============================================================================

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# ============================================================================
# Cleanup
# ============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ htmlcov/ .mypy_cache/
	rm -rf coverage.xml .coverage

distclean: clean
	rm -rf venv/ .venv/ ENV/
