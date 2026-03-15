#!/bin/bash

# =============================================================================
# Intent Engine - Complete Setup Verification Script
# =============================================================================
# This script verifies all services are running correctly
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.prod.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

log_pass() { echo -e "${GREEN}✓${NC} $1"; ((PASS++)); }
log_fail() { echo -e "${RED}✗${NC} $1"; ((FAIL++)); }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; ((WARN++)); }
log_info() { echo -e "${BLUE}ℹ${NC} $1"; }

section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}$1${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Check Docker
section "1. Docker Environment"
if command -v docker &> /dev/null; then
    log_pass "Docker installed: $(docker --version)"
else
    log_fail "Docker not installed"
fi

if docker compose version &> /dev/null 2>&1; then
    log_pass "Docker Compose installed: $(docker compose version)"
else
    log_fail "Docker Compose not installed"
fi

# Check containers
section "2. Container Status"
cd "$PROJECT_DIR"

containers=(
    "intent-engine-postgres"
    "intent-redis"
    "intent-engine-api"
    "searxng"
    "intent-go-search-api"
    "intent-go-crawler"
    "intent-go-indexer"
)

for container in "${containers[@]}"; do
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        log_pass "$container is running"
    else
        log_warn "$container is not running"
    fi
done

# Check health endpoints
section "3. Health Checks"

# PostgreSQL
if docker exec intent-engine-postgres pg_isready -U intent_user &> /dev/null; then
    log_pass "PostgreSQL is healthy"
else
    log_fail "PostgreSQL health check failed"
fi

# Redis
if docker exec intent-redis redis-cli ping &> /dev/null; then
    log_pass "Redis is healthy"
else
    log_fail "Redis health check failed"
fi

# Python API
if curl -s -f http://localhost:8000/health &> /dev/null; then
    log_pass "Python Intent API is healthy (Port 8000)"
else
    log_fail "Python Intent API health check failed"
fi

# Go Search API
if curl -s -f http://localhost:8081/health &> /dev/null; then
    log_pass "Go Search API is healthy (Port 8081)"
else
    log_fail "Go Search API health check failed"
fi

# SearXNG
if curl -s -f http://localhost:8080/healthz &> /dev/null; then
    log_pass "SearXNG is healthy (Port 8080)"
else
    log_fail "SearXNG health check failed"
fi

# Test search functionality
section "4. Search Functionality Tests"

# Test Python API search
log_info "Testing Python API search..."
search_result=$(curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"test","limit":1}' 2>/dev/null)

if [ -n "$search_result" ]; then
    log_pass "Python API search working"
else
    log_fail "Python API search failed"
fi

# Test Go Search API
log_info "Testing Go Search API..."
go_search_result=$(curl -s -X POST http://localhost:8081/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"test","limit":1}' 2>/dev/null)

if [ -n "$go_search_result" ]; then
    log_pass "Go Search API search working"
else
    log_fail "Go Search API search failed"
fi

# Check database
section "5. Database Checks"

# Check crawled pages table
page_count=$(docker exec intent-engine-postgres psql -U intent_user -d intent_engine -t -c \
  "SELECT COUNT(*) FROM crawled_pages;" 2>/dev/null | tr -d ' ')

if [ -n "$page_count" ]; then
    log_pass "Crawled pages: $page_count"
else
    log_warn "Could not check crawled pages"
fi

# Check if pages are indexed
indexed_count=$(docker exec intent-engine-postgres psql -U intent_user -d intent_engine -t -c \
  "SELECT COUNT(*) FROM crawled_pages WHERE is_indexed = true;" 2>/dev/null | tr -d ' ')

if [ -n "$indexed_count" ]; then
    log_pass "Indexed pages: $indexed_count"
else
    log_warn "Could not check indexed pages"
fi

# Check Redis queue
queue_size=$(docker exec intent-redis redis-cli ZCARD crawl_queue 2>/dev/null)

if [ -n "$queue_size" ]; then
    log_pass "Redis queue size: $queue_size"
else
    log_warn "Could not check Redis queue"
fi

# Check crawler logs
section "6. Crawler Status"
if docker logs intent-go-crawler 2>&1 | grep -q "Starting crawl"; then
    log_pass "Crawler has started"
else
    log_warn "Could not verify crawler status"
fi

# Summary
section "7. Summary"
echo ""
echo -e "Passed:  ${GREEN}$PASS${NC}"
echo -e "Failed:  ${RED}$FAIL${NC}"
echo -e "Warnings: ${YELLOW}$WARN${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo "Your Intent Engine is ready to use!"
    echo ""
    echo "Access Points:"
    echo "  - Python API:      http://localhost:8000"
    echo "  - API Docs:        http://localhost:8000/docs"
    echo "  - Go Search API:   http://localhost:8081"
    echo "  - SearXNG:         http://localhost:8080"
    echo "  - Grafana:         http://localhost:3000 (admin/admin)"
    echo "  - Prometheus:      http://localhost:9090"
    echo ""
    echo "Quick Test:"
    echo '  curl -X POST http://localhost:8000/search \'
    echo '    -H "Content-Type: application/json" \'
    echo '    -d '\''{"query":"best laptop for programming","limit":10}'\'''
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please review the logs.${NC}"
    echo ""
    echo "View logs:"
    echo "  docker compose -f docker-compose.prod.yml logs -f"
    echo ""
    exit 1
fi
