#!/bin/bash
# =============================================================================
# Intent Engine - Setup Verification Script
# =============================================================================
# This script verifies that all components are properly set up and working.
#
# Usage:
#   ./scripts/verify_setup.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

log_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASS++))
}

log_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAIL++))
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARN++))
}

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# =============================================================================
# Prerequisites Check
# =============================================================================
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Docker
    if command -v docker &> /dev/null; then
        docker_version=$(docker --version)
        log_pass "Docker installed: $docker_version"
    else
        log_fail "Docker not installed"
    fi
    
    # Docker Compose
    if docker compose version &> /dev/null 2>&1; then
        compose_version=$(docker compose version --short)
        log_pass "Docker Compose installed: $compose_version"
    elif command -v docker-compose &> /dev/null; then
        compose_version=$(docker-compose version --short)
        log_pass "Docker Compose installed: $compose_version"
    else
        log_fail "Docker Compose not installed"
    fi
    
    # curl
    if command -v curl &> /dev/null; then
        log_pass "curl installed"
    else
        log_warn "curl not installed (needed for API testing)"
    fi
    
    # Python (optional)
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version)
        log_pass "Python installed: $python_version"
    elif command -v python &> /dev/null; then
        python_version=$(python --version)
        log_pass "Python installed: $python_version"
    else
        log_warn "Python not installed (optional, for scripts)"
    fi
}

# =============================================================================
# File Structure Check
# =============================================================================
check_files() {
    print_header "Checking File Structure"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    
    # Required files
    required_files=(
        "docker-compose.prod.yml"
        ".env"
        "Dockerfile"
        "requirements.txt"
        "main_api.py"
        "README_PRODUCTION.md"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "$PROJECT_DIR/$file" ]; then
            log_pass "Found: $file"
        else
            log_fail "Missing: $file"
        fi
    done
    
    # Required directories
    required_dirs=(
        "scripts"
        "searxng"
        "core"
        "extraction"
        "ranking"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ -d "$PROJECT_DIR/$dir" ]; then
            log_pass "Found directory: $dir"
        else
            log_warn "Missing directory: $dir"
        fi
    done
    
    # Check .env configuration
    if [ -f "$PROJECT_DIR/.env" ]; then
        if grep -q "SECRET_KEY=change-this-to-a-secure-random-string-in-production" "$PROJECT_DIR/.env"; then
            log_warn "SECRET_KEY not changed from default (OK for testing, change for production)"
        else
            log_pass "SECRET_KEY configured"
        fi
        
        if grep -q "intent_secure_password_change_in_prod" "$PROJECT_DIR/.env"; then
            log_warn "Database password not changed from default (OK for testing, change for production)"
        else
            log_pass "Database password configured"
        fi
    else
        log_warn ".env file not found (will use defaults)"
    fi
}

# =============================================================================
# Docker Configuration Check
# =============================================================================
check_docker_config() {
    print_header "Checking Docker Configuration"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    
    # Validate docker-compose file
    if [ -f "$PROJECT_DIR/docker-compose.prod.yml" ]; then
        cd "$PROJECT_DIR"
        if docker-compose -f docker-compose.prod.yml config --quiet &> /dev/null; then
            log_pass "docker-compose.prod.yml is valid"
        else
            log_fail "docker-compose.prod.yml has errors"
        fi
    fi
    
    # Check Docker resources
    docker_info=$(docker info 2>/dev/null)
    if [ -n "$docker_info" ]; then
        total_memory=$(echo "$docker_info" | grep "Total Memory" | awk '{print $3, $4}')
        log_info "Available memory: $total_memory"
        
        # Check if enough memory (at least 2GB recommended)
        memory_bytes=$(echo "$docker_info" | grep "Total Memory" | awk '{print $3}')
        if [ -n "$memory_bytes" ] && [ "$memory_bytes" -lt 2000000000 ]; then
            log_warn "Less than 2GB memory available (may experience performance issues)"
        else
            log_pass "Sufficient memory available"
        fi
    fi
    
    # Check disk space
    disk_space=$(df -h . | awk 'NR==2 {print $4}')
    log_info "Available disk space: $disk_space"
}

# =============================================================================
# Service Connectivity Check (if running)
# =============================================================================
check_services() {
    print_header "Checking Running Services"
    
    # Check if services are running
    if docker ps | grep -q intent-engine-api; then
        log_pass "Intent Engine API is running"
        
        # Test API health
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ | grep -q "200"; then
            log_pass "API health check passed"
        else
            log_fail "API health check failed"
        fi
    else
        log_warn "Intent Engine API is not running"
        log_info "Start with: ./scripts/production_start.sh start"
    fi
    
    if docker ps | grep -q searxng; then
        log_pass "SearXNG is running"
        
        # Test SearXNG health
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/healthz | grep -q "200"; then
            log_pass "SearXNG health check passed"
        else
            log_fail "SearXNG health check failed"
        fi
    else
        log_warn "SearXNG is not running"
    fi
    
    if docker ps | grep -q intent-engine-postgres; then
        log_pass "PostgreSQL is running"
        
        # Test PostgreSQL health
        if docker exec intent-engine-postgres pg_isready -U intent_user -d intent_engine &> /dev/null; then
            log_pass "PostgreSQL health check passed"
        else
            log_fail "PostgreSQL health check failed"
        fi
    else
        log_warn "PostgreSQL is not running"
    fi
    
    if docker ps | grep -q intent-redis; then
        log_pass "Redis is running"
        
        # Test Redis health
        if docker exec intent-redis valkey-cli ping &> /dev/null 2>&1; then
            log_pass "Redis health check passed"
        else
            log_fail "Redis health check failed"
        fi
    else
        log_warn "Redis is not running"
    fi
}

# =============================================================================
# Summary
# =============================================================================
print_summary() {
    print_header "Verification Summary"
    
    echo -e "  ${GREEN}Passed:${NC}   $PASS"
    echo -e "  ${RED}Failed:${NC}   $FAIL"
    echo -e "  ${YELLOW}Warnings:${NC} $WARN"
    echo ""
    
    if [ $FAIL -eq 0 ]; then
        echo -e "${GREEN}✓ Setup verification passed!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Start services: ./scripts/production_start.sh start"
        echo "  2. Wait for initialization: sleep 60"
        echo "  3. Test search: curl http://localhost:8000/search -X POST -H 'Content-Type: application/json' -d '{\"query\":\"test\"}'"
        echo ""
        exit 0
    else
        echo -e "${RED}✗ Setup verification failed!${NC}"
        echo ""
        echo "Please fix the failed checks above."
        echo "See QUICKSTART.md for detailed setup instructions."
        echo ""
        exit 1
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo -e "${BLUE}Intent Engine - Setup Verification${NC}"
    echo ""
    
    check_prerequisites
    check_files
    check_docker_config
    check_services
    print_summary
}

main "$@"
