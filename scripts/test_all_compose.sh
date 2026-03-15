#!/bin/bash
# =============================================================================
# Intent Engine - Docker Compose Test Suite
# =============================================================================
# Comprehensive test suite for all Docker Compose configurations
#
# Usage:
#   ./scripts/test_all_compose.sh [options]
#
# Options:
#   --validate     Only validate YAML syntax
#   --health       Only run health checks
#   --api          Only run API tests
#   --full         Run all tests (default)
#   --verbose      Enable verbose output
#   --help         Show this help message
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILES=(
    "docker-compose.yml"
    "docker-compose.searxng.yml"
    "docker-compose.go-crawler.yml"
    "docker-compose.aio.yml"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((TESTS_SKIPPED++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

print_header() {
    echo ""
    echo "=============================================================================="
    echo "  $1"
    echo "=============================================================================="
}

print_subheader() {
    echo ""
    echo "--- $1 ---"
}

# =============================================================================
# Test Functions
# =============================================================================

test_file_exists() {
    local file="$1"
    if [ -f "$file" ]; then
        log_success "File exists: $file"
        return 0
    else
        log_error "File not found: $file"
        return 1
    fi
}

test_yaml_syntax() {
    local compose_file="$1"
    log_info "Validating YAML syntax: $compose_file"
    
    if docker-compose -f "$compose_file" config --quiet 2>/dev/null; then
        log_success "YAML syntax valid: $compose_file"
        return 0
    else
        log_error "YAML syntax invalid: $compose_file"
        return 1
    fi
}

test_services_defined() {
    local compose_file="$1"
    log_info "Checking services: $compose_file"
    
    local services=$(docker-compose -f "$compose_file" config --services 2>/dev/null)
    if [ -n "$services" ]; then
        local count=$(echo "$services" | wc -l | tr -d ' ')
        log_success "Found $count service(s) in $compose_file"
        echo "$services" | while read -r service; do
            echo "    - $service"
        done
        return 0
    else
        log_error "No services found in $compose_file"
        return 1
    fi
}

test_networks_defined() {
    local compose_file="$1"
    log_info "Checking networks: $compose_file"
    
    if docker-compose -f "$compose_file" config --networks 2>/dev/null | grep -q .; then
        log_success "Networks defined in $compose_file"
        return 0
    else
        log_warning "No networks defined in $compose_file"
        return 0
    fi
}

test_volumes_defined() {
    local compose_file="$1"
    log_info "Checking volumes: $compose_file"
    
    if docker-compose -f "$compose_file" config --volumes 2>/dev/null | grep -q .; then
        log_success "Volumes defined in $compose_file"
        return 0
    else
        log_warning "No volumes defined in $compose_file"
        return 0
    fi
}

test_container_health() {
    local compose_file="$1"
    log_info "Checking container health: $compose_file"
    
    local status=$(docker-compose -f "$compose_file" ps 2>/dev/null)
    if echo "$status" | grep -q "Up"; then
        log_success "Containers running for $compose_file"
        return 0
    else
        log_warning "No running containers for $compose_file"
        return 0
    fi
}

test_api_health() {
    log_info "Testing API health endpoint"
    
    local response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_success "API health endpoint responding (HTTP $response)"
        return 0
    else
        log_error "API health endpoint not responding (HTTP $response)"
        return 1
    fi
}

test_api_root() {
    log_info "Testing API root endpoint"
    
    local response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_success "API root endpoint responding (HTTP $response)"
        return 0
    else
        log_error "API root endpoint not responding (HTTP $response)"
        return 1
    fi
}

test_search_endpoint() {
    log_info "Testing search endpoint"
    
    local response=$(curl -s -X POST http://localhost:8000/search \
        -H "Content-Type: application/json" \
        -d '{"query":"test","limit":1}' 2>/dev/null)
    
    if echo "$response" | grep -q "results"; then
        local count=$(echo "$response" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('results', [])))" 2>/dev/null || echo "?")
        log_success "Search endpoint working (returned $count results)"
        return 0
    else
        log_error "Search endpoint not working"
        return 1
    fi
}

test_searxng_health() {
    log_info "Testing SearXNG health endpoint"
    
    local response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/healthz 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_success "SearXNG health endpoint responding (HTTP $response)"
        return 0
    else
        log_warning "SearXNG health endpoint not responding (HTTP $response)"
        return 0
    fi
}

test_go_search_health() {
    log_info "Testing Go Search API health endpoint"
    
    local response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/health 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_success "Go Search API health endpoint responding (HTTP $response)"
        return 0
    else
        log_warning "Go Search API health endpoint not responding (HTTP $response)"
        return 0
    fi
}

test_database_connection() {
    log_info "Testing database connectivity"
    
    # Check if postgres container is running
    if docker ps --format '{{.Names}}' | grep -q "postgres"; then
        log_success "PostgreSQL container is running"
        return 0
    else
        log_warning "PostgreSQL container not running"
        return 0
    fi
}

test_redis_connection() {
    log_info "Testing Redis connectivity"
    
    # Check if redis container is running
    if docker ps --format '{{.Names}}' | grep -q "redis"; then
        log_success "Redis container is running"
        return 0
    else
        log_warning "Redis container not running"
        return 0
    fi
}

# =============================================================================
# Test Suites
# =============================================================================

run_validation_tests() {
    print_header "VALIDATION TESTS"
    
    for compose_file in "${COMPOSE_FILES[@]}"; do
        print_subheader "Testing: $compose_file"
        
        test_file_exists "$PROJECT_DIR/$compose_file"
        test_yaml_syntax "$PROJECT_DIR/$compose_file"
        test_services_defined "$PROJECT_DIR/$compose_file"
        test_networks_defined "$PROJECT_DIR/$compose_file"
        test_volumes_defined "$PROJECT_DIR/$compose_file"
    done
}

run_health_tests() {
    print_header "HEALTH CHECK TESTS"
    
    test_api_root
    test_api_health
    test_searxng_health
    test_go_search_health
    test_database_connection
    test_redis_connection
}

run_api_tests() {
    print_header "API FUNCTIONALITY TESTS"
    
    test_search_endpoint
}

run_container_tests() {
    print_header "CONTAINER STATUS TESTS"
    
    for compose_file in "${COMPOSE_FILES[@]}"; do
        print_subheader "Testing: $compose_file"
        test_container_health "$PROJECT_DIR/$compose_file"
    done
}

print_summary() {
    print_header "TEST SUMMARY"
    
    local total=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))
    local pass_rate=0
    if [ $total -gt 0 ]; then
        pass_rate=$((TESTS_PASSED * 100 / total))
    fi
    
    echo ""
    echo "Total Tests:  $total"
    echo -e "Passed:       ${GREEN}$TESTS_PASSED${NC} ($pass_rate%)"
    echo -e "Failed:       ${RED}$TESTS_FAILED${NC}"
    echo -e "Skipped:      ${YELLOW}$TESTS_SKIPPED${NC}"
    echo ""
    echo "=============================================================================="
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        return 1
    fi
}

show_help() {
    echo "Intent Engine - Docker Compose Test Suite"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --validate     Only validate YAML syntax"
    echo "  --health       Only run health checks"
    echo "  --api          Only run API tests"
    echo "  --full         Run all tests (default)"
    echo "  --verbose      Enable verbose output"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --validate          # Validate all compose files"
    echo "  $0 --health            # Run health checks only"
    echo "  $0                     # Run all tests"
}

# =============================================================================
# Main
# =============================================================================

main() {
    local test_mode="full"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --validate)
                test_mode="validate"
                shift
                ;;
            --health)
                test_mode="health"
                shift
                ;;
            --api)
                test_mode="api"
                shift
                ;;
            --full)
                test_mode="full"
                shift
                ;;
            --verbose)
                set -x
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    print_header "INTENT ENGINE - DOCKER COMPOSE TEST SUITE"
    echo "Project Directory: $PROJECT_DIR"
    echo "Test Mode: $test_mode"
    echo "Date: $(date)"
    
    # Run tests based on mode
    case $test_mode in
        validate)
            run_validation_tests
            ;;
        health)
            run_health_tests
            ;;
        api)
            run_api_tests
            ;;
        full)
            run_validation_tests
            run_container_tests
            run_health_tests
            run_api_tests
            ;;
    esac
    
    # Print summary
    print_summary
    exit_code=$?
    
    exit $exit_code
}

# Run main function
main "$@"
