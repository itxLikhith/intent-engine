#!/bin/bash
# =============================================================================
# Intent Engine - Search API Examples
# =============================================================================
# This script contains example curl commands for testing the search API.
#
# Usage:
#   ./scripts/api_examples.sh [endpoint_name]
#
# Examples:
#   ./scripts/api_examples.sh health       # Health checks
#   ./scripts/api_examples.sh search       # Search endpoints
#   ./scripts/api_examples.sh intent       # Intent extraction
#   ./scripts/api_examples.sh all          # Run all examples
# =============================================================================

BASE_URL="${API_BASE_URL:-http://localhost:8000}"
SEARXNG_URL="${SEARXNG_URL:-http://localhost:8080}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_example() {
    echo -e "${YELLOW}Example: $1${NC}"
}

print_response() {
    echo -e "${GREEN}Response:${NC}"
    echo "$1" | python3 -m json.tool 2>/dev/null || echo "$1"
    echo ""
}

# =============================================================================
# Health Checks
# =============================================================================
test_health() {
    print_header "Health Checks"

    print_example "Root health check"
    curl -s "$BASE_URL/" | python3 -m json.tool
    echo ""

    print_example "Detailed health check"
    curl -s "$BASE_URL/health" | python3 -m json.tool
    echo ""

    print_example "SearXNG health"
    curl -s "$SEARXNG_URL/healthz" | python3 -m json.tool
    echo ""

    print_example "Metrics endpoint"
    curl -s "$BASE_URL/metrics" | head -20
    echo ""
}

# =============================================================================
# Search Endpoints
# =============================================================================
test_search() {
    print_header "Search Endpoints"

    print_example "Basic search"
    curl -s -X POST "$BASE_URL/search" \
        -H "Content-Type: application/json" \
        -d '{"query": "weather today"}' | python3 -m json.tool
    echo ""

    print_example "Search with intent extraction"
    curl -s -X POST "$BASE_URL/search" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "best laptop for programming under $1000",
            "extract_intent": true,
            "rank_results": true
        }' | python3 -m json.tool
    echo ""

    print_example "Search with privacy filters"
    curl -s -X POST "$BASE_URL/search" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "encrypted messaging apps",
            "exclude_big_tech": true,
            "min_privacy_score": 0.7
        }' | python3 -m json.tool
    echo ""

    print_example "Search in specific categories"
    curl -s -X POST "$BASE_URL/search" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "python tutorial",
            "categories": ["general", "it"],
            "language": "en"
        }' | python3 -m json.tool
    echo ""

    print_example "Search with time range"
    curl -s -X POST "$BASE_URL/search" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "AI developments",
            "time_range": "year",
            "sort_by": "date"
        }' | python3 -m json.tool
    echo ""

    print_example "Complex query with constraints"
    curl -s -X POST "$BASE_URL/search" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "best smartphone camera quality budget 500",
            "extract_intent": true,
            "rank_results": true,
            "weights": {
                "relevance": 0.4,
                "privacy": 0.3,
                "recency": 0.3
            }
        }' | python3 -m json.tool
    echo ""
}

# =============================================================================
# Intent Extraction
# =============================================================================
test_intent() {
    print_header "Intent Extraction Endpoints"

    print_example "Extract intent - shopping query"
    curl -s -X POST "$BASE_URL/extract-intent" \
        -H "Content-Type: application/json" \
        -d '{
            "product": "search",
            "input": {"text": "best laptop for programming under 50000 rupees"},
            "context": {"sessionId": "test-123", "userLocale": "en-US"}
        }' | python3 -m json.tool
    echo ""

    print_example "Extract intent - troubleshooting query"
    curl -s -X POST "$BASE_URL/extract-intent" \
        -H "Content-Type: application/json" \
        -d '{
            "product": "search",
            "input": {"text": "how to fix blue screen error Windows 11"},
            "context": {"sessionId": "test-124", "userLocale": "en-US"}
        }' | python3 -m json.tool
    echo ""

    print_example "Extract intent - comparison query"
    curl -s -X POST "$BASE_URL/extract-intent" \
        -H "Content-Type: application/json" \
        -d '{
            "product": "search",
            "input": {"text": "iPhone 15 vs Samsung S24 comparison"},
            "context": {"sessionId": "test-125", "userLocale": "en-US"}
        }' | python3 -m json.tool
    echo ""

    print_example "Extract intent - learning query"
    curl -s -X POST "$BASE_URL/extract-intent" \
        -H "Content-Type: application/json" \
        -d '{
            "product": "search",
            "input": {"text": "learn python programming for beginners"},
            "context": {"sessionId": "test-126", "userLocale": "en-US"}
        }' | python3 -m json.tool
    echo ""

    print_example "Extract intent - privacy-focused query"
    curl -s -X POST "$BASE_URL/extract-intent" \
        -H "Content-Type: application/json" \
        -d '{
            "product": "search",
            "input": {"text": "privacy-focused email service no tracking"},
            "context": {"sessionId": "test-127", "userLocale": "en-US"}
        }' | python3 -m json.tool
    echo ""
}

# =============================================================================
# URL Ranking
# =============================================================================
test_ranking() {
    print_header "URL Ranking Endpoints"

    print_example "Rank URLs"
    curl -s -X POST "$BASE_URL/rank-urls" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "best programming laptops",
            "urls": [
                "https://www.laptopmag.com/articles/best-programming-laptops",
                "https://www.pcmag.com/picks/the-best-laptops-for-programming",
                "https://www.techradar.com/news/best-laptops-for-coding"
            ],
            "options": {
                "weights": {
                    "relevance": 0.5,
                    "privacy": 0.3,
                    "quality": 0.2
                }
            }
        }' | python3 -m json.tool
    echo ""
}

# =============================================================================
# Service Recommendation
# =============================================================================
test_services() {
    print_header "Service Recommendation Endpoints"

    print_example "Recommend services"
    curl -s -X POST "$BASE_URL/recommend-services" \
        -H "Content-Type: application/json" \
        -d '{
            "product": "search",
            "input": {"text": "need help setting up encrypted email"},
            "context": {"sessionId": "test-128", "userLocale": "en-US"}
        }' | python3 -m json.tool
    echo ""
}

# =============================================================================
# Ad Matching (Optional)
# =============================================================================
test_ads() {
    print_header "Ad Matching Endpoints"

    print_example "Match ads"
    curl -s -X POST "$BASE_URL/match-ads" \
        -H "Content-Type: application/json" \
        -d '{
            "product": "search",
            "input": {"text": "best budget smartphones"},
            "context": {"sessionId": "test-129", "userLocale": "en-US"},
            "options": {
                "limit": 3
            }
        }' | python3 -m json.tool
    echo ""
}

# =============================================================================
# Main
# =============================================================================
check_services() {
    print_header "Checking Service Availability"
    
    echo -n "Checking API at $BASE_URL... "
    if curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/" | grep -q "200"; then
        echo -e "${GREEN}✓ Available${NC}"
    else
        echo -e "${YELLOW}✗ Not available${NC}"
        echo "Please start the services first:"
        echo "  ./scripts/production_start.sh start"
        exit 1
    fi

    echo -n "Checking SearXNG at $SEARXNG_URL... "
    if curl -s -o /dev/null -w "%{http_code}" "$SEARXNG_URL/healthz" | grep -q "200"; then
        echo -e "${GREEN}✓ Available${NC}"
    else
        echo -e "${YELLOW}✗ Not available${NC}"
    fi
    echo ""
}

print_usage() {
    echo "Intent Engine - Search API Examples"
    echo ""
    echo "Usage: $0 [endpoint]"
    echo ""
    echo "Endpoints:"
    echo "  health    - Health checks and metrics"
    echo "  search    - Search endpoints"
    echo "  intent    - Intent extraction"
    echo "  ranking   - URL ranking"
    echo "  services  - Service recommendation"
    echo "  ads       - Ad matching"
    echo "  all       - Run all examples"
    echo ""
    echo "Examples:"
    echo "  $0 health"
    echo "  $0 search"
    echo "  $0 all"
    echo ""
}

main() {
    local command=${1:-all}

    check_services

    case "$command" in
        health)
            test_health
            ;;
        search)
            test_search
            ;;
        intent)
            test_intent
            ;;
        ranking)
            test_ranking
            ;;
        services)
            test_services
            ;;
        ads)
            test_ads
            ;;
        all)
            test_health
            test_search
            test_intent
            ;;
        *)
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
