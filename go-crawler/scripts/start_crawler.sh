#!/bin/bash
# =============================================================================
# Intent Engine - Go Crawler & Indexer Production Startup Script
# =============================================================================
# This script handles the complete startup process for the Go crawler system
# with health checks and proper initialization.
#
# Usage:
#   ./scripts/start_crawler.sh [start|stop|restart|status|logs]
#
# Quick Start:
#   ./scripts/start_crawler.sh start
#   sleep 30
#   curl http://localhost:8081/health
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CRAWLER_DIR="$PROJECT_DIR/go-crawler"
COMPOSE_FILE="$CRAWLER_DIR/docker-compose.yml"
COMPOSE_PROJECT_NAME="intent-crawler"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    log_info "Using Docker Compose: $($COMPOSE_CMD version --short 2>/dev/null || $COMPOSE_CMD version)"
}

# Wait for service to be healthy
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=${3:-30}
    local delay=${4:-2}
    
    log_info "Waiting for $service_name to be ready..."
    
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s -o /dev/null "$url" 2>/dev/null; then
            log_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep $delay
        ((attempt++))
    done
    
    echo ""
    log_error "$service_name failed to start after $max_attempts attempts"
    return 1
}

# Wait for PostgreSQL
wait_for_postgres() {
    log_info "Waiting for PostgreSQL to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec intent-postgres pg_isready -U crawler -d intent_engine &> /dev/null; then
            log_success "PostgreSQL is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo ""
    log_error "PostgreSQL failed to start"
    return 1
}

# Check system health
check_health() {
    log_info "Checking system health..."
    
    local all_healthy=true
    
    # Check Search API
    if curl -f -s "http://localhost:8081/health" &> /dev/null; then
        log_success "✓ Search API is healthy (port 8081)"
    else
        log_error "✗ Search API is not responding"
        all_healthy=false
    fi
    
    # Check SearXNG
    if curl -f -s "http://localhost:8082/healthz" &> /dev/null; then
        log_success "✓ SearXNG is healthy (port 8082)"
    else
        log_error "✗ SearXNG is not responding"
        all_healthy=false
    fi
    
    # Check PostgreSQL
    if docker exec intent-postgres pg_isready -U crawler -d intent_engine &> /dev/null 2>&1; then
        log_success "✓ PostgreSQL is healthy"
    else
        log_error "✗ PostgreSQL is not ready"
        all_healthy=false
    fi
    
    # Check Redis
    if docker exec intent-redis redis-cli ping &> /dev/null 2>&1; then
        log_success "✓ Redis is healthy"
    else
        log_error "✗ Redis is not responding"
        all_healthy=false
    fi
    
    if [ "$all_healthy" = true ]; then
        log_success "All services are healthy!"
        return 0
    else
        log_error "Some services are unhealthy"
        return 1
    fi
}

# Start all services
start_services() {
    log_info "Starting Intent Engine Go Crawler System..."
    
    cd "$CRAWLER_DIR"
    
    # Start services
    log_info "Starting Docker containers..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" up -d
    
    # Wait for PostgreSQL
    wait_for_postgres || {
        log_error "Failed to start PostgreSQL"
        return 1
    }
    
    # Wait for Search API
    wait_for_service "Search API" "http://localhost:8081/health" 60 3 || {
        log_error "Failed to start Search API"
        return 1
    }
    
    # Wait for SearXNG
    wait_for_service "SearXNG" "http://localhost:8082/healthz" 60 3 || {
        log_error "Failed to start SearXNG"
        return 1
    }
    
    # Final health check
    sleep 5
    check_health
    
    log_success ""
    log_success "Intent Engine Go Crawler is ready!"
    log_success ""
    log_success "Services:"
    log_success "  - Search API:    http://localhost:8081"
    log_success "  - SearXNG:       http://localhost:8082"
    log_success "  - PostgreSQL:    localhost:5432"
    log_success "  - Redis:         localhost:6379"
    log_success ""
    log_success "Test the search API:"
    log_success "  curl http://localhost:8081/health"
    log_success "  curl -X POST http://localhost:8081/api/v1/search -H 'Content-Type: application/json' -d '{\"query\":\"golang\"}'"
    log_success ""
    log_success "Monitor crawling:"
    log_success "  docker-compose logs -f crawler"
    log_success "  curl http://localhost:8081/stats"
    log_success ""
}

# Stop all services
stop_services() {
    log_info "Stopping Intent Engine Go Crawler..."
    
    cd "$CRAWLER_DIR"
    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" down
    
    log_success "Intent Engine Go Crawler stopped"
}

# Restart all services
restart_services() {
    log_info "Restarting Intent Engine Go Crawler..."
    stop_services
    sleep 5
    start_services
}

# Show logs
show_logs() {
    local service=${1:-}
    
    cd "$CRAWLER_DIR"
    
    if [ -n "$service" ]; then
        $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" logs -f "$service"
    else
        $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" logs -f
    fi
}

# Show status
show_status() {
    cd "$CRAWLER_DIR"
    
    log_info "Service Status:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" ps
    
    echo ""
    check_health
}

# Print usage
print_usage() {
    echo "Intent Engine Go Crawler - Startup Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        Show logs (optionally specify service name)"
    echo "  health      Check system health"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs crawler"
    echo "  $0 health"
    echo ""
}

# Main entry point
main() {
    check_docker
    
    local command=${1:-start}
    
    case "$command" in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "$2"
            ;;
        health)
            check_health
            ;;
        *)
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
