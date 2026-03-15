#!/bin/bash
# =============================================================================
# Intent Engine - Production Startup Script
# =============================================================================
# This script handles the complete startup process for the Intent Engine
# search engine backend with health checks and proper initialization.
#
# Usage:
#   ./scripts/production_start.sh [start|stop|restart|status|logs]
#
# Quick Start:
#   ./scripts/production_start.sh start
#   sleep 60
#   curl http://localhost:8000/search -X POST -H "Content-Type: application/json" -d '{"query":"best laptop for programming"}'
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.prod.yml"
COMPOSE_PROJECT_NAME="intent-engine"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Use docker compose (v2) or docker-compose (v1)
    if docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    log_info "Using Docker Compose: $($COMPOSE_CMD version --short 2>/dev/null || $COMPOSE_CMD version)"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_warning ".env file not found, using default environment variables"
        log_info "Consider copying .env.example to .env and customizing it"
    else
        log_info "Using .env file: $PROJECT_DIR/.env"
    fi
}

# Wait for a service to be healthy
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

# Wait for PostgreSQL to be ready
wait_for_postgres() {
    log_info "Waiting for PostgreSQL to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec intent-engine-postgres pg_isready -U intent_user -d intent_engine &> /dev/null; then
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

# Check overall system health
check_health() {
    log_info "Checking system health..."
    
    local all_healthy=true
    
    # Check API
    if curl -f -s "http://localhost:8000/" &> /dev/null; then
        log_success "✓ API is healthy (port 8000)"
    else
        log_error "✗ API is not responding"
        all_healthy=false
    fi
    
    # Check SearXNG
    if curl -f -s "http://localhost:8080/healthz" &> /dev/null; then
        log_success "✓ SearXNG is healthy (port 8080)"
    else
        log_error "✗ SearXNG is not responding"
        all_healthy=false
    fi
    
    # Check PostgreSQL
    if docker exec intent-engine-postgres pg_isready -U intent_user -d intent_engine &> /dev/null 2>&1; then
        log_success "✓ PostgreSQL is healthy (port 5432)"
    else
        log_error "✗ PostgreSQL is not ready"
        all_healthy=false
    fi
    
    # Check Redis
    if docker exec intent-redis valkey-cli ping &> /dev/null 2>&1; then
        log_success "✓ Redis is healthy (port 6379)"
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
    log_info "Starting Intent Engine search backend..."
    
    cd "$PROJECT_DIR"
    
    # Start services
    log_info "Starting Docker containers..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" up -d
    
    # Wait for PostgreSQL
    wait_for_postgres || {
        log_error "Failed to start PostgreSQL"
        return 1
    }
    
    # Wait for migrations
    log_info "Waiting for database migrations..."
    sleep 10
    
    # Wait for SearXNG
    wait_for_service "SearXNG" "http://localhost:8080/healthz" 30 2 || {
        log_error "Failed to start SearXNG"
        return 1
    }
    
    # Wait for API
    wait_for_service "Intent Engine API" "http://localhost:8000/" 60 3 || {
        log_error "Failed to start Intent Engine API"
        return 1
    }
    
    # Final health check
    sleep 5
    check_health
    
    log_success ""
    log_success "Intent Engine is ready!"
    log_success ""
    log_success "API Endpoint: http://localhost:8000"
    log_success "Search Endpoint: http://localhost:8000/search"
    log_success "SearXNG: http://localhost:8080"
    log_success "PostgreSQL: localhost:5432"
    log_success "Redis: localhost:6379"
    log_success ""
    log_info "Test the search API:"
    log_info '  curl http://localhost:8000/search -X POST -H "Content-Type: application/json" -d '\''{"query":"best laptop for programming"}'\'''
    log_success ""
}

# Stop all services
stop_services() {
    log_info "Stopping Intent Engine..."
    
    cd "$PROJECT_DIR"
    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" down
    
    log_success "Intent Engine stopped"
}

# Restart all services
restart_services() {
    log_info "Restarting Intent Engine..."
    stop_services
    sleep 5
    start_services
}

# Show logs
show_logs() {
    local service=${1:-}
    
    cd "$PROJECT_DIR"
    
    if [ -n "$service" ]; then
        $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" logs -f "$service"
    else
        $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" logs -f
    fi
}

# Show status
show_status() {
    cd "$PROJECT_DIR"
    
    log_info "Service Status:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" ps
    
    echo ""
    check_health
}

# Print usage
print_usage() {
    echo "Intent Engine - Production Startup Script"
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
    echo "  $0 logs intent-engine-api"
    echo "  $0 health"
    echo ""
}

# Main entry point
main() {
    check_docker
    check_env_file
    
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
