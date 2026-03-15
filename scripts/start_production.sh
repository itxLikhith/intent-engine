#!/bin/bash

# =============================================================================
# Intent Engine - Production Startup Script (Linux/Mac)
# =============================================================================
# Usage:
#   ./scripts/start_production.sh start       - Start all services
#   ./scripts/start_production.sh stop        - Stop all services
#   ./scripts/start_production.sh restart     - Restart all services
#   ./scripts/start_production.sh status      - Check service status
#   ./scripts/start_production.sh logs        - View logs
#   ./scripts/start_production.sh rebuild     - Rebuild and restart
#   ./scripts/start_production.sh clean       - Stop and remove volumes
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.prod.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose v2+."
        exit 1
    fi
    
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose v2+ is required."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

check_env_file() {
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_warning ".env file not found. Creating from template..."
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        log_info "Please edit $PROJECT_DIR/.env with your configuration"
    fi
}

start_services() {
    log_info "Starting Intent Engine services..."
    
    cd "$PROJECT_DIR"
    
    # Check if .env exists
    check_env_file
    
    # Start services
    docker compose -f "$COMPOSE_FILE" up -d
    
    log_info "Waiting for services to start (60 seconds)..."
    sleep 60
    
    # Check health
    check_health
    
    log_success "Intent Engine started successfully!"
    show_status
}

stop_services() {
    log_info "Stopping Intent Engine services..."
    
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" down
    
    log_success "Intent Engine stopped"
}

restart_services() {
    log_info "Restarting Intent Engine services..."
    
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" restart
    
    log_success "Intent Engine restarted"
}

show_status() {
    log_info "Service Status:"
    echo ""
    
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" ps
    
    echo ""
    log_info "Health Checks:"
    
    # Check Python API
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Python API (Port 8000)"
    else
        echo -e "  ${RED}✗${NC} Python API (Port 8000)"
    fi
    
    # Check Go Search API
    if curl -s -f http://localhost:8081/health > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Go Search API (Port 8081)"
    else
        echo -e "  ${RED}✗${NC} Go Search API (Port 8081)"
    fi
    
    # Check SearXNG
    if curl -s -f http://localhost:8080/healthz > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} SearXNG (Port 8080)"
    else
        echo -e "  ${RED}✗${NC} SearXNG (Port 8080)"
    fi
    
    echo ""
    log_info "Access Points:"
    echo "  - Python API:      http://localhost:8000"
    echo "  - API Docs:        http://localhost:8000/docs"
    echo "  - Go Search API:   http://localhost:8081"
    echo "  - SearXNG:         http://localhost:8080"
    echo "  - Grafana:         http://localhost:3000 (admin/admin)"
    echo "  - Prometheus:      http://localhost:9090"
}

view_logs() {
    cd "$PROJECT_DIR"
    
    if [ -n "$1" ]; then
        log_info "Showing logs for: $1"
        docker compose -f "$COMPOSE_FILE" logs -f "$1"
    else
        log_info "Showing all logs..."
        docker compose -f "$COMPOSE_FILE" logs -f
    fi
}

rebuild_services() {
    log_info "Rebuilding Intent Engine services..."
    
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" down
    docker compose -f "$COMPOSE_FILE" build --no-cache
    docker compose -f "$COMPOSE_FILE" up -d
    
    log_info "Waiting for services to start (60 seconds)..."
    sleep 60
    
    check_health
    
    log_success "Intent Engine rebuilt and started"
}

clean_all() {
    log_warning "This will remove all data volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Cleaning all data..."
        
        cd "$PROJECT_DIR"
        docker compose -f "$COMPOSE_FILE" down -v
        docker volume prune -f
        
        log_success "All data cleaned"
    else
        log_info "Clean operation cancelled"
    fi
}

check_health() {
    log_info "Checking service health..."
    
    # Wait for PostgreSQL
    log_info "Waiting for PostgreSQL..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker exec intent-engine-postgres pg_isready -U intent_user > /dev/null 2>&1; then
            log_success "PostgreSQL is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -eq 0 ]; then
        log_error "PostgreSQL failed to start"
        return 1
    fi
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if docker exec intent-redis redis-cli ping > /dev/null 2>&1; then
            log_success "Redis is ready"
            break
        fi
        sleep 1
        timeout=$((timeout - 1))
    done
    
    if [ $timeout -eq 0 ]; then
        log_error "Redis failed to start"
        return 1
    fi
}

show_help() {
    echo "Intent Engine - Production Startup Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        View logs (optionally specify service name)"
    echo "  rebuild     Rebuild and restart all services"
    echo "  clean       Stop and remove all data volumes"
    echo "  health      Check service health"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs intent-engine-api"
    echo "  $0 status"
    echo ""
}

# Main command handler
case "${1:-help}" in
    start)
        check_prerequisites
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
        view_logs "$2"
        ;;
    rebuild)
        rebuild_services
        ;;
    clean)
        clean_all
        ;;
    health)
        check_health
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
