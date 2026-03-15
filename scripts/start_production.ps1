# =============================================================================
# Intent Engine - Production Startup Script (Windows PowerShell)
# =============================================================================
# Usage:
#   .\scripts\start_production.ps1 start       - Start all services
#   .\scripts\start_production.ps1 stop        - Stop all services
#   .\scripts\start_production.ps1 restart     - Restart all services
#   .\scripts\start_production.ps1 status      - Check service status
#   .\scripts\start_production.ps1 logs        - View logs
#   .\scripts\start_production.ps1 rebuild     - Rebuild and restart
#   .\scripts\start_production.ps1 clean       - Stop and remove volumes
# =============================================================================

$ErrorActionPreference = "Stop"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$ComposeFile = Join-Path $ProjectDir "docker-compose.prod.yml"

# Colors
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check prerequisites
function Check-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    try {
        $dockerVersion = docker --version
        Write-Info "Docker found: $dockerVersion"
    } catch {
        Write-Error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    }
    
    try {
        $composeVersion = docker compose version
        Write-Info "Docker Compose found: $composeVersion"
    } catch {
        Write-Error "Docker Compose v2+ is not installed. Please update Docker Desktop."
        exit 1
    }
    
    Write-Success "Prerequisites check passed"
}

# Check environment file
function Check-EnvFile {
    $envFile = Join-Path $ProjectDir ".env"
    $envExample = Join-Path $ProjectDir ".env.example"
    
    if (-not (Test-Path $envFile)) {
        Write-Warning ".env file not found. Creating from template..."
        Copy-Item $envExample $envFile
        Write-Info "Please edit $envFile with your configuration"
    }
}

# Start services
function Start-Services {
    Write-Info "Starting Intent Engine services..."
    
    Check-EnvFile
    
    Set-Location $ProjectDir
    
    # Start services
    docker compose -f $ComposeFile up -d
    
    Write-Info "Waiting for services to start (60 seconds)..."
    Start-Sleep -Seconds 60
    
    # Check health
    Check-Health
    
    Write-Success "Intent Engine started successfully!"
    Show-Status
}

# Stop services
function Stop-Services {
    Write-Info "Stopping Intent Engine services..."
    
    Set-Location $ProjectDir
    docker compose -f $ComposeFile down
    
    Write-Success "Intent Engine stopped"
}

# Restart services
function Restart-Services {
    Write-Info "Restarting Intent Engine services..."
    
    Set-Location $ProjectDir
    docker compose -f $ComposeFile restart
    
    Write-Success "Intent Engine restarted"
}

# Show status
function Show-Status {
    Write-Info "Service Status:"
    Write-Host ""
    
    Set-Location $ProjectDir
    docker compose -f $ComposeFile ps
    
    Write-Host ""
    Write-Info "Health Checks:"
    
    # Check Python API
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -UseBasicParsing
        Write-Host "  [OK] Python API (Port 8000)" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] Python API (Port 8000)" -ForegroundColor Red
    }
    
    # Check Go Search API
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8081/health" -TimeoutSec 5 -UseBasicParsing
        Write-Host "  [OK] Go Search API (Port 8081)" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] Go Search API (Port 8081)" -ForegroundColor Red
    }
    
    # Check SearXNG
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/healthz" -TimeoutSec 5 -UseBasicParsing
        Write-Host "  [OK] SearXNG (Port 8080)" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] SearXNG (Port 8080)" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Info "Access Points:"
    Write-Host "  - Python API:      http://localhost:8000"
    Write-Host "  - API Docs:        http://localhost:8000/docs"
    Write-Host "  - Go Search API:   http://localhost:8081"
    Write-Host "  - SearXNG:         http://localhost:8080"
    Write-Host "  - Grafana:         http://localhost:3000 (admin/admin)"
    Write-Host "  - Prometheus:      http://localhost:9090"
}

# View logs
function View-Logs {
    param([string]$ServiceName)
    
    Set-Location $ProjectDir
    
    if ($ServiceName) {
        Write-Info "Showing logs for: $ServiceName"
        docker compose -f $ComposeFile logs -f $ServiceName
    } else {
        Write-Info "Showing all logs..."
        docker compose -f $ComposeFile logs -f
    }
}

# Rebuild services
function Rebuild-Services {
    Write-Info "Rebuilding Intent Engine services..."
    
    Set-Location $ProjectDir
    docker compose -f $ComposeFile down
    docker compose -f $ComposeFile build --no-cache
    docker compose -f $ComposeFile up -d
    
    Write-Info "Waiting for services to start (60 seconds)..."
    Start-Sleep -Seconds 60
    
    Check-Health
    
    Write-Success "Intent Engine rebuilt and started"
}

# Clean all
function Clean-All {
    $response = Read-Host "This will remove all data volumes. Are you sure? (y/N)"
    
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Info "Cleaning all data..."
        
        Set-Location $ProjectDir
        docker compose -f $ComposeFile down -v
        docker volume prune -f -Force
        
        Write-Success "All data cleaned"
    } else {
        Write-Info "Clean operation cancelled"
    }
}

# Check health
function Check-Health {
    Write-Info "Checking service health..."
    
    # Wait for PostgreSQL
    Write-Info "Waiting for PostgreSQL..."
    $timeout = 60
    while ($timeout -gt 0) {
        try {
            $result = docker exec intent-engine-postgres pg_isready -U intent_user 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Success "PostgreSQL is ready"
                break
            }
        } catch {
            # Continue waiting
        }
        Start-Sleep -Seconds 2
        $timeout -= 2
    }
    
    if ($timeout -eq 0) {
        Write-Error "PostgreSQL failed to start"
        return 1
    }
    
    # Wait for Redis
    Write-Info "Waiting for Redis..."
    $timeout = 30
    while ($timeout -gt 0) {
        try {
            $result = docker exec intent-redis redis-cli ping 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Redis is ready"
                break
            }
        } catch {
            # Continue waiting
        }
        Start-Sleep -Seconds 1
        $timeout -= 1
    }
    
    if ($timeout -eq 0) {
        Write-Error "Redis failed to start"
        return 1
    }
    
    return 0
}

# Show help
function Show-Help {
    Write-Host "Intent Engine - Production Startup Script"
    Write-Host ""
    Write-Host "Usage: .\start_production.ps1 <command> [options]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  start       Start all services"
    Write-Host "  stop        Stop all services"
    Write-Host "  restart     Restart all services"
    Write-Host "  status      Show service status"
    Write-Host "  logs        View logs (optionally specify service name)"
    Write-Host "  rebuild     Rebuild and restart all services"
    Write-Host "  clean       Stop and remove all data volumes"
    Write-Host "  health      Check service health"
    Write-Host "  help        Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\start_production.ps1 start"
    Write-Host "  .\start_production.ps1 logs intent-engine-api"
    Write-Host "  .\start_production.ps1 status"
    Write-Host ""
}

# Main command handler
param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$ServiceName
)

switch ($Command.ToLower()) {
    "start" {
        Check-Prerequisites
        Start-Services
    }
    "stop" {
        Stop-Services
    }
    "restart" {
        Restart-Services
    }
    "status" {
        Show-Status
    }
    "logs" {
        View-Logs -ServiceName $ServiceName
    }
    "rebuild" {
        Rebuild-Services
    }
    "clean" {
        Clean-All
    }
    "health" {
        Check-Health
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Unknown command: $Command"
        Show-Help
        exit 1
    }
}
