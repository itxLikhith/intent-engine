# =============================================================================
# Intent Engine - Go Crawler & Indexer Production Startup Script (PowerShell)
# =============================================================================
# This script handles the complete startup process for the Go crawler system
# with health checks and proper initialization.
#
# Usage:
#   .\scripts\start_crawler.ps1 [start|stop|restart|status|logs]
#
# Quick Start:
#   .\scripts\start_crawler.ps1 start
#   Start-Sleep -Seconds 30
#   curl http://localhost:8081/health
# =============================================================================

$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$CrawlerDir = $ProjectDir
$ComposeFile = Join-Path $CrawlerDir "docker-compose.yml"
$ComposeProjectName = "intent-crawler"

# Determine Docker Compose command
function Get-DockerComposeCommand {
    try {
        $null = docker compose version -ErrorAction Stop
        return "docker compose"
    } catch {
        try {
            $null = docker-compose version -ErrorAction Stop
            return "docker-compose"
        } catch {
            throw "Docker Compose is not installed"
        }
    }
}

$DockerCompose = Get-DockerComposeCommand

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Wait for service to be healthy
function Wait-ForService {
    param(
        [string]$ServiceName,
        [string]$Url,
        [int]$MaxAttempts = 30,
        [int]$Delay = 2
    )
    
    Write-Info "Waiting for $ServiceName to be ready..."
    
    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri $Url -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Success "$ServiceName is ready!"
                return $true
            }
        } catch {
            # Ignore errors, keep trying
        }
        
        Write-Host -NoNewline "."
        Start-Sleep -Seconds $Delay
        $attempt++
    }
    
    Write-Host ""
    Write-Error-Custom "$ServiceName failed to start after $MaxAttempts attempts"
    return $false
}

# Wait for PostgreSQL
function Wait-ForPostgres {
    Write-Info "Waiting for PostgreSQL to be ready..."
    
    $maxAttempts = 30
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            $null = docker exec intent-postgres pg_isready -U crawler -d intent_engine 2>$null
            Write-Success "PostgreSQL is ready!"
            return $true
        } catch {
            # Ignore errors, keep trying
        }
        
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 2
        $attempt++
    }
    
    Write-Host ""
    Write-Error-Custom "PostgreSQL failed to start"
    return $false
}

# Check system health
function Test-SystemHealth {
    Write-Info "Checking system health..."
    
    $allHealthy = $true
    
    # Check Search API
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8081/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        Write-Success "✓ Search API is healthy (port 8081)"
    } catch {
        Write-Error-Custom "✗ Search API is not responding"
        $allHealthy = $false
    }
    
    # Check SearXNG
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8082/healthz" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        Write-Success "✓ SearXNG is healthy (port 8082)"
    } catch {
        Write-Error-Custom "✗ SearXNG is not responding"
        $allHealthy = $false
    }
    
    # Check PostgreSQL
    try {
        $null = docker exec intent-postgres pg_isready -U crawler -d intent_engine 2>$null
        Write-Success "✓ PostgreSQL is healthy"
    } catch {
        Write-Error-Custom "✗ PostgreSQL is not ready"
        $allHealthy = $false
    }
    
    # Check Redis
    try {
        $null = docker exec intent-redis redis-cli ping 2>$null
        Write-Success "✓ Redis is healthy"
    } catch {
        Write-Error-Custom "✗ Redis is not responding"
        $allHealthy = $false
    }
    
    if ($allHealthy) {
        Write-Success "All services are healthy!"
        return $true
    } else {
        Write-Error-Custom "Some services are unhealthy"
        return $false
    }
}

# Start all services
function Start-Services {
    Write-Info "Starting Intent Engine Go Crawler System..."
    
    Set-Location $CrawlerDir
    
    # Start services
    Write-Info "Starting Docker containers..."
    & $DockerCompose -f $ComposeFile -p $ComposeProjectName up -d
    
    # Wait for PostgreSQL
    if (!(Wait-ForPostgres)) {
        Write-Error-Custom "Failed to start PostgreSQL"
        return $false
    }
    
    # Wait for Search API
    if (!(Wait-ForService -ServiceName "Search API" -Url "http://localhost:8081/health" -MaxAttempts 60 -Delay 3)) {
        Write-Error-Custom "Failed to start Search API"
        return $false
    }
    
    # Wait for SearXNG
    if (!(Wait-ForService -ServiceName "SearXNG" -Url "http://localhost:8082/healthz" -MaxAttempts 60 -Delay 3)) {
        Write-Error-Custom "Failed to start SearXNG"
        return $false
    }
    
    # Final health check
    Start-Sleep -Seconds 5
    Test-SystemHealth
    
    Write-Success ""
    Write-Success "Intent Engine Go Crawler is ready!"
    Write-Success ""
    Write-Success "Services:"
    Write-Success "  - Search API:    http://localhost:8081"
    Write-Success "  - SearXNG:       http://localhost:8082"
    Write-Success "  - PostgreSQL:    localhost:5432"
    Write-Success "  - Redis:         localhost:6379"
    Write-Success ""
    Write-Success "Test the search API:"
    Write-Success "  curl http://localhost:8081/health"
    Write-Success "  curl -X POST http://localhost:8081/api/v1/search -H 'Content-Type: application/json' -d '{`"query`":`"golang`"}'"
    Write-Success ""
    Write-Success "Monitor crawling:"
    Write-Success "  docker-compose logs -f crawler"
    Write-Success "  curl http://localhost:8081/stats"
    Write-Success ""
    
    return $true
}

# Stop all services
function Stop-Services {
    Write-Info "Stopping Intent Engine Go Crawler..."
    
    Set-Location $CrawlerDir
    & $DockerCompose -f $ComposeFile -p $ComposeProjectName down
    
    Write-Success "Intent Engine Go Crawler stopped"
}

# Restart all services
function Restart-Services {
    Write-Info "Restarting Intent Engine Go Crawler..."
    Stop-Services
    Start-Sleep -Seconds 5
    Start-Services
}

# Show logs
function Show-Logs {
    param([string]$Service = "")
    
    Set-Location $CrawlerDir
    
    if ($Service) {
        & $DockerCompose -f $ComposeFile -p $ComposeProjectName logs -f $Service
    } else {
        & $DockerCompose -f $ComposeFile -p $ComposeProjectName logs -f
    }
}

# Show status
function Show-Status {
    Set-Location $CrawlerDir
    
    Write-Info "Service Status:"
    & $DockerCompose -f $ComposeFile -p $ComposeProjectName ps
    
    Write-Host ""
    Test-SystemHealth
}

# Print usage
function Print-Usage {
    Write-Host "Intent Engine Go Crawler - Startup Script"
    Write-Host ""
    Write-Host "Usage: .\start_crawler.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  start       Start all services"
    Write-Host "  stop        Stop all services"
    Write-Host "  restart     Restart all services"
    Write-Host "  status      Show service status"
    Write-Host "  logs        Show logs (optionally specify service name)"
    Write-Host "  health      Check system health"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\start_crawler.ps1 start"
    Write-Host "  .\start_crawler.ps1 logs crawler"
    Write-Host "  .\start_crawler.ps1 health"
    Write-Host ""
}

# Main entry point
param(
    [Parameter(Position = 0)]
    [string]$Command = "start",
    
    [Parameter(Position = 1)]
    [string]$ServiceName = ""
)

# Main execution
switch ($Command.ToLower()) {
    "start" {
        if (!(Start-Services)) {
            exit 1
        }
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
        Show-Logs -Service $ServiceName
    }
    "health" {
        Test-SystemHealth
    }
    default {
        Print-Usage
        exit 1
    }
}
