# =============================================================================
# Intent Engine - Production Startup Script (PowerShell)
# =============================================================================
# This script handles the complete startup process for the Intent Engine
# search engine backend with health checks and proper initialization.
#
# Usage:
#   .\scripts\production_start.ps1 [start|stop|restart|status|logs]
#
# Quick Start:
#   .\scripts\production_start.ps1 start
#   Start-Sleep -Seconds 60
#   curl http://localhost:8000/search -X POST -H "Content-Type: application/json" -d '{"query":"best laptop for programming"}'
# =============================================================================

$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$ComposeFile = Join-Path $ProjectDir "docker-compose.prod.yml"
$ComposeProjectName = "intent-engine"

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

# Check if Docker is available
function Test-Docker {
    try {
        $null = docker --version -ErrorAction Stop
        $null = $DockerCompose -ErrorAction Stop
        Write-Info "Docker and Docker Compose are available"
        return $true
    } catch {
        Write-Error-Custom "Docker or Docker Compose is not installed"
        return $false
    }
}

# Check if .env file exists
function Test-EnvFile {
    $envFile = Join-Path $ProjectDir ".env"
    if (Test-Path $envFile) {
        Write-Info "Using .env file: $envFile"
        return $true
    } else {
        Write-Warning-Custom ".env file not found, using default environment variables"
        Write-Info "Consider copying .env.example to .env and customizing it"
        return $false
    }
}

# Wait for a service to be healthy
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

# Wait for PostgreSQL to be ready
function Wait-ForPostgres {
    Write-Info "Waiting for PostgreSQL to be ready..."
    
    $maxAttempts = 30
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            $null = docker exec intent-engine-postgres pg_isready -U intent_user -d intent_engine 2>$null
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

# Check overall system health
function Test-SystemHealth {
    Write-Info "Checking system health..."
    
    $allHealthy = $true
    
    # Check API
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8000/" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        Write-Success "✓ API is healthy (port 8000)"
    } catch {
        Write-Error-Custom "✗ API is not responding"
        $allHealthy = $false
    }
    
    # Check SearXNG
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8080/healthz" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        Write-Success "✓ SearXNG is healthy (port 8080)"
    } catch {
        Write-Error-Custom "✗ SearXNG is not responding"
        $allHealthy = $false
    }
    
    # Check PostgreSQL
    try {
        $null = docker exec intent-engine-postgres pg_isready -U intent_user -d intent_engine 2>$null
        Write-Success "✓ PostgreSQL is healthy (port 5432)"
    } catch {
        Write-Error-Custom "✗ PostgreSQL is not ready"
        $allHealthy = $false
    }
    
    # Check Redis
    try {
        $null = docker exec intent-redis valkey-cli ping 2>$null
        Write-Success "✓ Redis is healthy (port 6379)"
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
    Write-Info "Starting Intent Engine search backend..."
    
    Set-Location $ProjectDir
    
    # Start services
    Write-Info "Starting Docker containers..."
    & $DockerCompose -f $ComposeFile -p $ComposeProjectName up -d
    
    # Wait for PostgreSQL
    if (!(Wait-ForPostgres)) {
        Write-Error-Custom "Failed to start PostgreSQL"
        return $false
    }
    
    # Wait for migrations
    Write-Info "Waiting for database migrations..."
    Start-Sleep -Seconds 10
    
    # Wait for SearXNG
    if (!(Wait-ForService -ServiceName "SearXNG" -Url "http://localhost:8080/healthz" -MaxAttempts 30 -Delay 2)) {
        Write-Error-Custom "Failed to start SearXNG"
        return $false
    }
    
    # Wait for API
    if (!(Wait-ForService -ServiceName "Intent Engine API" -Url "http://localhost:8000/" -MaxAttempts 60 -Delay 3)) {
        Write-Error-Custom "Failed to start Intent Engine API"
        return $false
    }
    
    # Final health check
    Start-Sleep -Seconds 5
    Test-SystemHealth
    
    Write-Success ""
    Write-Success "Intent Engine is ready!"
    Write-Success ""
    Write-Success "API Endpoint: http://localhost:8000"
    Write-Success "Search Endpoint: http://localhost:8000/search"
    Write-Success "SearXNG: http://localhost:8080"
    Write-Success "PostgreSQL: localhost:5432"
    Write-Success "Redis: localhost:6379"
    Write-Success ""
    Write-Info "Test the search API:"
    Write-Info '  curl http://localhost:8000/search -X POST -H "Content-Type: application/json" -d "{\"query\":\"best laptop for programming\"}"'
    Write-Success ""
    
    return $true
}

# Stop all services
function Stop-Services {
    Write-Info "Stopping Intent Engine..."
    
    Set-Location $ProjectDir
    & $DockerCompose -f $ComposeFile -p $ComposeProjectName down
    
    Write-Success "Intent Engine stopped"
}

# Restart all services
function Restart-Services {
    Write-Info "Restarting Intent Engine..."
    Stop-Services
    Start-Sleep -Seconds 5
    Start-Services
}

# Show logs
function Show-Logs {
    param([string]$Service = "")
    
    Set-Location $ProjectDir
    
    if ($Service) {
        & $DockerCompose -f $ComposeFile -p $ComposeProjectName logs -f $Service
    } else {
        & $DockerCompose -f $ComposeFile -p $ComposeProjectName logs -f
    }
}

# Show status
function Show-Status {
    Set-Location $ProjectDir
    
    Write-Info "Service Status:"
    & $DockerCompose -f $ComposeFile -p $ComposeProjectName ps
    
    Write-Host ""
    Test-SystemHealth
}

# Print usage
function Print-Usage {
    Write-Host "Intent Engine - Production Startup Script"
    Write-Host ""
    Write-Host "Usage: .\production_start.ps1 [command]"
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
    Write-Host "  .\production_start.ps1 start"
    Write-Host "  .\production_start.ps1 logs intent-engine-api"
    Write-Host "  .\production_start.ps1 health"
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
if (!(Test-Docker)) {
    exit 1
}

Test-EnvFile

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
