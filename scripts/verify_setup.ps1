# =============================================================================
# Intent Engine - Setup Verification Script (PowerShell)
# =============================================================================
# This script verifies that all components are properly set up and working.
#
# Usage:
#   .\scripts\verify_setup.ps1
# =============================================================================

$ErrorActionPreference = "Continue"

# Counters
$Pass = 0
$Fail = 0
$Warn = 0

function Log-Pass {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
    $script:Pass++
}

function Log-Fail {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
    $script:Fail++
}

function Log-Warn {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
    $script:Warn++
}

function Log-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Blue
}

function Print-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host $Text -ForegroundColor Blue
    Write-Host "========================================" -ForegroundColor Blue
}

# =============================================================================
# Prerequisites Check
# =============================================================================
function Check-Prerequisites {
    Print-Header "Checking Prerequisites"
    
    # Docker
    try {
        $dockerVersion = docker --version 2>&1
        Log-Pass "Docker installed: $dockerVersion"
    } catch {
        Log-Fail "Docker not installed"
    }
    
    # Docker Compose
    try {
        $composeVersion = docker compose version --short 2>&1
        Log-Pass "Docker Compose installed: $composeVersion"
    } catch {
        try {
            $composeVersion = docker-compose version --short 2>&1
            Log-Pass "Docker Compose installed: $composeVersion"
        } catch {
            Log-Fail "Docker Compose not installed"
        }
    }
    
    # curl
    if (Get-Command curl -ErrorAction SilentlyContinue) {
        Log-Pass "curl installed"
    } else {
        Log-Warn "curl not installed (needed for API testing)"
    }
    
    # Python (optional)
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonVersion = python --version 2>&1
        Log-Pass "Python installed: $pythonVersion"
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        $pythonVersion = python3 --version 2>&1
        Log-Pass "Python installed: $pythonVersion"
    } else {
        Log-Warn "Python not installed (optional, for scripts)"
    }
}

# =============================================================================
# File Structure Check
# =============================================================================
function Check-Files {
    Print-Header "Checking File Structure"
    
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ProjectDir = Split-Path -Parent $ScriptDir
    
    # Required files
    $requiredFiles = @(
        "docker-compose.prod.yml",
        ".env",
        "Dockerfile",
        "requirements.txt",
        "main_api.py",
        "README_PRODUCTION.md"
    )
    
    foreach ($file in $requiredFiles) {
        $filePath = Join-Path $ProjectDir $file
        if (Test-Path $filePath) {
            Log-Pass "Found: $file"
        } else {
            Log-Fail "Missing: $file"
        }
    }
    
    # Required directories
    $requiredDirs = @("scripts", "searxng", "core", "extraction", "ranking")
    
    foreach ($dir in $requiredDirs) {
        $dirPath = Join-Path $ProjectDir $dir
        if (Test-Path $dirPath -PathType Container) {
            Log-Pass "Found directory: $dir"
        } else {
            Log-Warn "Missing directory: $dir"
        }
    }
    
    # Check .env configuration
    $envFile = Join-Path $ProjectDir ".env"
    if (Test-Path $envFile) {
        $envContent = Get-Content $envFile -Raw
        
        if ($envContent -match "SECRET_KEY=change-this-to-a-secure-random-string-in-production") {
            Log-Warn "SECRET_KEY not changed from default (OK for testing, change for production)"
        } else {
            Log-Pass "SECRET_KEY configured"
        }
        
        if ($envContent -match "intent_secure_password_change_in_prod") {
            Log-Warn "Database password not changed from default (OK for testing, change for production)"
        } else {
            Log-Pass "Database password configured"
        }
    } else {
        Log-Warn ".env file not found (will use defaults)"
    }
}

# =============================================================================
# Docker Configuration Check
# =============================================================================
function Check-DockerConfig {
    Print-Header "Checking Docker Configuration"
    
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ProjectDir = Split-Path -Parent $ScriptDir
    $ComposeFile = Join-Path $ProjectDir "docker-compose.prod.yml"
    
    # Validate docker-compose file
    if (Test-Path $ComposeFile) {
        Set-Location $ProjectDir
        try {
            $null = docker compose -f $ComposeFile config --quiet 2>&1
            Log-Pass "docker-compose.prod.yml is valid"
        } catch {
            Log-Fail "docker-compose.prod.yml has errors"
        }
    }
    
    # Check Docker resources
    try {
        $dockerInfo = docker info 2>&1 | Out-String
        if ($dockerInfo -match "Total Memory:\s+(\d+)(\w+)") {
            $memoryValue = [int]$matches[1]
            $memoryUnit = $matches[2]
            
            # Convert to bytes
            $multiplier = switch ($memoryUnit) {
                "GiB" { 1GB }
                "MiB" { 1MB }
                "KiB" { 1KB }
                default { 1 }
            }
            $totalBytes = $memoryValue * $multiplier
            
            Log-Info "Available memory: $($memoryValue)$($memoryUnit)"
            
            if ($totalBytes -lt 2GB) {
                Log-Warn "Less than 2GB memory available (may experience performance issues)"
            } else {
                Log-Pass "Sufficient memory available"
            }
        }
    } catch {
        Log-Warn "Could not retrieve Docker info"
    }
    
    # Check disk space
    $diskInfo = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Root -eq (Split-Path $ProjectDir -Qualifier) }
    if ($diskInfo) {
        $freeSpaceGB = [math]::Round($diskInfo.Free / 1GB, 2)
        Log-Info "Available disk space: ${freeSpaceGB}GB"
    }
}

# =============================================================================
# Service Connectivity Check (if running)
# =============================================================================
function Check-Services {
    Print-Header "Checking Running Services"
    
    # Check if services are running
    try {
        $containers = docker ps --format "{{.Names}}" 2>&1
        
        if ($containers -match "intent-engine-api") {
            Log-Pass "Intent Engine API is running"
            
            # Test API health
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:8000/" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
                if ($response.StatusCode -eq 200) {
                    Log-Pass "API health check passed"
                } else {
                    Log-Fail "API health check failed"
                }
            } catch {
                Log-Fail "API health check failed"
            }
        } else {
            Log-Warn "Intent Engine API is not running"
            Log-Info "Start with: .\scripts\production_start.ps1 start"
        }
        
        if ($containers -match "searxng") {
            Log-Pass "SearXNG is running"
            
            # Test SearXNG health
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:8080/healthz" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
                if ($response.StatusCode -eq 200) {
                    Log-Pass "SearXNG health check passed"
                } else {
                    Log-Fail "SearXNG health check failed"
                }
            } catch {
                Log-Fail "SearXNG health check failed"
            }
        } else {
            Log-Warn "SearXNG is not running"
        }
        
        if ($containers -match "intent-engine-postgres") {
            Log-Pass "PostgreSQL is running"
            
            # Test PostgreSQL health
            try {
                $null = docker exec intent-engine-postgres pg_isready -U intent_user -d intent_engine 2>$null
                Log-Pass "PostgreSQL health check passed"
            } catch {
                Log-Fail "PostgreSQL health check failed"
            }
        } else {
            Log-Warn "PostgreSQL is not running"
        }
        
        if ($containers -match "intent-redis") {
            Log-Pass "Redis is running"
            
            # Test Redis health
            try {
                $null = docker exec intent-redis valkey-cli ping 2>$null
                Log-Pass "Redis health check passed"
            } catch {
                Log-Fail "Redis health check failed"
            }
        } else {
            Log-Warn "Redis is not running"
        }
    } catch {
        Log-Warn "Could not check running services"
    }
}

# =============================================================================
# Summary
# =============================================================================
function Print-Summary {
    Print-Header "Verification Summary"
    
    Write-Host "  Passed:   $Pass" -ForegroundColor Green
    Write-Host "  Failed:   $Fail" -ForegroundColor Red
    Write-Host "  Warnings: $Warn" -ForegroundColor Yellow
    Write-Host ""
    
    if ($Fail -eq 0) {
        Write-Host "✓ Setup verification passed!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Blue
        Write-Host "  1. Start services: .\scripts\production_start.ps1 start"
        Write-Host "  2. Wait for initialization: Start-Sleep -Seconds 60"
        Write-Host "  3. Test search: curl http://localhost:8000/search -X POST -H 'Content-Type: application/json' -d '{`"query`":`"test`"}'"
        Write-Host ""
        exit 0
    } else {
        Write-Host "✗ Setup verification failed!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please fix the failed checks above."
        Write-Host "See QUICKSTART.md for detailed setup instructions."
        Write-Host ""
        exit 1
    }
}

# =============================================================================
# Main
# =============================================================================
Write-Host "Intent Engine - Setup Verification" -ForegroundColor Blue
Write-Host ""

Check-Prerequisites
Check-Files
Check-DockerConfig
Check-Services
Print-Summary
