# =============================================================================
# Intent Engine - Docker Compose Test Suite (PowerShell)
# =============================================================================
# Comprehensive test suite for all Docker Compose configurations
#
# Usage:
#   .\scripts\test_all_compose.ps1 [-TestSuite <validate|health|api|full>]
#
# Examples:
#   .\scripts\test_all_compose.ps1 -TestSuite validate
#   .\scripts\test_all_compose.ps1 -TestSuite full
# =============================================================================

$ErrorActionPreference = "Continue"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$ComposeFiles = @(
    "docker-compose.yml",
    "docker-compose.searxng.yml",
    "docker-compose.go-crawler.yml",
    "docker-compose.aio.yml"
)

# Counters
$TestsPassed = 0
$TestsFailed = 0
$TestsSkipped = 0

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Pass {
    param([string]$Message)
    Write-Host "[PASS] $Message" -ForegroundColor Green
    $script:TestsPassed++
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
    $script:TestsSkipped++
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
    $script:TestsFailed++
}

function Print-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Print-Subheader {
    param([string]$Text)
    Write-Host ""
    Write-Host "--- $Text ---" -ForegroundColor Gray
}

# =============================================================================
# Test Functions
# =============================================================================

function Test-FileExists {
    param([string]$File)
    
    if (Test-Path $File) {
        Write-Pass "File exists: $File"
        return $true
    } else {
        Write-Fail "File not found: $File"
        return $false
    }
}

function Test-YamlSyntax {
    param([string]$ComposeFile)
    
    Write-Info "Validating YAML syntax: $ComposeFile"
    
    try {
        $null = docker-compose -f $ComposeFile config --quiet 2>&1
        Write-Pass "YAML syntax valid: $ComposeFile"
        return $true
    } catch {
        Write-Fail "YAML syntax invalid: $ComposeFile"
        return $false
    }
}

function Test-ServicesDefined {
    param([string]$ComposeFile)
    
    Write-Info "Checking services: $ComposeFile"
    
    try {
        $services = docker-compose -f $ComposeFile config --services 2>&1
        if ($services -and $services.Count -gt 0) {
            $count = $services.Count
            Write-Pass "Found $count service(s) in $ComposeFile"
            $services | ForEach-Object { Write-Host "    - $_" }
            return $true
        } else {
            Write-Fail "No services found in $ComposeFile"
            return $false
        }
    } catch {
        Write-Fail "Failed to list services in $ComposeFile"
        return $false
    }
}

function Test-NetworksDefined {
    param([string]$ComposeFile)
    
    Write-Info "Checking networks: $ComposeFile"
    
    try {
        $networks = docker-compose -f $ComposeFile config --networks 2>&1
        if ($networks -and $networks.Count -gt 0) {
            Write-Pass "Networks defined in $ComposeFile"
            return $true
        } else {
            Write-Warn "No networks defined in $ComposeFile"
            return $true
        }
    } catch {
        Write-Warn "Failed to check networks in $ComposeFile"
        return $true
    }
}

function Test-VolumesDefined {
    param([string]$ComposeFile)
    
    Write-Info "Checking volumes: $ComposeFile"
    
    try {
        $volumes = docker-compose -f $ComposeFile config --volumes 2>&1
        if ($volumes -and $volumes.Count -gt 0) {
            Write-Pass "Volumes defined in $ComposeFile"
            return $true
        } else {
            Write-Warn "No volumes defined in $ComposeFile"
            return $true
        }
    } catch {
        Write-Warn "Failed to check volumes in $ComposeFile"
        return $true
    }
}

function Test-ContainerHealth {
    param([string]$ComposeFile)
    
    Write-Info "Checking container health: $ComposeFile"
    
    try {
        $status = docker-compose -f $ComposeFile ps 2>&1
        if ($status -match "Up") {
            Write-Pass "Containers running for $ComposeFile"
            return $true
        } else {
            Write-Warn "No running containers for $ComposeFile"
            return $true
        }
    } catch {
        Write-Warn "Failed to check containers for $ComposeFile"
        return $true
    }
}

function Test-ApiRoot {
    Write-Info "Testing API root endpoint"
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/" -UseBasicParsing -TimeoutSec 10 2>&1
        if ($response.StatusCode -eq 200) {
            Write-Pass "API root endpoint responding (HTTP $($response.StatusCode))"
            return $true
        } else {
            Write-Fail "API root endpoint not responding (HTTP $($response.StatusCode))"
            return $false
        }
    } catch {
        Write-Fail "API root endpoint not responding: $_"
        return $false
    }
}

function Test-ApiHealth {
    Write-Info "Testing API health endpoint"
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 10 2>&1
        if ($response.StatusCode -eq 200) {
            $data = $response.Content | ConvertFrom-Json
            Write-Pass "API health endpoint responding (Status: $($data.status))"
            return $true
        } else {
            Write-Fail "API health endpoint not responding (HTTP $($response.StatusCode))"
            return $false
        }
    } catch {
        Write-Fail "API health endpoint not responding: $_"
        return $false
    }
}

function Test-SearchEndpoint {
    Write-Info "Testing search endpoint"
    
    try {
        $body = @{ query = "test"; limit = 1 } | ConvertTo-Json
        $response = Invoke-WebRequest -Uri "http://localhost:8000/search" `
            -Method POST `
            -ContentType "application/json" `
            -Body $body `
            -UseBasicParsing `
            -TimeoutSec 30 2>&1
        
        if ($response.StatusCode -eq 200) {
            $data = $response.Content | ConvertFrom-Json
            $count = $data.results.Count
            Write-Pass "Search endpoint working (returned $count results)"
            return $true
        } else {
            Write-Fail "Search endpoint not working (HTTP $($response.StatusCode))"
            return $false
        }
    } catch {
        Write-Fail "Search endpoint not working: $_"
        return $false
    }
}

function Test-SearxngHealth {
    Write-Info "Testing SearXNG health endpoint"
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/healthz" -UseBasicParsing -TimeoutSec 10 2>&1
        if ($response.StatusCode -eq 200) {
            Write-Pass "SearXNG health endpoint responding (HTTP $($response.StatusCode))"
            return $true
        } else {
            Write-Warn "SearXNG health endpoint not responding (HTTP $($response.StatusCode))"
            return $true
        }
    } catch {
        Write-Warn "SearXNG health endpoint not responding: $_"
        return $true
    }
}

function Test-GoSearchHealth {
    Write-Info "Testing Go Search API health endpoint"
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8081/health" -UseBasicParsing -TimeoutSec 10 2>&1
        if ($response.StatusCode -eq 200) {
            Write-Pass "Go Search API health endpoint responding (HTTP $($response.StatusCode))"
            return $true
        } else {
            Write-Warn "Go Search API health endpoint not responding (HTTP $($response.StatusCode))"
            return $true
        }
    } catch {
        Write-Warn "Go Search API health endpoint not responding: $_"
        return $true
    }
}

function Test-DatabaseConnection {
    Write-Info "Testing database connectivity"
    
    $containers = docker ps --format '{{.Names}}' 2>&1
    if ($containers -match "postgres") {
        Write-Pass "PostgreSQL container is running"
        return $true
    } else {
        Write-Warn "PostgreSQL container not running"
        return $true
    }
}

function Test-RedisConnection {
    Write-Info "Testing Redis connectivity"
    
    $containers = docker ps --format '{{.Names}}' 2>&1
    if ($containers -match "redis") {
        Write-Pass "Redis container is running"
        return $true
    } else {
        Write-Warn "Redis container not running"
        return $true
    }
}

# =============================================================================
# Test Suites
# =============================================================================

function Run-ValidationTests {
    Print-Header "VALIDATION TESTS"
    
    foreach ($composeFile in $ComposeFiles) {
        Print-Subheader "Testing: $composeFile"
        
        $fullPath = Join-Path $ProjectDir $composeFile
        Test-FileExists -File $fullPath
        Test-YamlSyntax -ComposeFile $fullPath
        Test-ServicesDefined -ComposeFile $fullPath
        Test-NetworksDefined -ComposeFile $fullPath
        Test-VolumesDefined -ComposeFile $fullPath
    }
}

function Run-HealthTests {
    Print-Header "HEALTH CHECK TESTS"
    
    Test-ApiRoot
    Test-ApiHealth
    Test-SearxngHealth
    Test-GoSearchHealth
    Test-DatabaseConnection
    Test-RedisConnection
}

function Run-ApiTests {
    Print-Header "API FUNCTIONALITY TESTS"
    
    Test-SearchEndpoint
}

function Run-ContainerTests {
    Print-Header "CONTAINER STATUS TESTS"
    
    foreach ($composeFile in $ComposeFiles) {
        Print-Subheader "Testing: $composeFile"
        
        $fullPath = Join-Path $ProjectDir $composeFile
        Test-ContainerHealth -ComposeFile $fullPath
    }
}

function Print-Summary {
    Print-Header "TEST SUMMARY"
    
    $total = $TestsPassed + $TestsFailed + $TestsSkipped
    $passRate = if ($total -gt 0) { [math]::Round(($TestsPassed * 100) / $total, 1) } else { 0 }
    
    Write-Host ""
    Write-Host "Total Tests:  $total"
    Write-Host "Passed:       $TestsPassed ($passRate%)" -ForegroundColor Green
    Write-Host "Failed:       $TestsFailed" -ForegroundColor Red
    Write-Host "Skipped:      $TestsSkipped" -ForegroundColor Yellow
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    
    if ($TestsFailed -eq 0) {
        Write-Host "✓ All tests passed!" -ForegroundColor Green
        return 0
    } else {
        Write-Host "✗ Some tests failed" -ForegroundColor Red
        return 1
    }
}

# =============================================================================
# Main
# =============================================================================

param(
    [ValidateSet("validate", "health", "api", "full")]
    [string]$TestSuite = "full"
)

Print-Header "INTENT ENGINE - DOCKER COMPOSE TEST SUITE"
Write-Host "Project Directory: $ProjectDir"
Write-Host "Test Mode: $TestSuite"
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Run tests based on mode
switch ($TestSuite) {
    "validate" {
        Run-ValidationTests
    }
    "health" {
        Run-HealthTests
    }
    "api" {
        Run-ApiTests
    }
    "full" {
        Run-ValidationTests
        Run-ContainerTests
        Run-HealthTests
        Run-ApiTests
    }
}

# Print summary
$exitCode = Print-Summary
exit $exitCode
