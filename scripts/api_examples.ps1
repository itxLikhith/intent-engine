# =============================================================================
# Intent Engine - Search API Examples (PowerShell)
# =============================================================================
# This script contains example curl commands for testing the search API.
#
# Usage:
#   .\scripts\api_examples.ps1 [endpoint_name]
#
# Examples:
#   .\scripts\api_examples.ps1 health       # Health checks
#   .\scripts\api_examples.ps1 search       # Search endpoints
#   .\scripts\api_examples.ps1 all          # Run all examples
# =============================================================================

$BASE_URL = $env:API_BASE_URL ?? "http://localhost:8000"
$SEARXNG_URL = $env:SEARXNG_URL ?? "http://localhost:8080"

function Print-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host $Text -ForegroundColor Blue
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host ""
}

function Print-Example {
    param([string]$Text)
    Write-Host "Example: $Text" -ForegroundColor Yellow
}

function Print-Response {
    param([string]$Response)
    Write-Host "Response:" -ForegroundColor Green
    try {
        $Response | ConvertFrom-Json | ConvertTo-Json -Depth 10
    } catch {
        Write-Host $Response
    }
    Write-Host ""
}

# =============================================================================
# Health Checks
# =============================================================================
function Test-Health {
    Print-Header "Health Checks"

    Print-Example "Root health check"
    $response = Invoke-RestMethod -Uri "$BASE_URL/" -Method Get -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)

    Print-Example "Detailed health check"
    $response = Invoke-RestMethod -Uri "$BASE_URL/health" -Method Get -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)

    Print-Example "SearXNG health"
    $response = Invoke-RestMethod -Uri "$SEARXNG_URL/healthz" -Method Get -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)

    Print-Example "Metrics endpoint (first 20 lines)"
    $response = Invoke-RestMethod -Uri "$BASE_URL/metrics" -Method Get -ErrorAction SilentlyContinue
    $response -split "`n" | Select-Object -First 20
    Write-Host ""
}

# =============================================================================
# Search Endpoints
# =============================================================================
function Test-Search {
    Print-Header "Search Endpoints"

    Print-Example "Basic search"
    $body = @{
        query = "weather today"
    } | ConvertTo-Json
    $response = Invoke-RestMethod -Uri "$BASE_URL/search" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)

    Print-Example "Search with intent extraction"
    $body = @{
        query = "best laptop for programming under `$1000"
        extract_intent = $true
        rank_results = $true
    } | ConvertTo-Json
    $response = Invoke-RestMethod -Uri "$BASE_URL/search" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)

    Print-Example "Search with privacy filters"
    $body = @{
        query = "encrypted messaging apps"
        exclude_big_tech = $true
        min_privacy_score = 0.7
    } | ConvertTo-Json
    $response = Invoke-RestMethod -Uri "$BASE_URL/search" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)

    Print-Example "Search in specific categories"
    $body = @{
        query = "python tutorial"
        categories = @("general", "it")
        language = "en"
    } | ConvertTo-Json
    $response = Invoke-RestMethod -Uri "$BASE_URL/search" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)
}

# =============================================================================
# Intent Extraction
# =============================================================================
function Test-Intent {
    Print-Header "Intent Extraction Endpoints"

    Print-Example "Extract intent - shopping query"
    $body = @{
        product = "search"
        input = @{
            text = "best laptop for programming under 50000 rupees"
        }
        context = @{
            sessionId = "test-123"
            userLocale = "en-US"
        }
    } | ConvertTo-Json -Depth 10
    $response = Invoke-RestMethod -Uri "$BASE_URL/extract-intent" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)

    Print-Example "Extract intent - troubleshooting query"
    $body = @{
        product = "search"
        input = @{
            text = "how to fix blue screen error Windows 11"
        }
        context = @{
            sessionId = "test-124"
            userLocale = "en-US"
        }
    } | ConvertTo-Json -Depth 10
    $response = Invoke-RestMethod -Uri "$BASE_URL/extract-intent" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)

    Print-Example "Extract intent - comparison query"
    $body = @{
        product = "search"
        input = @{
            text = "iPhone 15 vs Samsung S24 comparison"
        }
        context = @{
            sessionId = "test-125"
            userLocale = "en-US"
        }
    } | ConvertTo-Json -Depth 10
    $response = Invoke-RestMethod -Uri "$BASE_URL/extract-intent" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)

    Print-Example "Extract intent - learning query"
    $body = @{
        product = "search"
        input = @{
            text = "learn python programming for beginners"
        }
        context = @{
            sessionId = "test-126"
            userLocale = "en-US"
        }
    } | ConvertTo-Json -Depth 10
    $response = Invoke-RestMethod -Uri "$BASE_URL/extract-intent" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)
}

# =============================================================================
# URL Ranking
# =============================================================================
function Test-Ranking {
    Print-Header "URL Ranking Endpoints"

    Print-Example "Rank URLs"
    $body = @{
        query = "best programming laptops"
        urls = @(
            "https://www.laptopmag.com/articles/best-programming-laptops",
            "https://www.pcmag.com/picks/the-best-laptops-for-programming",
            "https://www.techradar.com/news/best-laptops-for-coding"
        )
        options = @{
            weights = @{
                relevance = 0.5
                privacy = 0.3
                quality = 0.2
            }
        }
    } | ConvertTo-Json -Depth 10
    $response = Invoke-RestMethod -Uri "$BASE_URL/rank-urls" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)
}

# =============================================================================
# Service Recommendation
# =============================================================================
function Test-Services {
    Print-Header "Service Recommendation Endpoints"

    Print-Example "Recommend services"
    $body = @{
        product = "search"
        input = @{
            text = "need help setting up encrypted email"
        }
        context = @{
            sessionId = "test-128"
            userLocale = "en-US"
        }
    } | ConvertTo-Json -Depth 10
    $response = Invoke-RestMethod -Uri "$BASE_URL/recommend-services" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
    Print-Response ($response | ConvertTo-Json)
}

# =============================================================================
# Check Services
# =============================================================================
function Test-ServicesAvailable {
    Print-Header "Checking Service Availability"
    
    Write-Host -NoNewline "Checking API at $BASE_URL... "
    try {
        $null = Invoke-RestMethod -Uri "$BASE_URL/" -Method Get -ErrorAction Stop
        Write-Host "✓ Available" -ForegroundColor Green
    } catch {
        Write-Host "✗ Not available" -ForegroundColor Yellow
        Write-Host "Please start the services first:" -ForegroundColor Red
        Write-Host "  .\scripts\production_start.ps1 start" -ForegroundColor Yellow
        return $false
    }

    Write-Host -NoNewline "Checking SearXNG at $SEARXNG_URL... "
    try {
        $null = Invoke-RestMethod -Uri "$SEARXNG_URL/healthz" -Method Get -ErrorAction Stop
        Write-Host "✓ Available" -ForegroundColor Green
    } catch {
        Write-Host "✗ Not available" -ForegroundColor Yellow
    }
    Write-Host ""
    
    return $true
}

# =============================================================================
# Main
# =============================================================================
param(
    [Parameter(Position = 0)]
    [string]$Command = "all"
)

if (!(Test-ServicesAvailable)) {
    exit 1
}

switch ($Command.ToLower()) {
    "health" {
        Test-Health
    }
    "search" {
        Test-Search
    }
    "intent" {
        Test-Intent
    }
    "ranking" {
        Test-Ranking
    }
    "services" {
        Test-Services
    }
    "all" {
        Test-Health
        Test-Search
        Test-Intent
    }
    default {
        Write-Host "Intent Engine - Search API Examples"
        Write-Host ""
        Write-Host "Usage: .\api_examples.ps1 [endpoint]"
        Write-Host ""
        Write-Host "Endpoints:"
        Write-Host "  health    - Health checks and metrics"
        Write-Host "  search    - Search endpoints"
        Write-Host "  intent    - Intent extraction"
        Write-Host "  ranking   - URL ranking"
        Write-Host "  services  - Service recommendation"
        Write-Host "  all       - Run all examples"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  .\api_examples.ps1 health"
        Write-Host "  .\api_examples.ps1 search"
        Write-Host "  .\api_examples.ps1 all"
        Write-Host ""
    }
}
