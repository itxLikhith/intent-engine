# Docker Compose Test Suite

Comprehensive test suite for Intent Engine Docker Compose configurations.

---

## Overview

This test suite validates all Docker Compose configurations for the Intent Engine project, ensuring they are:

- ✅ Syntactically correct (YAML validation)
- ✅ Properly configured (services, networks, volumes)
- ✅ Running correctly (container health checks)
- ✅ Functionally operational (API endpoint tests)

---

## Quick Start

### Run All Tests

**Python (Cross-platform):**
```bash
python scripts/test_docker_compose.py
```

**Bash (Linux/Mac):**
```bash
./scripts/test_all_compose.sh --full
```

**PowerShell (Windows):**
```powershell
.\scripts\test_all_compose.ps1 -TestSuite full
```

---

## Test Scripts

### 1. Python Test Suite (`test_docker_compose.py`)

**Features:**
- Comprehensive automated testing
- JSON output support
- Multiple test suites
- Detailed reporting

**Usage:**
```bash
# Run all tests
python scripts/test_docker_compose.py

# Test specific compose file
python scripts/test_docker_compose.py -c docker-compose.yml

# Run specific test suite
python scripts/test_docker_compose.py -t validation

# Save results to JSON
python scripts/test_docker_compose.py -o results.json

# Verbose output
python scripts/test_docker_compose.py -v
```

**Options:**
- `-c, --compose-file`: Specific compose file to test
  - `docker-compose.yml`
  - `docker-compose.searxng.yml`
  - `docker-compose.go-crawler.yml`
  - `docker-compose.aio.yml`
- `-t, --test-suite`: Specific test suite to run
  - `validation` - YAML syntax and configuration
  - `health` - Health endpoint checks
  - `api` - API functionality tests
  - `all` - Run all tests (default)
- `-v, --verbose` - Enable verbose output
- `-o, --output` - Output file for results (JSON)

---

### 2. Bash Test Suite (`test_all_compose.sh`)

**Features:**
- Color-coded output
- Modular test suites
- Summary reporting

**Usage:**
```bash
# Run all tests
./scripts/test_all_compose.sh --full

# Validate YAML syntax only
./scripts/test_all_compose.sh --validate

# Run health checks only
./scripts/test_all_compose.sh --health

# Run API tests only
./scripts/test_all_compose.sh --api

# Verbose output
./scripts/test_all_compose.sh --verbose
```

---

### 3. PowerShell Test Suite (`test_all_compose.ps1`)

**Features:**
- Windows PowerShell native
- Color-coded output
- Object-based testing

**Usage:**
```powershell
# Run all tests
.\scripts\test_all_compose.ps1 -TestSuite full

# Validate YAML syntax only
.\scripts\test_all_compose.ps1 -TestSuite validate

# Run health checks only
.\scripts\test_all_compose.ps1 -TestSuite health

# Run API tests only
.\scripts\test_all_compose.ps1 -TestSuite api
```

---

## Test Suites

### Validation Tests

Tests Docker Compose file structure and configuration.

**Tests:**
- File exists
- YAML syntax valid
- Services defined
- Networks configured
- Volumes configured

**Example Output:**
```
✓ File exists: docker-compose.yml
✓ YAML syntax valid: docker-compose.yml
✓ Services defined: Found 5 service(s)
✓ Volumes defined: Volumes configured
```

---

### Container Status Tests

Checks if containers are running and healthy.

**Tests:**
- Containers running
- Health status check
- Service availability

**Example Output:**
```
✓ Containers running: Found 7 container(s)
✓ Container intent-engine-api: Status: running, Health: healthy
✓ Container postgres: Status: running, Health: healthy
```

---

### Health Endpoint Tests

Verifies service health endpoints are responding.

**Endpoints Tested:**
- API Root: `http://localhost:8000/`
- API Health: `http://localhost:8000/health`
- SearXNG: `http://localhost:8080/healthz`
- Go Search: `http://localhost:8081/health`

**Example Output:**
```
✓ API health endpoint responding (HTTP 200)
✓ SearXNG health endpoint responding (HTTP 200)
✓ Go Search API health endpoint responding (HTTP 200)
```

---

### API Functionality Tests

Tests actual API functionality.

**Tests:**
- Root endpoint response
- Health endpoint response
- Search endpoint functionality

**Example Output:**
```
✓ Root endpoint: API responding: healthy
✓ Health endpoint: Health status: healthy
✓ Search endpoint: Search returned 35 results
```

---

## Docker Compose Files

### Files Tested

| File | Purpose | Services |
|------|---------|----------|
| `docker-compose.yml` | Main production stack | 5 + 4 optional |
| `docker-compose.searxng.yml` | Standalone SearXNG | 2 |
| `docker-compose.go-crawler.yml` | Go crawler integration | 3 |
| `docker-compose.aio.yml` | All-in-One deployment | 1 (multi-service) |
| `go-crawler/docker-compose.yml` | Standalone Go stack | 5 |

---

## Interpreting Results

### Test Status Symbols

| Symbol | Status | Meaning |
|--------|--------|---------|
| ✓ | PASSED | Test completed successfully |
| ✗ | FAILED | Test failed |
| ○ | SKIPPED | Test skipped (not applicable) |
| ⚠ | ERROR | Test encountered an error |

### Success Rate

- **100%**: All tests passed
- **80-99%**: Minor issues (warnings)
- **<80%**: Significant issues (review failures)

### Common Issues

**"File not found"**
- Check file path
- Ensure you're in the project root

**"YAML syntax invalid"**
- Run `docker-compose -f <file> config` to see error
- Check indentation and syntax

**"Failed to reach endpoint"**
- Ensure containers are running: `docker-compose ps`
- Check port conflicts
- Verify service health: `docker-compose logs`

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Docker Compose Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker
        uses: docker/setup-buildx-action@v2
      
      - name: Validate Compose Files
        run: |
          for file in docker-compose*.yml; do
            docker-compose -f "$file" config --quiet
          done
      
      - name: Run Test Suite
        run: python scripts/test_docker_compose.py -t validation
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: docker-compose-validate
      name: Validate Docker Compose
      entry: docker-compose -f docker-compose.yml config --quiet
      language: system
      files: docker-compose.*\.yml$
```

---

## Troubleshooting

### Docker Compose Warnings

**"attribute `version` is obsolete"**
- Docker Compose v2+ ignores the version attribute
- Safe to remove or ignore
- Not a critical error

### Test Failures

**Health endpoint tests failing but services working:**
- This is a known issue with the Python test suite
- Manual verification shows all endpoints working
- Use bash/PowerShell scripts for accurate health checks

**Container health showing empty:**
- Some containers don't have health checks configured
- go-crawler and go-indexer run without health checks
- They are operational but not monitored

### Port Conflicts

If ports are already in use:

```bash
# Check what's using the port
netstat -ano | findstr :8000

# Stop conflicting process or change port in .env
```

---

## Performance

### Typical Test Duration

| Test Suite | Duration |
|------------|----------|
| Validation | <1s per file |
| Container Status | <1s |
| Health Endpoints | <2s |
| API Functionality | <5s |
| **Total (all files)** | <10s |

---

## Contributing

### Adding New Tests

1. Add test function to appropriate script
2. Follow naming convention: `test_<feature>()`
3. Use logging functions for consistent output
4. Update this README with new test details

### Test Function Template (Python)

```python
def test_<feature>(self, compose_file: str) -> TestResult:
    """Test description."""
    start = time.time()
    success, stdout, stderr = self.run_command(
        'command to test',
        timeout=10
    )
    duration = time.time() - start
    
    return TestResult(
        name="<feature> test",
        status=TestStatus.PASSED if success else TestStatus.FAILED,
        duration=duration,
        message="Description of result"
    )
```

---

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Compose File Reference](https://docs.docker.com/compose/compose-file/)
- [Intent Engine Deployment Guide](docs/deployment/DEPLOYMENT_CHECKLIST.md)
- [Test Results](docs/deployment/DOCKER_COMPOSE_TEST_RESULTS.md)

---

**License:** Intent Engine Community License (IECL) v1.0  
**Last Updated:** March 15, 2026  
**Version:** 1.0.0
