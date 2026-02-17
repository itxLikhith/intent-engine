# Configuration Changes: PostgreSQL, Redis, and CORS

> **Note:** This document is a historical record of configuration changes made to the project. For the current configuration, please refer to the relevant source code files (`docker-compose.yml`, `.env.example`, `main_api.py`, etc.).

## Summary

This document describes the changes made to configure the Intent Engine to use:
1. **PostgreSQL** instead of SQLite for production database
2. **Redis** for distributed caching
3. **Configurable CORS** origins for production security

---

## 1. PostgreSQL Configuration

### Changes Made

#### `database.py`
- Added connection pool settings from environment variables:
  - `DATABASE_POOL_SIZE` (default: 10)
  - `DATABASE_MAX_OVERFLOW` (default: 20)
  - `DATABASE_POOL_TIMEOUT` (default: 30 seconds)
  - `DATABASE_POOL_RECYCLE` (default: 1800 seconds)
- Configured PostgreSQL-specific engine settings with proper pooling
- Added logging for database initialization

#### `.env`
```bash
DATABASE_URL=postgresql://intent_user:intent_secure_password_change_in_prod@postgres:5432/intent_engine
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=1800
```

#### `docker-compose.yml`
- Updated PostgreSQL service with secure credentials
- Added `depends_on: postgres` to intent-engine-api
- Configured proper health checks

---

## 2. Redis Caching Configuration

### Changes Made

#### `requirements.txt`
- Added `redis==5.0.1` for Redis client support

#### `.env`
```bash
REDIS_ENABLED=true
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_MAX_CONNECTIONS=50
REDIS_TIMEOUT=5.0
```

#### `docker-compose.yml`
- Added Redis environment variables to intent-engine-api
- Added `depends_on: redis` to intent-engine-api
- Redis (Valkey) service already configured with persistence

---

## 3. CORS Configuration

### Changes Made

#### `main_api.py`
- Added `os` import for environment variable access
- Added helper functions:
  - `get_cors_origins()` - Parse comma-separated origins
  - `get_cors_allow_methods()` - Parse allowed HTTP methods
  - `get_cors_allow_headers()` - Parse allowed headers
- Made CORS middleware dynamic based on environment variables
- Added `ENABLE_CORS` toggle
- Added logging for CORS configuration

#### `.env`
```bash
ENABLE_CORS=true
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,https://yourdomain.com
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=Authorization,Content-Type,X-Requested-With
```

#### `docker-compose.yml`
- Added all CORS environment variables to intent-engine-api

---

## Files Modified

| File | Changes |
|------|---------|
| `.env` | Created with PostgreSQL, Redis, CORS settings |
| `database.py` | PostgreSQL connection pooling configuration |
| `main_api.py` | Dynamic CORS configuration from environment |
| `docker-compose.yml` | Redis env vars, PostgreSQL credentials, dependencies |
| `requirements.txt` | Added redis==5.0.1 |
| `searxng/client.py` | Fixed indentation error (pre-existing bug) |

---

## Running the Configuration

### Start All Services
```bash
cd "intent engine"
docker-compose up -d
```

### Check Service Status
```bash
docker-compose ps
```

### View Logs
```bash
docker-compose logs -f intent-engine-api
```

### Test Health Endpoint
```bash
curl http://localhost:8000/
```

### Test Intent Extraction
```bash
curl -X POST http://localhost:8000/extract-intent \
  -H "Content-Type: application/json" \
  -d '{"product": "search", "input": {"text": "best laptop for programming"}}'
```

---

## Production Deployment Checklist

### Security
- [ ] Change `SECRET_KEY` to a secure random string
- [ ] Change PostgreSQL password in `.env` and `docker-compose.yml`
- [ ] Update `CORS_ORIGINS` to your actual frontend domains
- [ ] Enable SSL/TLS for database connections
- [ ] Configure Redis password authentication

### Database
- [ ] Use external PostgreSQL service (not containerized)
- [ ] Configure database backups
- [ ] Set up connection pooling based on expected load
- [ ] Monitor database performance

### Redis
- [ ] Use external Redis service for production
- [ ] Enable Redis authentication
- [ ] Configure Redis persistence (RDB/AOF)
- [ ] Monitor Redis memory usage

### Monitoring
- [ ] Set up Prometheus + Grafana
- [ ] Configure log aggregation
- [ ] Set up health check alerts
- [ ] Monitor API response times

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Intent Engine API                         │
│  FastAPI Application (Port 8000)                             │
│  - CORS Middleware (configurable origins)                    │
│  - PostgreSQL Connection Pool                                │
│  - Redis Cache Client                                        │
└───────────────┬─────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┬──────────────┐
    │           │           │              │
    ▼           ▼           ▼              ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐
│Postgres│  │ Redis  │  │SearXNG │  │  Client  │
│ :5432  │  │ :6379  │  │ :8080  │  │(Frontend)│
└────────┘  └────────┘  └────────┘  └──────────┘
```

---

## Troubleshooting

### PostgreSQL Connection Issues
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Verify connection
docker-compose exec postgres psql -U intent_user -d intent_engine -c "SELECT 1"
```

### Redis Connection Issues
```bash
# Check Redis logs
docker-compose logs redis

# Test Redis connection
docker-compose exec redis redis-cli ping
```

### CORS Issues
```bash
# Check CORS configuration in logs
docker-compose logs intent-engine-api | grep CORS

# Test with curl
curl -H "Origin: http://localhost:3000" -v http://localhost:8000/
```

---

## Performance Tuning

### Database Pool Settings
Adjust based on expected concurrent connections:
- **Low traffic** (< 100 req/s): `POOL_SIZE=5`, `MAX_OVERFLOW=10`
- **Medium traffic** (100-500 req/s): `POOL_SIZE=10`, `MAX_OVERFLOW=20`
- **High traffic** (> 500 req/s): `POOL_SIZE=20`, `MAX_OVERFLOW=40`

### Redis Settings
- Enable Redis persistence for caching across restarts
- Configure maxmemory policy: `maxmemory-policy allkeys-lru`
- Monitor hit rate and adjust TTL accordingly

---

## Migration from SQLite

If you have existing SQLite data:

1. Export SQLite data:
```bash
sqlite3 intent_engine.db ".dump" > backup.sql
```

2. Modify for PostgreSQL syntax (sequences, types)

3. Import to PostgreSQL:
```bash
psql -U intent_user -d intent_engine -f backup.sql
```

4. Or use a migration tool like `pgloader`

---

## Next Steps

1. **Update CORS origins** for your production domains
2. **Change default passwords** in `.env` and `docker-compose.yml`
3. **Set up monitoring** for database and cache performance
4. **Configure backups** for PostgreSQL data
5. **Test failover** scenarios
6. **Review security** settings for production deployment
