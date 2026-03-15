#!/bin/bash
set -e

echo "=== Intent Engine All-in-One Container Starting ==="

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to wait for a service
wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=30
    local attempt=1

    log "Waiting for $service_name..."
    while [ $attempt -le $max_attempts ]; do
        if eval "$check_command" > /dev/null 2>&1; then
            log "$service_name is ready!"
            return 0
        fi
        log "Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done

    log "ERROR: $service_name failed to start after $max_attempts attempts"
    return 1
}

# Initialize PostgreSQL if needed
init_postgres() {
    log "Initializing PostgreSQL..."
    
    # Check if database is already initialized
    if [ -f "$PGDATA/PG_VERSION" ]; then
        log "PostgreSQL data directory already exists, skipping initialization"
        # Fix permissions if needed
        chown -R postgres:postgres "$PGDATA" 2>/dev/null || true
        return 0
    fi

    # Initialize PostgreSQL database cluster as postgres user
    log "Creating PostgreSQL database cluster..."
    
    # Find initdb path
    INITDB_PATH=$(which initdb 2>/dev/null || find /usr -name initdb -type f 2>/dev/null | head -1)
    if [ -z "$INITDB_PATH" ]; then
        INITDB_PATH="/usr/lib/postgresql/*/bin/initdb"
        # Use glob to find the actual path
        for path in /usr/lib/postgresql/*/bin/initdb; do
            if [ -x "$path" ]; then
                INITDB_PATH="$path"
                break
            fi
        done
    fi
    
    log "Using initdb at: $INITDB_PATH"
    
    su postgres -c "$INITDB_PATH -D $PGDATA" || {
        log "ERROR: Failed to initialize PostgreSQL"
        return 1
    }

    # Configure PostgreSQL to listen on all interfaces
    log "Configuring PostgreSQL..."
    cat >> "$PGDATA/postgresql.conf" << EOF
listen_addresses = '*'
port = 5432
max_connections = 100
shared_buffers = 128MB
work_mem = 4MB
maintenance_work_mem = 64MB
effective_cache_size = 512MB
checkpoint_completion_target = 0.9
wal_buffers = 4MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
EOF

    # Configure pg_hba.conf for authentication (trust for local connections)
    cat > "$PGDATA/pg_hba.conf" << EOF
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
host    all             all             0.0.0.0/0               md5
EOF

    log "PostgreSQL initialization complete"
    return 0
}

# Start PostgreSQL
start_postgres() {
    log "Starting PostgreSQL..."
    
    # Ensure proper ownership
    chown -R postgres:postgres "$PGDATA" 2>/dev/null || true
    chown -R postgres:postgres /var/run/postgresql 2>/dev/null || true
    chown -R postgres:postgres /app/data 2>/dev/null || true
    
    # Find pg_ctl path
    PGCTL_PATH=$(which pg_ctl 2>/dev/null || find /usr -name pg_ctl -type f 2>/dev/null | head -1)
    if [ -z "$PGCTL_PATH" ]; then
        for path in /usr/lib/postgresql/*/bin/pg_ctl; do
            if [ -x "$path" ]; then
                PGCTL_PATH="$path"
                break
            fi
        done
    fi
    
    # Create log directory and file
    mkdir -p /app/data
    touch /app/data/postgresql.log
    chown postgres:postgres /app/data/postgresql.log
    
    # Start PostgreSQL in background as postgres user
    su postgres -c "$PGCTL_PATH -D $PGDATA -l /app/data/postgresql.log start" || {
        log "ERROR: Failed to start PostgreSQL"
        return 1
    }

    # Wait for PostgreSQL to be ready
    wait_for_service "PostgreSQL" "pg_isready -h 127.0.0.1 -U $POSTGRES_USER" || return 1

    # Create database and user if they don't exist
    log "Setting up database..."
    
    # Wait a bit more for PostgreSQL to be fully ready
    sleep 2
    
    # Find psql path
    PSQL_PATH=$(which psql 2>/dev/null || find /usr -name psql -type f 2>/dev/null | head -1)
    if [ -z "$PSQL_PATH" ]; then
        for path in /usr/lib/postgresql/*/bin/psql; do
            if [ -x "$path" ]; then
                PSQL_PATH="$path"
                break
            fi
        done
    fi
    
    # Use postgres superuser to create role and database
    # With trust auth, we can connect without password
    su postgres -c "$PSQL_PATH -h 127.0.0.1 -c \"DROP DATABASE IF EXISTS $POSTGRES_DB;\"" 2>/dev/null || true
    su postgres -c "$PSQL_PATH -h 127.0.0.1 -c \"DROP ROLE IF EXISTS $POSTGRES_USER;\"" 2>/dev/null || true
    sleep 1
    su postgres -c "$PSQL_PATH -h 127.0.0.1 -c \"CREATE ROLE $POSTGRES_USER LOGIN CREATEDB;\"" 2>/dev/null || true
    su postgres -c "$PSQL_PATH -h 127.0.0.1 -c \"CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER;\"" 2>/dev/null || true
    su postgres -c "$PSQL_PATH -h 127.0.0.1 -c \"GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;\"" 2>/dev/null || true
    sleep 1

    log "PostgreSQL started successfully"
    return 0
}

# Start Redis
start_redis() {
    log "Starting Redis..."
    
    # Create Redis data directory
    mkdir -p /app/data/redis
    chown -R redis:redis /app/data/redis 2>/dev/null || chown -R appuser:appuser /app/data/redis 2>/dev/null || true
    
    # Start Redis in background
    log "Starting Redis server..."
    redis-server --daemonize yes \
        --port 6379 \
        --bind 127.0.0.1 \
        --dir /app/data/redis \
        --save 30 1 \
        --loglevel warning \
        --maxmemory 512mb \
        --maxmemory-policy allkeys-lru || {
        log "ERROR: Failed to start Redis"
        return 1
    }

    # Wait for Redis to be ready
    sleep 2
    wait_for_service "Redis" "redis-cli -h 127.0.0.1 -p 6379 ping" || return 1

    log "Redis started successfully"
    return 0
}

# Start SearXNG
start_searxng() {
    log "Starting SearXNG..."
    
    # Create SearXNG directories
    mkdir -p /app/data/searxng
    
    # Generate secret key if not set
    if [ -z "$SEARXNG_SECRET_KEY" ]; then
        export SEARXNG_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || echo "default-secret-key-$(date +%s)")
        log "Generated SearXNG secret key"
    fi

    # Set up SearXNG environment
    export SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
    
    # Start SearXNG in background
    cd /app
    nohup python3 -m searx.webapp --port 8080 --bind 127.0.0.1 > /app/data/searxng.log 2>&1 &
    
    SEARXNG_PID=$!
    log "SearXNG started with PID: $SEARXNG_PID"

    # Wait for SearXNG to be ready (give it more time)
    sleep 10
    local attempt=1
    while [ $attempt -le 15 ]; do
        if curl -f http://127.0.0.1:8080/healthz > /dev/null 2>&1; then
            log "SearXNG is ready!"
            return 0
        fi
        log "Waiting for SearXNG... attempt $attempt/15"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log "WARNING: SearXNG may not be fully ready yet, continuing..."
    return 0
}

# Start nginx
start_nginx() {
    log "Starting nginx..."
    
    # Start nginx (needs root or proper capabilities)
    if [ "$(id -u)" = "0" ]; then
        nginx || {
            log "WARNING: nginx failed to start"
            return 1
        }
    else
        # Try to start nginx with sudo or directly
        sudo nginx 2>/dev/null || nginx -g "daemon off;" &
        sleep 2
    fi

    log "nginx started"
    return 0
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    cd /app
    
    # Wait for database to be ready
    wait_for_service "PostgreSQL" "pg_isready -h 127.0.0.1 -U $POSTGRES_USER -d $POSTGRES_DB" || return 1

    # Run Python migrations
    python3 /app/scripts/init_db_standalone.py 2>&1 || {
        log "WARNING: init_db_standalone.py failed or doesn't exist"
    }

    # Run SQL migrations if they exist
    for migration_file in /app/migrations/*.sql; do
        if [ -f "$migration_file" ]; then
            log "Applying migration: $migration_file"
            PGPASSWORD="$POSTGRES_PASSWORD" psql \
                -h 127.0.0.1 \
                -U "$POSTGRES_USER" \
                -d "$POSTGRES_DB" \
                -f "$migration_file" 2>&1 || {
                log "WARNING: Migration $migration_file had issues (may already be applied)"
            }
        fi
    done

    log "Database migrations complete"
    return 0
}

# Start the main API
start_api() {
    log "Starting Intent Engine API..."
    
    cd /app
    
    # Set environment variables
    export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:5432/${POSTGRES_DB}"
    export REDIS_URL="redis://127.0.0.1:6379/0"
    export SEARXNG_BASE_URL="http://127.0.0.1:8080"
    export ENVIRONMENT="production"
    
    # Start uvicorn with multiple workers
    nohup uvicorn main_api:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers ${WORKERS:-2} > /app/data/api.log 2>&1 &
    
    API_PID=$!
    log "API started with PID: $API_PID"

    # Wait for API to be ready
    wait_for_service "Intent Engine API" "curl -f http://127.0.0.1:8000/health" || return 1

    log "Intent Engine API started successfully"
    return 0
}

# Main execution
main() {
    log "Starting all services..."
    
    # Initialize and start PostgreSQL
    init_postgres || exit 1
    start_postgres || exit 1
    
    # Start Redis
    start_redis || exit 1
    
    # Start SearXNG (optional, continue if it fails)
    start_searxng || log "WARNING: SearXNG failed to start, continuing without it"
    
    # Run database migrations
    run_migrations || log "WARNING: Some migrations failed, continuing..."
    
    # Start the main API
    start_api || exit 1
    
    # Start nginx
    start_nginx || log "WARNING: nginx failed to start"
    
    log "=== All services started successfully ==="
    log "API available at: http://localhost:80"
    log "PostgreSQL: localhost:5432"
    log "Redis: localhost:6379"
    log "SearXNG: localhost:8080"
    
    # Keep container running by waiting on all background processes
    # Use a trap to handle shutdown gracefully
    trap 'log "Shutting down..."; kill $(jobs -p) 2>/dev/null; exit 0' SIGTERM SIGINT
    
    # Wait forever (or until container is stopped)
    while true; do
        sleep 60
        # Check if API is still running, restart if not
        if ! curl -f http://127.0.0.1:8000/health > /dev/null 2>&1; then
            log "WARNING: API health check failed, attempting restart..."
            pkill -f "uvicorn main_api:app" || true
            sleep 5
            cd /app
            nohup uvicorn main_api:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2} > /app/data/api.log 2>&1 &
            log "API restart initiated"
        fi
    done
}

# Run main function
main "$@"
