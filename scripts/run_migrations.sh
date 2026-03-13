#!/bin/bash
# Run database migrations on startup

echo "Running database migrations..."

# Wait for PostgreSQL to be ready
until pg_isready -h postgres -U intent_user -d intent_engine; do
  echo "Waiting for PostgreSQL..."
  sleep 2
done

# First, create all base tables using standalone script to avoid SQLAlchemy metadata caching issues
echo "Creating base tables with standalone script..."
cd /app && python init_db_standalone.py

# Run all migration files in order (these add additional indexes/columns)
for migration in /app/migrations/*.sql; do
  if [ -f "$migration" ]; then
    echo "Running migration: $migration"
    psql -h postgres -U intent_user -d intent_engine -f "$migration" || true
  fi
done

echo "Migrations complete!"
