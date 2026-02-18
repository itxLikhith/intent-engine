#!/bin/bash
# Run database migrations on startup

echo "Running database migrations..."

# Wait for PostgreSQL to be ready
until pg_isready -h postgres -U intent_user -d intent_engine; do
  echo "Waiting for PostgreSQL..."
  sleep 2
done

# Run all migration files in order
for migration in /app/migrations/*.sql; do
  if [ -f "$migration" ]; then
    echo "Running migration: $migration"
    psql -h postgres -U intent_user -d intent_engine -f "$migration"
  fi
done

echo "Migrations complete!"
