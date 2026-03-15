#!/usr/bin/env python
"""
Intent Engine - Database Initialization Script

This script initializes the database with all required tables.
It's designed to be run during container startup or manual deployment.

Usage:
    python scripts/init_db.py

Environment Variables:
    DATABASE_URL: Database connection URL (default: sqlite:///./intent_engine.db)
"""

import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get database URL from environment or use default."""
    return os.getenv("DATABASE_URL", "sqlite:///./intent_engine.db")


def initialize_database() -> bool:
    """
    Initialize database tables.

    Returns:
        True if successful, False otherwise
    """
    database_url = get_database_url()
    logger.info(f"Initializing database: {database_url.split('@')[0] if '@' in database_url else database_url}")

    try:
        # Import database module
        from database import Base, engine

        # Create all tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("Database tables created successfully")

        # Verify tables were created
        from sqlalchemy import inspect, text

        with engine.connect() as connection:
            inspector = inspect(connection)
            tables = inspector.get_table_names()
            logger.info(f"Created {len(tables)} tables: {', '.join(tables)}")

            # Test database connectivity
            connection.execute(text("SELECT 1"))
            logger.info("Database connectivity verified")

        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def run_migrations() -> bool:
    """
    Run any pending SQL migrations.

    Returns:
        True if successful, False otherwise
    """
    import subprocess

    migrations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "migrations")

    if not os.path.exists(migrations_dir):
        logger.warning(f"Migrations directory not found: {migrations_dir}")
        return True

    migration_files = sorted([f for f in os.listdir(migrations_dir) if f.endswith(".sql")])

    if not migration_files:
        logger.info("No SQL migrations found")
        return True

    database_url = get_database_url()

    # Only run migrations for PostgreSQL
    if not database_url.startswith("postgresql"):
        logger.info("Skipping SQL migrations (not PostgreSQL)")
        return True

    logger.info(f"Running {len(migration_files)} SQL migrations...")

    for migration_file in migration_files:
        migration_path = os.path.join(migrations_dir, migration_file)
        logger.info(f"Running migration: {migration_file}")

        try:
            # Use psql to run migration
            result = subprocess.run(
                ["psql", database_url, "-f", migration_path], capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                logger.warning(f"Migration {migration_file} had issues: {result.stderr}")
            else:
                logger.info(f"Migration {migration_file} completed")

        except subprocess.TimeoutExpired:
            logger.error(f"Migration {migration_file} timed out")
            return False
        except Exception as e:
            logger.warning(f"Migration {migration_file} failed: {e}")
            # Continue with next migration

    return True


def create_sample_data() -> bool:
    """
    Create sample data for testing (optional).

    Returns:
        True if successful, False otherwise
    """
    if os.getenv("SKIP_SAMPLE_DATA", "true").lower() == "true":
        logger.info("Skipping sample data creation")
        return True

    try:
        logger.info("Creating sample data...")
        from scripts.seed_data import seed_sample_data

        seed_sample_data()
        logger.info("Sample data created successfully")
        return True
    except ImportError:
        logger.warning("Sample data script not found, skipping")
        return True
    except Exception as e:
        logger.error(f"Sample data creation failed: {e}")
        return False


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Intent Engine - Database Initialization")
    logger.info("=" * 60)

    start_time = datetime.now()

    # Step 1: Initialize database tables
    if not initialize_database():
        logger.error("Database initialization failed")
        sys.exit(1)

    # Step 2: Run SQL migrations
    if not run_migrations():
        logger.warning("Some migrations failed, continuing...")

    # Step 3: Create sample data (optional)
    if not create_sample_data():
        logger.warning("Sample data creation failed, continuing...")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info(f"Database initialization completed in {elapsed:.2f}s")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
