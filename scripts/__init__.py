"""Utility scripts for Intent Engine maintenance and development.

This package contains standalone scripts for various development and maintenance tasks:

Database Management:
    - init_db_standalone.py: Initialize database with all tables
    - init_db_direct.py: Direct database initialization
    - init_sample_data.py: Initialize database with sample data
    - seed_data.py: Seed database with production data
    - seed_sample_data.py: Seed database with sample data
    - reset_and_seed.py: Reset database and seed with data
    - clear_db.py: Clear all database tables
    - check_db_schema.py: Check database schema
    - check_db_status.py: Check database connection status

Migrations:
    - run_migrations.sh: Run SQL migrations

Development Tools:
    - bump_version.py: Bump project version
    - commit-gen.py: Generate commit messages
    - commit.py: Commit helper script
    - autopush.py: Auto-commit and push script
    - setup_git_hooks.py: Setup Git hooks
    - install_hooks.py: Install pre-commit hooks

Testing & Benchmarking:
    - benchmark.py: Run benchmarks
    - stress_test_all.py: Run stress tests
    - test_api_comprehensive.sh: Comprehensive API tests
    - test_api_docker.sh: API tests for Docker

Utilities:
    - check_model.py: Check model status
    - read_seed.py: Read seed data
    - verify_api_routes.py: Verify API routes
"""

__all__ = [
    # Database
    "init_db_standalone",
    "init_db_direct",
    "init_sample_data",
    "seed_data",
    "seed_sample_data",
    "reset_and_seed",
    "clear_db",
    "check_db_schema",
    "check_db_status",
    # Migrations
    "run_migrations",
    # Development
    "bump_version",
    "commit_gen",
    "commit",
    "autopush",
    "setup_git_hooks",
    "install_hooks",
    # Testing
    "benchmark",
    "stress_test_all",
    # Utilities
    "check_model",
    "read_seed",
    "verify_api_routes",
]
