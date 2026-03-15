# Intent Engine - Production Dockerfile
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKERS=2

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install numpy first to avoid compilation issues
RUN pip install --upgrade pip && \
    pip install numpy==1.24.3

# Install PyTorch CPU version separately (large download, cache this layer)
RUN pip install --no-cache-dir torch==2.1.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install Python dependencies (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make migration script executable
RUN chmod +x /app/scripts/run_migrations.sh 2>/dev/null || true

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/data && chown appuser:appuser /app/data
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn main_api:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2}"]
