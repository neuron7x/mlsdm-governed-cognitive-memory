# Production Dockerfile for MLSDM Governed Cognitive Memory
# ===========================================================
# Multi-stage build for optimized image size and security

# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.12-slim

# Create non-root user for security
RUN useradd -m -u 1000 mlsdm && \
    mkdir -p /app && \
    chown -R mlsdm:mlsdm /app

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/mlsdm/.local

# Copy application code
COPY --chown=mlsdm:mlsdm . .

# Switch to non-root user
USER mlsdm

# Make Python packages available
ENV PATH=/home/mlsdm/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose API port
EXPOSE 8000

# Default configuration (override with environment variables)
ENV CONFIG_PATH=config/production.yaml \
    LOG_LEVEL=INFO \
    JSON_LOGGING=true \
    MLSDM_RUNTIME_MODE=cloud-prod \
    MLSDM_SECURE_MODE=1

# Run API server using cloud entrypoint
CMD ["python", "-m", "mlsdm.entrypoints.cloud"]
