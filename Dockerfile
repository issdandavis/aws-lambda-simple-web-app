# SCBE-AETHERMOORE Python SDK
# Production-Ready Docker Image
#
# Build: docker build -t scbe-aethermoore:3.0.0 .
# Run:   docker run -p 8000:8000 scbe-aethermoore:3.0.0

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source and install package
COPY symphonic_cipher/ ./symphonic_cipher/
COPY scbe_production/ ./scbe_production/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# ==============================================================================
# Stage 2: Production Runtime
# ==============================================================================
FROM python:3.11-slim AS production

# Security: Run as non-root user
RUN groupadd -r scbe && useradd -r -g scbe scbe

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=scbe:scbe symphonic_cipher/ ./symphonic_cipher/
COPY --chown=scbe:scbe scbe_production/ ./scbe_production/
COPY --chown=scbe:scbe pyproject.toml .

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SCBE_ENVIRONMENT=production \
    SCBE_LOG_LEVEL=INFO \
    SCBE_LOG_FORMAT=json \
    SCBE_AUDIT_ENABLED=true \
    SCBE_PQC_KEM_LEVEL=768 \
    SCBE_PQC_DSA_LEVEL=65 \
    SCBE_PQC_HYBRID=true

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from scbe_production.service import SCBEProductionService; s = SCBEProductionService(); print(s.health_check())" || exit 1

# Switch to non-root user
USER scbe

# Default command: Run API server
CMD ["python", "-m", "uvicorn", "scbe_production.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Stage 3: Development (optional)
# ==============================================================================
FROM production AS development

USER root

# Install dev dependencies
RUN pip install --no-cache-dir pytest pytest-cov black mypy flake8

# Copy test files
COPY --chown=scbe:scbe tests/ ./tests/

USER scbe

# Development command: Run tests
CMD ["pytest", "-v", "--cov=symphonic_cipher", "--cov=scbe_production"]
