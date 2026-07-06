# ────────────────────────────────────────────────────────────────
# Heart Disease Prediction API — Production Dockerfile
# Builds a minimal, secure container for serving the ML model.
# ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="MLOps Assignment - BITS Pilani AIMLCZG523"
LABEL version="1.0.0"
LABEL description="Heart Disease Prediction FastAPI Service"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_ALLOW_FILE_STORE=true \
    PORT=8000

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user and logs directory with correct permissions
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appgroup /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI with uvicorn
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
