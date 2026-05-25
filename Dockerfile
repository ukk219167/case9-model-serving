# =============================================================================
# Multi-stage Dockerfile
# =============================================================================
# Stage 1 (builder) — install Python deps into a clean venv.
# Stage 2 (runtime) — copy only the venv + app code; no build tools shipped.
#
# Design decisions:
# - CPU-only torch index URL keeps the image ~800 MB smaller than the default.
# - Model weights are downloaded at runtime (first request / warmup), NOT baked
#   into the image. Baking weights makes the image ~270 MB heavier and means
#   every code change triggers a full weight re-download in CI. Instead, the
#   HuggingFace cache dir is mounted as a volume on the host so weights persist
#   across container restarts.
# - Non-root user (appuser) follows least-privilege principle.
# - PYTHONDONTWRITEBYTECODE + PYTHONUNBUFFERED are standard prod settings.
# - Healthcheck uses /health so Docker / Render / Fly.io know when the
#   container is ready (model loaded, drift DB reachable).
# =============================================================================

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (needed for some C-extension packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first — maximises Docker layer cache hits.
# The venv layer only rebuilds when requirements.txt changes.
COPY requirements.txt .

# Create isolated venv and install into it.
# CPU-only PyTorch index avoids pulling the massive CUDA wheel (~2.5 GB).
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip --quiet \
    && /opt/venv/bin/pip install \
        --quiet \
        --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Standard production Python settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# App configuration — override any of these at runtime via -e flags or
# the host's secret store (never bake secrets into the image).
ENV MODEL_NAME="distilbert-base-uncased-finetuned-sst-2-english" \
    MODEL_VERSION="distilbert-sst2-v1" \
    LOG_LEVEL="INFO" \
    LOG_FORMAT="json" \
    DRIFT_DB_PATH="/tmp/drift.db" \
    DRIFT_WINDOW_SIZE="100" \
    DRIFT_ALERT_THRESHOLD="2.0" \
    # HuggingFace cache — mount a volume here to persist model weights.
    HF_HOME="/tmp/hf_cache"

WORKDIR /app

# Copy the venv from the builder stage (no gcc / g++ in the final image)
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY app/ ./app/

# Create a non-root user and hand ownership over
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app /tmp
USER appuser

# Expose the port uvicorn will listen on
EXPOSE 8000

# Healthcheck — Docker will mark the container unhealthy if /health
# returns non-2xx for 3 consecutive checks.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Start the server.
# --workers 1 : safe with lru_cache singleton; scale horizontally instead.
# --host 0.0.0.0: required to receive traffic inside the container.
# --port 8000: matches EXPOSE above.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "warning"]
