"""
app/main.py
-----------
FastAPI application. Wires together model, logging, and drift monitoring.

Endpoints:
    POST /predict        — sentiment classification
    GET  /health         — liveness + component checks
    GET  /drift-report   — input distribution statistics
    GET  /               — redirect to /docs

Design decisions:
- Lifespan context manager (not deprecated @app.on_event) handles startup
  and shutdown cleanly, including model warm-up and logging initialisation.
- DriftMonitor is instantiated once and stored on app.state so it is shared
  across all requests without a global variable.
- process_time middleware adds X-Process-Time-Ms to every response header —
  useful for latency debugging without opening a log file.
- HTTPException handlers return the same JSON shape as normal responses so
  clients never have to branch on error format.
- /predict is intentionally synchronous (def, not async def) because
  HuggingFace pipeline() releases the GIL during tokenisation but the
  torch forward pass does not. Running it in a thread pool (FastAPI's
  default for sync handlers) keeps the event loop free.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from app.drift import DriftMonitor
from app.logging_cfg import bind_request_id, configure_logging, get_logger, log_prediction
from app.model import MODEL_VERSION, PredictionResult, get_model, predict
from app.schemas import (
    DriftReportResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

# ---------------------------------------------------------------------------
# Module-level logger (used outside request context, e.g. startup)
# ---------------------------------------------------------------------------
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Process start time — used by /health to report uptime
# ---------------------------------------------------------------------------
_PROCESS_START: float = time.monotonic()


# ---------------------------------------------------------------------------
# Lifespan — startup & shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: configure logging, pre-load model, initialise drift monitor.
    Shutdown: nothing to clean up (SQLite closes automatically).
    """
    # Logging must be configured before anything else logs.
    configure_logging()
    log.info("app.startup", model_version=MODEL_VERSION)

    # Pre-load model — blocks until weights are downloaded + warmed up.
    # This ensures the first /predict request has no cold-start penalty.
    get_model()
    log.info("app.model_ready", model_version=MODEL_VERSION)

    # Attach drift monitor to app.state for shared access across requests.
    app.state.drift_monitor = DriftMonitor()
    log.info("app.drift_monitor_ready")

    yield  # Application runs here

    log.info("app.shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sentiment Classifier API",
    description=(
        "Production-ready sentiment classification service built on "
        "distilbert-base-uncased-finetuned-sst-2-english. "
        "Includes structured logging, input drift monitoring, and a CI retrain gate."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Middleware — add X-Process-Time-Ms header to every response
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1_000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Exception handlers — uniform error shape
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled_exception", path=str(request.url.path), exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error. Check logs for details.",
            "status_code": 500,
            "path": str(request.url.path),
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect bare root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify sentiment",
    tags=["Inference"],
)
def predict_endpoint(body: PredictRequest, request: Request) -> PredictResponse:
    """
    Classify the sentiment of the provided text.

    Returns POSITIVE or NEGATIVE with a confidence score.
    Every call is logged (structured JSON) and recorded in the drift monitor.

    - **text**: The text to classify (1–5,000 characters).
    - **request_id**: Optional client-supplied UUID for end-to-end tracing.
    """
    # Resolve or generate request_id
    req_id: uuid.UUID = body.request_id or uuid.uuid4()
    req_id_str = str(req_id)

    # Bind request_id into the context logger — all log calls below carry it.
    request_log = bind_request_id(req_id_str, endpoint="/predict")
    request_log.info("predict.start", text_length=len(body.text))

    # --- Inference ---
    try:
        result: PredictionResult = predict(body.text)
    except ValueError as exc:
        request_log.warning("predict.invalid_input", error=str(exc))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        request_log.error("predict.model_error", error=str(exc), exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference failed. Please retry.",
        )

    # --- Drift recording ---
    drift_monitor: DriftMonitor = request.app.state.drift_monitor
    try:
        drift_flagged: bool = drift_monitor.record(body.text)
    except Exception as exc:
        # Drift monitoring failure must never break the prediction response.
        request_log.warning("drift.record_failed", error=str(exc))
        drift_flagged = False

    # --- Structured log ---
    text_preview = (body.text[:80] + "…") if len(body.text) > 80 else body.text
    log_prediction(
        request_id=req_id_str,
        input_length=len(body.text),
        input_preview=text_preview,
        label=result.label,
        confidence=result.confidence,
        inference_ms=result.inference_ms,
        drift_flagged=drift_flagged,
        model_version=result.model_version,
    )

    return PredictResponse(
        request_id=req_id,
        timestamp=datetime.now(UTC),
        text_preview=text_preview,
        label=result.label,
        confidence=result.confidence,
        model_version=result.model_version,
        inference_ms=result.inference_ms,
        drift_flagged=drift_flagged,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Operations"],
)
async def health_endpoint(request: Request) -> HealthResponse:
    """
    Liveness and readiness check.

    Returns 'ok' when the model is loaded and the drift DB is reachable.
    Returns 'degraded' (HTTP 200) when a non-critical component is down —
    the service can still serve predictions but monitoring is impaired.

    Use this as your uptime monitor's probe URL.
    """
    model_loaded = True
    db_ok = True

    # Check model is in the lru_cache (non-blocking — no inference)
    try:
        get_model()
    except Exception:
        model_loaded = False

    # Check drift DB is reachable
    try:
        request.app.state.drift_monitor.report()
    except Exception:
        db_ok = False

    all_ok = model_loaded and db_ok
    uptime = time.monotonic() - _PROCESS_START

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        model_loaded=model_loaded,
        model_version=MODEL_VERSION,
        uptime_seconds=round(uptime, 1),
        checks={"model": model_loaded, "drift_db": db_ok},
    )


@app.get(
    "/drift-report",
    response_model=DriftReportResponse,
    summary="Input distribution drift report",
    tags=["Operations"],
)
async def drift_report_endpoint(request: Request) -> DriftReportResponse:
    """
    Returns drift statistics for the last N requests (default N=100).

    Three signals are tracked against the SST-2 training baseline:
    - **text_length**: character count distribution.
    - **oov_rate**: fraction of tokens outside the training vocabulary.
    - **non_ascii_rate**: fraction of non-ASCII characters.

    A z-score > 2.0 on any signal sets **drift_detected: true**.
    This does not affect predictions — it is an early-warning indicator.
    """
    monitor: DriftMonitor = request.app.state.drift_monitor
    try:
        data = monitor.report()
    except Exception as exc:
        log.error("drift.report_failed", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift monitor unavailable.",
        )
    return DriftReportResponse(**data)
