"""
app/logging_cfg.py
------------------
Structured JSON logging setup using structlog.

Design decisions:
- Every log line is a JSON object — grep-able, jq-parseable, and ingestible
  by any log aggregator (Datadog, Loki, CloudWatch) without a parser rule.
- stdlib logging is bridged into structlog so third-party libraries
  (uvicorn, transformers) emit JSON too, not interleaved plaintext.
- In development (LOG_FORMAT=pretty) logs are rendered as colourised
  key=value pairs for readability. In production (default) they are JSON.
- request_id is bound to a context-local logger at the start of each
  request so every log line within that request carries it automatically,
  without passing the logger around manually.

Usage:
    from app.logging_cfg import configure_logging, get_logger, bind_request_id

    # Call once at startup (in main.py lifespan):
    configure_logging()

    # In a request handler:
    log = bind_request_id(request_id=str(req.request_id))
    log.info("predict.start", text_length=len(text))
    log.info("predict.done", label=result.label, confidence=result.confidence)
"""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")  # "json" | "pretty"


def configure_logging() -> None:
    """
    Wire structlog and stdlib logging together.
    Call this exactly once, from the FastAPI lifespan startup hook.
    """

    # --- Shared processors applied to every log record ---
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,  # picks up bound context (request_id, etc.)
        structlog.stdlib.add_logger_name,  # adds "logger" key
        structlog.stdlib.add_log_level,  # adds "level" key
        structlog.processors.TimeStamper(fmt="iso"),  # adds "timestamp" in ISO-8601
        structlog.processors.StackInfoRenderer(),  # renders stack_info if present
    ]

    if LOG_FORMAT == "pretty":
        # Human-readable for local dev — colourised key=value
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Machine-readable for production — newline-delimited JSON
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            # ExceptionRenderer must come before the final renderer
            structlog.processors.ExceptionRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging → structlog so uvicorn/transformers logs are JSON too
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ExtraAdder(),
            structlog.processors.ExceptionRenderer(),
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]  # replace any default handlers
    root_logger.setLevel(LOG_LEVEL)

    # Quieten noisy third-party loggers that don't add value in demos
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Per-request context binding
# ---------------------------------------------------------------------------


def bind_request_id(request_id: str, **extra: Any) -> structlog.stdlib.BoundLogger:
    """
    Bind request_id (and any extra fields) into the current context-local
    logger. Every subsequent log call in this request automatically includes
    these fields without any manual passing.

    Args:
        request_id: String UUID for the current request.
        **extra:    Any additional fields to bind (e.g. endpoint="/predict").

    Returns:
        A BoundLogger with the fields already set.

    Example:
        log = bind_request_id(request_id="abc-123", endpoint="/predict")
        log.info("request.received", text_length=42)
        # → {"timestamp": "...", "level": "info", "event": "request.received",
        #    "request_id": "abc-123", "endpoint": "/predict", "text_length": 42}
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, **extra)
    return get_logger()


def get_logger(name: str = "app") -> structlog.stdlib.BoundLogger:
    """
    Return a structlog logger. Use this instead of logging.getLogger()
    so all loggers go through the JSON pipeline.
    """
    return structlog.get_logger(name)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Convenience: log a complete prediction event in one call
# ---------------------------------------------------------------------------


def log_prediction(
    *,
    request_id: str,
    input_length: int,
    input_preview: str,
    label: str,
    confidence: float,
    inference_ms: float,
    drift_flagged: bool,
    model_version: str,
) -> None:
    """
    Emit one structured log line capturing the full context of a /predict call.

    Keeping this in logging_cfg.py (not main.py) means the log schema is
    defined in one place — easy to audit for PII or schema drift.

    Note: input_preview should be pre-truncated to ≤80 chars before passing in.
    """
    log = get_logger()
    log.info(
        "predict.complete",
        request_id=request_id,
        input_length=input_length,
        input_preview=input_preview,
        label=label,
        confidence=confidence,
        inference_ms=inference_ms,
        drift_flagged=drift_flagged,
        model_version=model_version,
    )
