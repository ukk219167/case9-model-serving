"""
app/schemas.py
--------------
Pydantic v2 request and response models for all endpoints.

Design decisions:
- One file for all schemas so the surface area is easy to audit.
- Response models carry 'request_id' so a caller can correlate logs
  without needing to grep by text content (which may be PII).
- Field-level constraints (min_length, max_length) are enforced here
  so main.py stays clean and FastAPI auto-documents the limits in /docs.
- All response models inherit from BaseResponse to guarantee every
  response carries a timestamp and request_id.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class BaseResponse(BaseModel):
    """Every response carries a timestamp and the request_id for log correlation."""

    request_id: UUID = Field(description="UUID echoed from the request (or generated server-side).")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC time the response was generated.",
    )


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Payload for POST /predict."""

    text: str = Field(
        min_length=1,
        max_length=5_000,
        description="Text to classify. Truncated to 512 tokens internally.",
        examples=["I absolutely loved this product, it exceeded my expectations!"],
    )
    request_id: UUID | None = Field(
        default=None,
        description=(
            "Optional client-supplied UUID. "
            "If omitted, one is generated server-side. "
            "Useful for end-to-end tracing across services."
        ),
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain at least one non-whitespace character.")
        return v


class PredictResponse(BaseResponse):
    """Response from POST /predict."""

    text_preview: str = Field(
        description="First 80 characters of the input (for log readability, not full PII)."
    )
    label: Literal["POSITIVE", "NEGATIVE"] = Field(description="Predicted sentiment class.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in the predicted label (0–1).",
    )
    model_version: str = Field(
        description="Version tag of the model that produced this prediction."
    )
    inference_ms: float = Field(
        description="Wall-clock inference time in milliseconds (excludes network)."
    )
    drift_flagged: bool = Field(
        description=(
            "True if the drift monitor raised an alert on this request. "
            "Does not affect the prediction — informational only."
        )
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: Literal["ok", "degraded"] = Field(
        description="'ok' if all components are healthy, 'degraded' otherwise."
    )
    model_loaded: bool = Field(description="True if the model singleton is cached.")
    model_version: str = Field(description="Currently loaded model version tag.")
    uptime_seconds: float = Field(description="Seconds since the process started.")
    checks: dict[str, bool] = Field(
        description="Per-component health flags, e.g. {'model': True, 'db': True}."
    )


# ---------------------------------------------------------------------------
# /drift-report
# ---------------------------------------------------------------------------


class DriftStats(BaseModel):
    """Statistics for one drift signal over the rolling window."""

    baseline_mean: float = Field(description="Mean computed from training distribution.")
    baseline_std: float = Field(description="Std dev computed from training distribution.")
    window_mean: float = Field(description="Mean over the last N requests.")
    window_std: float = Field(description="Std dev over the last N requests.")
    z_score: float = Field(
        description="(window_mean - baseline_mean) / baseline_std. |z| > 2 triggers alert."
    )
    alert: bool = Field(description="True if |z_score| exceeds the alert threshold.")


class DriftReportResponse(BaseModel):
    """Response from GET /drift-report."""

    window_size: int = Field(description="Number of recent requests included in the window.")
    requests_seen: int = Field(description="Total requests processed since startup.")
    drift_detected: bool = Field(description="True if ANY individual signal has raised an alert.")
    signals: dict[str, DriftStats] = Field(
        description=("Per-signal statistics. Keys: 'text_length', 'oov_rate', 'non_ascii_rate'.")
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC time the report was generated.",
    )
