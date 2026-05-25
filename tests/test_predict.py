"""
tests/test_predict.py
---------------------
Unit + integration tests for the /predict and /health endpoints.

Strategy:
- The real HuggingFace model is MOCKED in all tests. This keeps the suite
  fast (<5 seconds) and runnable on CI without downloading weights.
- One smoke test (marked with @pytest.mark.slow) runs against the real model
  and is skipped in CI unless RUN_SLOW_TESTS=1 is set.
- Tests use httpx.AsyncClient with FastAPI's ASGI transport — no real server
  needed, no port binding, no flakiness from network.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.model import PredictionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_RESULT = PredictionResult(
    label="POSITIVE",
    confidence=0.9823,
    model_version="distilbert-sst2-v1",
    inference_ms=42.1,
)


@pytest_asyncio.fixture
async def client():
    """
    ASGI test client with a fresh DriftMonitor backed by an in-memory DB.
    Model is NOT loaded — patched at the app.model level.
    """
    from app.drift import DriftMonitor

    # Use an in-memory SQLite DB so tests never touch the filesystem.
    test_monitor = DriftMonitor(db_path=":memory:")
    app.state.drift_monitor = test_monitor

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_positive(client: AsyncClient):
    """Valid text returns POSITIVE label with correct response shape."""
    with patch("app.main.predict", return_value=MOCK_RESULT):
        response = await client.post("/predict", json={"text": "This film was fantastic!"})

    assert response.status_code == 200
    data = response.json()

    assert data["label"] == "POSITIVE"
    assert 0.0 <= data["confidence"] <= 1.0
    assert "request_id" in data
    assert "timestamp" in data
    assert "inference_ms" in data
    assert "drift_flagged" in data
    assert data["model_version"] == "distilbert-sst2-v1"


@pytest.mark.asyncio
async def test_predict_negative(client: AsyncClient):
    """Negative sentiment prediction is correctly returned."""
    neg_result = PredictionResult(
        label="NEGATIVE",
        confidence=0.9541,
        model_version="distilbert-sst2-v1",
        inference_ms=38.7,
    )
    with patch("app.main.predict", return_value=neg_result):
        response = await client.post("/predict", json={"text": "Terrible, I hated it."})

    assert response.status_code == 200
    assert response.json()["label"] == "NEGATIVE"


@pytest.mark.asyncio
async def test_predict_echoes_client_request_id(client: AsyncClient):
    """Client-supplied request_id is echoed back unchanged."""
    client_id = str(uuid.uuid4())
    with patch("app.main.predict", return_value=MOCK_RESULT):
        response = await client.post(
            "/predict",
            json={"text": "Great movie.", "request_id": client_id},
        )

    assert response.status_code == 200
    assert response.json()["request_id"] == client_id


@pytest.mark.asyncio
async def test_predict_generates_request_id_when_omitted(client: AsyncClient):
    """Server generates a UUID when the client doesn't supply one."""
    with patch("app.main.predict", return_value=MOCK_RESULT):
        response = await client.post("/predict", json={"text": "Good enough."})

    data = response.json()
    assert "request_id" in data
    # Should be a valid UUID
    uuid.UUID(data["request_id"])  # raises ValueError if invalid


@pytest.mark.asyncio
async def test_predict_text_preview_truncated(client: AsyncClient):
    """text_preview in the response is capped at 80 characters."""
    long_text = "A" * 200
    with patch("app.main.predict", return_value=MOCK_RESULT):
        response = await client.post("/predict", json={"text": long_text})

    preview = response.json()["text_preview"]
    assert len(preview) <= 82  # 80 chars + "…" ellipsis


@pytest.mark.asyncio
async def test_predict_process_time_header(client: AsyncClient):
    """Every response carries the X-Process-Time-Ms header."""
    with patch("app.main.predict", return_value=MOCK_RESULT):
        response = await client.post("/predict", json={"text": "Nice film."})

    assert "x-process-time-ms" in response.headers


# ---------------------------------------------------------------------------
# /predict — validation errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_empty_text_rejected(client: AsyncClient):
    """Empty string is rejected with 422."""
    response = await client.post("/predict", json={"text": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_whitespace_only_rejected(client: AsyncClient):
    """Whitespace-only text is rejected with 422."""
    response = await client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_missing_text_rejected(client: AsyncClient):
    """Missing 'text' field is rejected with 422."""
    response = await client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_text_too_long_rejected(client: AsyncClient):
    """Text exceeding 5000 characters is rejected with 422."""
    response = await client.post("/predict", json={"text": "x" * 5001})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /predict — model failure handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_model_error_returns_503(client: AsyncClient):
    """If the model raises an unexpected exception, the API returns 503."""
    with patch("app.main.predict", side_effect=RuntimeError("GPU OOM")):
        response = await client.post("/predict", json={"text": "Test."})

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_predict_drift_failure_does_not_break_response(client: AsyncClient):
    """If drift recording fails, the prediction response is still returned."""
    with patch("app.main.predict", return_value=MOCK_RESULT):
        with patch.object(
            app.state.drift_monitor, "record", side_effect=Exception("DB full")
        ):
            response = await client.post("/predict", json={"text": "Fine film."})

    # Should still return 200 — drift failure is non-fatal
    assert response.status_code == 200
    assert response.json()["drift_flagged"] is False


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient):
    """Health endpoint returns status=ok when model is loaded."""
    with patch("app.main.get_model", return_value=MagicMock()):
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "model_loaded" in data
    assert "uptime_seconds" in data
    assert "checks" in data


@pytest.mark.asyncio
async def test_health_uptime_is_positive(client: AsyncClient):
    """Uptime reported by /health is a positive number."""
    with patch("app.main.get_model", return_value=MagicMock()):
        response = await client.get("/health")

    assert response.json()["uptime_seconds"] > 0


# ---------------------------------------------------------------------------
# Slow tests — run only with RUN_SLOW_TESTS=1 (skipped in CI by default)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1",
    reason="Skipped in CI — set RUN_SLOW_TESTS=1 to run against real model.",
)
@pytest.mark.asyncio
async def test_predict_real_model_positive(client: AsyncClient):
    """Smoke test: real model correctly classifies an obvious positive review."""
    response = await client.post(
        "/predict",
        json={"text": "Absolutely wonderful film, one of the best ever made!"},
    )
    assert response.status_code == 200
    assert response.json()["label"] == "POSITIVE"
    assert response.json()["confidence"] > 0.9


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1",
    reason="Skipped in CI — set RUN_SLOW_TESTS=1 to run against real model.",
)
@pytest.mark.asyncio
async def test_predict_real_model_negative(client: AsyncClient):
    """Smoke test: real model correctly classifies an obvious negative review."""
    response = await client.post(
        "/predict",
        json={"text": "Dreadful, boring, and utterly pointless. A complete waste."},
    )
    assert response.status_code == 200
    assert response.json()["label"] == "NEGATIVE"
    assert response.json()["confidence"] > 0.9
