"""
app/model.py
------------
Singleton model loader and inference wrapper for the sentiment classifier.

Design decisions:
- Model loads once at startup via get_model() — avoids cold-load on first request.
- Uses transformers pipeline() so the tokeniser + model stay coupled.
- Returns a typed PredictionResult dataclass; callers never touch raw pipeline output.
- MODEL_VERSION is read from an env var so CI can stamp the deployed image version
  without changing code.
- Warmup call on load ensures the first real request doesn't pay the JIT cost.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from transformers import Pipeline, pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME: str = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "distilbert-sst2-v1")

# Map the raw label strings the pipeline returns to clean public-facing labels.
_LABEL_MAP: dict[str, str] = {
    "POSITIVE": "POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "LABEL_0": "NEGATIVE",  # fallback for some HF model variants
    "LABEL_1": "POSITIVE",
}

SentimentLabel = Literal["POSITIVE", "NEGATIVE"]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Immutable result returned by predict()."""

    label: SentimentLabel  # "POSITIVE" | "NEGATIVE"
    confidence: float  # 0.0 – 1.0
    model_version: str  # e.g. "distilbert-sst2-v1"
    inference_ms: float  # wall-clock inference time in milliseconds


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    """
    Load and cache the HuggingFace pipeline.

    lru_cache(maxsize=1) means this is called exactly once per process.
    Subsequent calls return the same pipeline object with zero overhead.

    The model is downloaded to HF_HOME (default ~/.cache/huggingface) on
    first run; Docker layer-caching keeps this fast on subsequent builds.
    """
    logger.info("Loading model '%s' …", MODEL_NAME)
    t0 = time.perf_counter()

    clf: Pipeline = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        # device=-1 forces CPU; safe on free-tier hosts with no GPU.
        # Swap to device=0 if a GPU is available.
        device=-1,
    )

    elapsed = (time.perf_counter() - t0) * 1_000
    logger.info("Model loaded in %.0f ms. Running warmup …", elapsed)

    # Warmup: one dummy forward pass so the first real request isn't slow.
    clf("warmup")
    logger.info("Warmup complete. Model ready.")

    return clf


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict(text: str) -> PredictionResult:
    """
    Run sentiment classification on *text*.

    Args:
        text: Raw input string from the user. May be any length; the
              tokeniser will truncate at 512 tokens automatically.

    Returns:
        PredictionResult with label, confidence, version, and latency.

    Raises:
        ValueError: If *text* is empty after stripping whitespace.
        RuntimeError: If the pipeline returns an unexpected label.
    """
    text = text.strip()
    if not text:
        raise ValueError("Input text must not be empty.")

    clf = get_model()

    t0 = time.perf_counter()
    # pipeline() always returns a list; [0] picks the top result.
    raw: dict = clf(text, truncation=True, max_length=512)[0]
    inference_ms = (time.perf_counter() - t0) * 1_000

    raw_label: str = raw["label"].upper()
    label = _LABEL_MAP.get(raw_label)
    if label is None:
        raise RuntimeError(
            f"Unexpected label '{raw_label}' from pipeline. " f"Known labels: {list(_LABEL_MAP)}"
        )

    return PredictionResult(
        label=label,  # type: ignore[arg-type]
        confidence=round(float(raw["score"]), 6),
        model_version=MODEL_VERSION,
        inference_ms=round(inference_ms, 2),
    )
