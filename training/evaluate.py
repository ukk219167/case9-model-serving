"""
training/evaluate.py
--------------------
Evaluate the trained model on a held-out set and write metrics.json.

The regression gate in retrain.yml reads metrics.json and compares
accuracy against baseline_metrics.json. This script produces that file.

Metrics reported:
    accuracy   — fraction of correct predictions (primary gate metric)
    f1         — macro F1 (catches class imbalance issues accuracy misses)
    precision  — macro precision
    recall     — macro recall
    threshold  — the decision threshold used (from threshold.json if present)
    n_samples  — number of rows evaluated

Usage:
    python training/evaluate.py \
        --model-dir  training/model_output/ \
        --data-path  training/data/held_out.csv \
        --output     metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from sklearn.metrics import (  # type: ignore[import]
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline, Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the sentiment model on held-out data.")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("training/model_output"),
        help="Directory containing threshold.json (output of train.py).",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path("training/data/held_out.csv"),
        help="CSV file with 'text' and 'label' columns for evaluation.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("metrics.json"),
        help="Path to write the metrics JSON file.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=5_000,
        help="Cap evaluation rows.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_csv(path: Path, max_rows: int) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            text = row.get("text", "").strip()
            label_raw = row.get("label", "").strip()
            if not text or label_raw not in ("0", "1"):
                continue
            texts.append(text)
            labels.append(int(label_raw))
    log.info("Loaded %d evaluation rows from %s", len(texts), path)
    return texts, labels


# ---------------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------------


def load_threshold(model_dir: Path) -> float:
    """Load calibrated threshold from train.py output, or fall back to 0.5."""
    threshold_path = model_dir / "threshold.json"
    if threshold_path.exists():
        config = json.loads(threshold_path.read_text())
        threshold = float(config.get("threshold", 0.5))
        log.info("Using calibrated threshold: %.4f", threshold)
        return threshold
    log.info("No threshold.json found — using default threshold: 0.5")
    return 0.5


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    clf: Pipeline,
    texts: list[str],
    threshold: float,
) -> list[int]:
    """
    Run the pipeline in batches and apply the calibrated threshold.
    Returns a list of integer labels (0 = NEGATIVE, 1 = POSITIVE).
    """
    predictions = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = clf(batch, truncation=True, max_length=512)
        for r in results:
            if r["label"].upper() in ("POSITIVE", "LABEL_1"):
                positive_score = r["score"]
            else:
                positive_score = 1.0 - r["score"]
            predictions.append(1 if positive_score >= threshold else 0)

        if i % 320 == 0:
            log.info("  … %d / %d", min(i + batch_size, len(texts)), len(texts))

    return predictions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Held-out data not found at {args.data_path}")

    texts, true_labels = load_csv(args.data_path, max_rows=args.max_rows)
    if len(texts) == 0:
        raise ValueError("Evaluation CSV is empty or has no valid rows.")

    threshold = load_threshold(args.model_dir)

    log.info("Loading model '%s' …", MODEL_NAME)
    clf: Pipeline = pipeline("text-classification", model=MODEL_NAME, device=-1)

    log.info("Running inference on %d samples …", len(texts))
    pred_labels = run_inference(clf, texts, threshold)

    # ---------------------------------------------------------------------------
    # Compute metrics
    # ---------------------------------------------------------------------------
    accuracy  = accuracy_score(true_labels, pred_labels)
    f1        = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall    = recall_score(true_labels, pred_labels, average="macro", zero_division=0)

    metrics = {
        "accuracy":  round(accuracy,  4),
        "f1":        round(f1,        4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "threshold": threshold,
        "n_samples": len(texts),
        "model_name": MODEL_NAME,
    }

    log.info("Results:")
    for k, v in metrics.items():
        log.info("  %-12s %s", k, v)

    args.output.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics written to %s", args.output)


if __name__ == "__main__":
    main()
