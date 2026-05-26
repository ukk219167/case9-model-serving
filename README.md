---
title: Case9 Sentiment API
emoji: 🎭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Case 9 · Model Serving Lite

**Live demo:** https://huggingface.co/spaces/verryt/case9-sentiment-api  
**Repo:** https://github.com/ukk219167/case9-model-serving  

> A production-ready sentiment classification service — from HuggingFace notebook to a monitored, retrainable API.

---

## What this is

Takes the `distilbert-base-uncased-finetuned-sst-2-english` model and wraps it in a FastAPI service with structured JSON logging, input-distribution drift monitoring, and a CI retrain gate — so the team knows when the model is working, when the inputs are drifting, and how to safely update the model without breaking production.

---

## How to run locally

```bash
git clone https://github.com/ukk219167/case9-model-serving.git
cd case9-model-serving

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt

LOG_FORMAT=pretty uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API explorer.

---

## Example curl commands

**Predict:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This film was absolutely fantastic!"}'
```

**With your own request_id (for end-to-end tracing):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible movie.", "request_id": "550e8400-e29b-41d4-a716-446655440000"}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

**Drift report:**
```bash
curl http://localhost:8000/drift-report
```

---

## Run with Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 \
  -e LOG_FORMAT=pretty \
  -e MODEL_VERSION=distilbert-sst2-v1 \
  sentiment-api
```

---

## Run tests

```bash
# Fast tests only (model mocked — <5 seconds)
pytest tests/ -q

# Include real-model smoke tests (~2 minutes, downloads weights)
RUN_SLOW_TESTS=1 pytest tests/ -q
```

---

## Stack

| Component | Choice | Why |
|---|---|---|
| API framework | FastAPI + uvicorn | Async-native, auto OpenAPI docs, explicit in JD |
| Model | distilbert-base-uncased-finetuned-sst-2-english | Fast CPU inference, 92.7% SST-2 accuracy |
| Logging | structlog (JSON) | Grep-able, jq-parseable, ingestible by any log aggregator |
| Drift store | SQLite (WAL mode) | Zero infra, survives restarts, queryable with sqlite3 CLI |
| Container registry | ghcr.io | Free, integrated with GitHub Actions |
| Deployment | Hugging Face Spaces | Free CPU, ML-native, instant public URL |
| Testing | pytest + httpx ASGI | No real server needed, fast, no port conflicts |
| Linting | ruff | Replaces flake8 + black + isort in one tool |

---

## Architecture

```
POST /predict
    │
    ├── Pydantic validation (schemas.py)
    ├── predict() → HuggingFace pipeline (model.py)
    ├── DriftMonitor.record() → SQLite (drift.py)
    ├── log_prediction() → structured JSON stdout (logging_cfg.py)
    └── PredictResponse → caller

GET /health     → model liveness + drift DB check
GET /drift-report → rolling window stats (last 100 requests)
```

---

## How I would know this model is failing before customers do

Three layers of observability:

**1. Structured logs with drift flag**  
Every prediction emits a JSON log line with `drift_flagged`, `confidence`, and `inference_ms`. A dashboard query like `avg(confidence) WHERE drift_flagged = true` shows whether drifting inputs are also low-confidence predictions — the strongest early signal of real degradation.

**2. `/drift-report` endpoint (three signals)**  
- `text_length` — baseline: μ=72.5 chars. Spike means inputs are changing (longer documents, truncated inputs).  
- `oov_rate` — baseline: μ=0.04. Spike means the domain is shifting (medical notes instead of reviews).  
- `non_ascii_rate` — baseline: μ=0.002. Spike means language switch or junk inputs.  
A z-score > 2.0 on any signal sets `drift_detected: true` before accuracy has had a chance to tank.

**3. CI retrain gate**  
Any PR touching `training/data/` triggers an automatic retrain + evaluation. The gate blocks the PR if accuracy drops more than 2 percentage points below `baseline_metrics.json`. The baseline self-updates on every passing gate.

**What I would add with another week:**  
- Prometheus `/metrics` endpoint + Grafana dashboard for real-time confidence distribution.  
- Shadow deployment: route 5% of traffic to a candidate model, compare accuracy on the same inputs before full rollout.  
- Alert on sustained low-confidence predictions (rolling 15-minute window, p10 confidence < 0.6 → PagerDuty).

---

## What's NOT done

- GPU inference (not available on free-tier hosts; swap `device=-1` to `device=0` to enable).
- Full LoRA fine-tuning (stubbed in `training/train.py` — adapter/threshold calibration is used instead).
- Prometheus metrics endpoint (logged to stdout instead; easily added with `prometheus-fastapi-instrumentator`).
- Redis feature store for embedding caching (overkill for this model's 42ms latency).

---

## CI badges

![CI](https://github.com/ukk219167/case9-model-serving/actions/workflows/ci.yml/badge.svg)
