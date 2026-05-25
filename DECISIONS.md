# Decisions Log — Case 9: Model Serving Lite

## Assumptions I made

1. **Free-tier CPU only** — The host (HF Spaces) has no GPU. All inference runs on CPU with `device=-1`. This is fine for distilbert (42ms p50 latency on CPU) but would need revisiting for larger models.
2. **Single worker** — `uvicorn --workers 1` to avoid running multiple copies of the 250MB model in RAM. Horizontal scaling (multiple containers) is the correct production strategy.
3. **SST-2 baseline stats** — Drift baselines (text length μ=72.5, OOV rate μ=0.04) are approximated from the SST-2 corpus. In production these would be computed offline from actual production traffic.
4. **Adapter mode for retraining** — Assumed the new training data volume would be small (<5k rows), making threshold calibration more appropriate than full fine-tuning.

---

## Trade-offs

| Choice | Alternative | Why I picked this |
|---|---|---|
| `structlog` JSON logging | Python stdlib logging | structlog binds context vars (request_id) automatically across a request; stdlib requires manual passing |
| SQLite for drift stats | In-memory dict | Survives container restarts; queryable with `sqlite3` CLI during incidents |
| SQLite WAL mode | Default journal mode | WAL allows concurrent readers without blocking the writer on the hot /predict path |
| `@lru_cache` singleton for model | Module-level global | lru_cache is more explicit, testable (cache_clear()), and avoids import-time side effects |
| `def` not `async def` on /predict | async def | HuggingFace pipeline is not async-native; `def` tells FastAPI to run it in a threadpool, keeping the event loop free |
| CPU-only torch wheel | Default torch | Saves ~1.7GB in the Docker image; free-tier hosts have no CUDA |
| Multi-stage Dockerfile | Single-stage | Strips gcc/g++ from the runtime image — smaller attack surface, ~200MB smaller image |
| Macro F1 in baseline metrics | Accuracy only | Accuracy is gameable on imbalanced sets; F1 catches class imbalance issues immediately |
| Platt-scaling for retraining | LoRA fine-tuning | Base model already at 92.7% accuracy; fine-tuning on <5k examples risks overfitting without improving generalisation |
| `drift_detected` non-fatal | Block request on drift | Drift is a monitoring signal, not an error condition. Blocking predictions on drift would silently degrade user experience |

---

## What I de-scoped and why

- **Prometheus `/metrics` endpoint** — Added `X-Process-Time-Ms` header as a lightweight alternative. A Prometheus endpoint would require a sidecar scraper and Grafana, which is too much infra for a free-tier demo. The `/drift-report` endpoint serves the same monitoring purpose for this submission.
- **Redis embedding cache** — At 42ms inference latency on CPU, caching embeddings in Redis would add network overhead that likely exceeds the cache benefit for this model size. Would revisit if using a larger model (>1B parameters).
- **Full LoRA fine-tuning** — Stubbed in `training/train.py` with a `NotImplementedError`. The adapter/threshold calibration path is the correct choice for this data volume and time constraint.
- **Blue/green deployment** — The HF Spaces deploy is atomic (git push triggers a rebuild). A proper blue/green would require two Spaces + a routing layer, which is beyond the free-tier scope.

---

## What I would do differently with another day

- Compute drift baselines from actual SST-2 training data offline, store in `baseline_metrics.json`, and load them in `drift.py` instead of hard-coding approximations.
- Add a Prometheus `/metrics` endpoint and a pre-built Grafana dashboard JSON in the repo.
- Implement the shadow deployment stretch goal: route 5% of `/predict` traffic to a candidate model endpoint, log both predictions, and compute accuracy delta over a 24-hour window before promoting.
- Write a load test (locust or k6) to measure p50/p95/p99 latency under 50 concurrent users and document the breaking point.
- Add a proper vocabulary file (`app/sst2_vocab_10k.txt`) computed from the actual SST-2 training set so the OOV drift signal is calibrated correctly.
