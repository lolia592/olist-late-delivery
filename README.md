# Olist Late Delivery Prediction

Predicts whether an order will be delivered **Late** or **On-time**, based on order, payment, and customer features from the Olist e-commerce dataset.

This repository takes the model and feature-engineering logic developed in `notebooks/` (Notebooks 1–6) and turns it into a reproducible inference pipeline: same config, same fitted transformers, same model — no retraining happens outside the notebooks at this stage.

## Project structure

    .
    ├── app/           # FastAPI application (routes, schemas) — served over HTTP
    ├── config/        # config.yaml — all paths and parameters, no hardcoded values in code
    ├── data/
    │   ├── processed/ # parquet artifacts produced by Notebooks 1, 2, 3, 5
    │   └── charts/    # saved EDA charts from Notebook 4
    ├── models/        # trained model + fitted transformers (Notebooks 5 & 6 outputs)
    ├── notebooks/     # original Notebooks 1–6 (exploration + training — not modified)
    ├── src/           # production Python modules mirroring the notebook pipeline
    ├── tests/         # pytest suite (unit, data, model, integration)
    └── requirements/
        ├── base.txt   # exact versions needed to run the pipeline / API
        └── dev.txt    # base.txt + notebooks, tests, and dev tools

## Setup — running this from scratch

1. **Clone the repo and enter it**

```bash
   git clone <repo-url>
   cd olist-eda
```

2. **Create and activate a virtual environment**

```bash
   python3 -m venv .venv
   source .venv/bin/activate
```

3. **Install dependencies**

   - For running the pipeline / API only:
```bash
     pip install -r requirements/base.txt
```
   - For development (notebooks, tests, everything above included):
```bash
     pip install -r requirements/dev.txt
```

4. **Set up environment variables**

```bash
   cp .env.example .env
```

   Then edit `.env` and fill in the real database credentials. `.env` is git-ignored and must never be committed.

5. **Check `config/config.yaml`** for paths and parameters — nothing in the code is hardcoded; everything reads from this file.

## Known limitation (intentional, documented)

Notebook 5 fits and saves a `StandardScaler` (`models/notebook5_scaler.pkl`), but the final training table built in that notebook never actually applies its output — the model in `models/notebook6_final_model.pkl` was trained on **log1p-transformed values without scaling**.

To reproduce the notebook's results exactly, the inference pipeline **does not apply the saved scaler** (`config.yaml` → `features.apply_scaler: false`). Applying it would feed the model values on a completely different scale than it was trained on, breaking predictions. This is tracked as a known issue to fix in a future retraining pass — see `notebooks/eda_report.md` for context.

## Status

🚧 Work in progress — repository structure and configuration are in place. Pipeline modules (`src/`), tests, FastAPI service, Docker, MLflow, DVC, and CI/CD are being added incrementally.
## Data versioning (DVC)

All data and model artifacts (`data/processed/*.parquet`,
`models/*.pkl`, `models/*.csv`) are tracked with DVC instead of
git directly. Git stores only small `.dvc` pointer files; the
actual data lives in a DVC remote.

```bash
dvc pull   # fetch the real data/model files
dvc push   # after adding/updating a tracked artifact
```

The current remote is a local folder (`~/dvc-storage`) for this
training exercise — swapping it for S3/GCS in production only
requires changing `.dvc/config`, no code changes.

## Data validation (Great Expectations)

Before every prediction, incoming order data is checked against
expectations defined in `src/data_validation.py` (column types,
value ranges, allowed states, missing rates — based on the real
ranges observed in Notebook 4's EDA).

Failures are split by severity:
- **Critical** (null values, unknown state code) → the order is
  **rejected** with a clear `ValueError`.
- **Warning** (a statistically unusual but still valid value, e.g.
  an unusually expensive order) → **logged**, and the prediction
  still proceeds.

## Experiment tracking & model registry (MLflow)

Training runs are logged with MLflow: parameters, metrics
(accuracy, precision, recall, f1, roc_auc), and artifacts (the
fitted encoder, scaler, rare-states list, and feature names) are
recorded in `notebooks/Notebook6.ipynb`.

The chosen model is registered in the MLflow Model Registry under
`olist-late-delivery-model`, tagged with the `production` alias
(MLflow's modern replacement for the deprecated stage-based system).
`src/model_loader.py` loads the model directly from the registry —
never from a local `.pkl` file — so promoting a new version never
requires a code change.

**Known limitation (to be addressed in the Docker step):** the
MLflow tracking store (`mlflow.db`) and artifact store currently
live on the local filesystem. Once the service is containerized,
these need to be reachable from inside the container — either by
mounting them as a volume in `docker-compose.yml`, or by running
MLflow as its own service in the stack. This is deferred to that
step, once the full container layout (API + database + MLflow) is
decided together.

## Testing

The project uses `pytest`. Run the entire suite with a single command:

```bash
pytest
```

Tests are organized by type:
- `tests/unit/` — pure function tests for preprocessing, feature
  building, and validation logic (no I/O, no model).
- `tests/data/` — schema, value ranges, missing values, and a
  leakage check (no `order_id` appears in more than one split).
- `tests/model/` — the model loads, predicts the right shape, and
  behaves sensibly on known inputs; also confirms the feature
  pipeline output is byte-for-byte identical to Notebook 5's saved
  output on real data.
- `tests/integration/` — end-to-end API route tests, added once
  the API exists (Task 3's next step).

## API (FastAPI)

Run the service locally:

```bash
uvicorn app.main:app --reload
```

Interactive docs (try every route from the browser): `http://127.0.0.1:8000/docs`

**Routes:**

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/model-info` | Currently served model name, alias, type, version |
| POST | `/predict` | Predict Late/On-time for a single order |
| POST | `/predict/batch` | Predict for multiple orders in one request |

Request/response shapes are enforced by Pydantic schemas
(`app/schemas.py`). Malformed requests (wrong types, missing
fields, out-of-range values) are rejected with a `422` response
before reaching the pipeline. Requests that pass schema validation
but fail deeper checks (e.g. an unknown state code, caught by
Great Expectations inside `src/pipeline.py`) are also rejected
with a `422` and a clear reason.

All route logic is a thin wrapper around `src/pipeline.run_pipeline()` —
no prediction logic is duplicated in the API layer.

## Running with Docker

The entire stack (API + a fresh Postgres database) starts with a
single command on a clean machine:

```bash
docker compose up --build
```

The API is then available at `http://localhost:8000` (Postgres on
`localhost:5433`, to avoid clashing with any local Postgres already
running on the default port).

**Design notes:**
- The image only contains application code (`app/`, `src/`,
  `config/`) — no notebooks, no data, no models baked in, keeping
  it small.
- Model artifacts and the MLflow tracking database are mounted as
  volumes (`models/`, `mlflow.db`, `mlruns/`) rather than copied
  into the image, so a new model version doesn't require rebuilding
  the image.
- Secrets and connection strings come from `.env` (never committed)
  via `env_file` in `docker-compose.yml`.
- The container's `WORKDIR` matches the host's absolute project
  path. This is required because MLflow's local file store records
  absolute artifact paths at logging time; matching paths let the
  container resolve them without any translation.

## CI test coverage (updated)

GitHub Actions runs `pytest` on `tests/unit/` only (26 tests: 
preprocessing, feature building, validation). The following are
excluded from CI and run locally / in Docker instead:

- `tests/data/` and `tests/model/test_pipeline_matches_notebook.py`
  — read `data/processed/*.parquet`, which are DVC-tracked and not
  present in a fresh CI checkout (the CI runner has no access to
  the local DVC remote).
- `tests/model/test_model.py` and `tests/integration/test_api.py`
  — import `src.model_loader`, which loads the model from MLflow's
  registry. MLflow's local file store records absolute host paths
  that don't resolve on GitHub's runners (they do resolve locally
  and inside Docker Compose, where volumes are mounted consistently).

Run the full 53-test suite locally or via `docker compose exec api pytest`.

## CI test coverage (updated)

GitHub Actions runs `pytest` on `tests/unit/` only (26 tests: 
preprocessing, feature building, validation). The following are
excluded from CI and run locally / in Docker instead:

- `tests/data/` and `tests/model/test_pipeline_matches_notebook.py`
  — read `data/processed/*.parquet`, which are DVC-tracked and not
  present in a fresh CI checkout (the CI runner has no access to
  the local DVC remote).
- `tests/model/test_model.py` and `tests/integration/test_api.py`
  — import `src.model_loader`, which loads the model from MLflow's
  registry. MLflow's local file store records absolute host paths
  that don't resolve on GitHub's runners (they do resolve locally
  and inside Docker Compose, where volumes are mounted consistently).

Run the full 53-test suite locally or via `docker compose exec api pytest`.

## Monitoring

### Metrics

The service exposes Prometheus-format metrics at `GET /metrics`:

| Metric | Type | What it tracks |
|---|---|---|
| `prediction_requests_total` | Counter | Total prediction requests received |
| `prediction_errors_total{error_type}` | Counter | Failed requests, split into `invalid_input` (bad data — expected) vs `internal_error` (unexpected bugs) |
| `prediction_latency_seconds` | Histogram | End-to-end request latency distribution |
| `predictions_by_class_total{prediction}` | Counter | Predictions per class ("Late"/"On-time") — the basis for spotting prediction drift over time |

Point a real Prometheus server at `/metrics` to scrape and graph these in Grafana.

### Prediction logging (for future evaluation)

Every prediction is persisted to `predictions.db` (SQLite, git-ignored — it grows continuously and is runtime state, not code) via `src/prediction_log.py`. Each row stores the input, the prediction, the probability, the model version, and an `actual_outcome` column left `NULL` at prediction time.

Once an order's real delivery date is known, `actual_outcome` can be backfilled and compared against the original prediction — this is what enables measuring real-world model accuracy over time, not just accuracy on the original test set (a separate evaluation job, out of scope for this task).

### Alerting

See [`docs/ALERTING.md`](docs/ALERTING.md) for the documented decision on what conditions should trigger an alert (error rate, latency, prediction drift, uptime) and why. No alerting infrastructure is wired up — this task calls for the decision to be made and written down, which is what that document is.
