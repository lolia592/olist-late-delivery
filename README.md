# Olist Late Delivery Prediction

Predicts whether an order will be delivered **Late** or **On-time**, based on order, payment, and customer features from the Olist e-commerce dataset.

This repository takes the model and feature-engineering logic developed in `notebooks/` (Notebooks 1–6) and turns it into a full production system: a reproducible inference pipeline, a REST API, containerization, CI/CD, and monitoring. Training stays in the notebooks — this repo only serves the model that comes out of them.

## Project structure
.
├── app/ # FastAPI application (main.py, schemas.py)
├── config/ # config.yaml — all paths and parameters, no hardcoded values in code
├── data/
│ ├── processed/ # parquet artifacts from Notebooks 1, 2, 3, 5 (DVC-tracked)
│ └── charts/ # saved EDA charts from Notebook 4
├── docs/
│ └── ALERTING.md # documented alerting policy
├── logs/ # app.log (git-ignored)
├── models/ # trained model + fitted transformers (Notebooks 5 & 6 outputs)
├── notebooks/ # original Notebooks 1–6 (exploration + training — not modified)
├── scripts/
│ └── check_everything.sh # one-command health check for the whole project
├── src/ # production Python modules mirroring the notebook pipeline
├── tests/
│ ├── unit/ # preprocessing, feature building, validation (26 tests)
│ ├── data/ # schema, ranges, missing values, leakage checks (13 tests)
│ ├── model/ # model loading, prediction shape, notebook-parity (6 tests)
│ └── integration/ # end-to-end API route tests (8 tests)
├── requirements/
│ ├── base.txt # exact versions needed to run the pipeline / API
│ └── dev.txt # base.txt + notebooks, tests, and dev tools
├── .github/workflows/ci.yml # lint, format check, tests, build & push image
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
└── pytest.ini

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

## Known limitation: the saved scaler is intentionally unused

Notebook 5 fits and saves a `StandardScaler` (`models/notebook5_scaler.pkl`), but the final training table built in that notebook never actually applies its output — the model in `models/notebook6_final_model.pkl` was trained on **log1p-transformed values without scaling**.

To reproduce the notebook's results exactly, the inference pipeline **does not apply the saved scaler** (`config.yaml` → `features.apply_scaler: false`). Applying it would feed the model values on a completely different scale than it was trained on, breaking predictions. This is tracked as a known issue to fix in a future retraining pass.

## Data versioning (DVC)

Data artifacts (`data/processed/*.parquet`, `data/charts/*`) are tracked with DVC instead of git directly. Git stores only small `.dvc` pointer files; the actual data lives in a DVC remote.

```bash
dvc pull   # fetch the real data files
dvc push   # after adding/updating a tracked artifact
```

The current remote is a local folder (`~/dvc-storage`) for this training exercise — swapping it for S3/GCS in production only requires changing `.dvc/config`, no code changes.

Small model artifacts (`models/*.pkl`, `models/*.csv`) and MLflow's tracking files (`mlflow.db`, `mlruns/`) are committed directly to git rather than DVC, since CI and Docker need to access them without a connection to the local DVC remote.

## Data validation (Great Expectations)

Before every prediction, incoming order data is checked against expectations defined in `src/data_validation.py` (column types, value ranges, allowed states, missing rates — based on the real ranges observed in Notebook 4's EDA).

Failures are split by severity:
- **Critical** (null values, unknown state code) → the order is **rejected** with a clear `ValueError`.
- **Warning** (a statistically unusual but still valid value, e.g. an unusually expensive order) → **logged**, and the prediction still proceeds.

## Experiment tracking & model registry (MLflow)

Training runs are logged with MLflow: parameters, metrics (accuracy, precision, recall, f1, roc_auc), and artifacts (the fitted encoder, scaler, rare-states list, and feature names) are recorded in `notebooks/Notebook6.ipynb`.

The chosen model is registered in the MLflow Model Registry under `olist-late-delivery-model`, tagged with the `production` alias (MLflow's modern replacement for the deprecated stage-based system). `src/model_loader.py` loads the model directly from the registry — never from a local `.pkl` file — so promoting a new version never requires a code change.

**Note on absolute paths:** MLflow's local file store records absolute host paths at logging time. Inside Docker, this is handled by matching the container's `WORKDIR` to the host's absolute project path (see the Docker section below). In CI, where paths can't be matched this way, the affected tests are skipped (see the CI section below).

## Testing

The project uses `pytest`. Run the entire suite with a single command:

```bash
pytest
```

Tests are organized by type:
- `tests/unit/` — pure function tests for preprocessing, feature building, and validation logic (no I/O, no model).
- `tests/data/` — schema, value ranges, missing values, and a leakage check (no `order_id` appears in more than one split).
- `tests/model/` — the model loads, predicts the right shape, and behaves sensibly on known inputs; also confirms the feature pipeline output is byte-for-byte identical to Notebook 5's saved output on real data.
- `tests/integration/` — end-to-end FastAPI route tests using `TestClient`.

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
| GET | `/metrics` | Prometheus metrics for scraping |

Request/response shapes are enforced by Pydantic schemas (`app/schemas.py`). Malformed requests (wrong types, missing fields, out-of-range values) are rejected with a `422` response before reaching the pipeline. Requests that pass schema validation but fail deeper checks (e.g. an unknown state code, caught by Great Expectations inside `src/pipeline.py`) are also rejected with a `422` and a clear reason.

All route logic is a thin wrapper around `src/pipeline.run_pipeline()` — no prediction logic is duplicated in the API layer.

## Running with Docker

The entire stack (API + a fresh Postgres database) starts with a single command on a clean machine:

```bash
docker compose up --build
```

The API is then available at `http://localhost:8000` (Postgres on `localhost:5433`, to avoid clashing with any local Postgres already running on the default port).

**Design notes:**
- The image only contains application code (`app/`, `src/`, `config/`) — no notebooks, no data, no models baked in, keeping it small.
- Model artifacts and the MLflow tracking database are mounted as volumes (`models/`, `mlflow.db`, `mlruns/`) rather than copied into the image, so a new model version doesn't require rebuilding the image.
- Secrets and connection strings come from `.env` (never committed) via `env_file` in `docker-compose.yml`.
- The container's `WORKDIR` matches the host's absolute project path. This is required because MLflow's local file store records absolute artifact paths at logging time; matching paths let the container resolve them without any translation.

## CI/CD (GitHub Actions)

Every push to `main` runs `.github/workflows/ci.yml`:
1. **Lint** with `ruff check .`
2. **Format check** with `ruff format --check .`
3. **Tests** with `pytest`
4. If everything passes, **build and push** the Docker image to GitHub Container Registry (`ghcr.io`).

A failing lint, format check, or test stops the pipeline — the image is never built or pushed on failure.

`.pre-commit-config.yaml` runs the same lint/format checks locally before each commit, catching issues before they ever reach GitHub.

**CI test coverage:** GitHub Actions runs `tests/unit/` only (26 tests). The following are excluded from CI and run locally / in Docker instead, since they depend on local-only state:
- `tests/data/` and `tests/model/test_pipeline_matches_notebook.py` — read `data/processed/*.parquet`, which are DVC-tracked and not present in a fresh CI checkout (the CI runner has no access to the local DVC remote).
- `tests/model/test_model.py` and `tests/integration/test_api.py` — import `src.model_loader`, which loads the model from MLflow's registry. MLflow's local file store records absolute host paths that don't resolve on GitHub's runners (they resolve locally and inside Docker Compose, where volumes are mounted consistently).

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

Once an order's real delivery date is known, `actual_outcome` can be backfilled and compared against the original prediction — this is what enables measuring real-world model accuracy over time, not just accuracy on the original test set.

### Alerting

See [`docs/ALERTING.md`](docs/ALERTING.md) for the documented decision on what conditions should trigger an alert (error rate, latency, prediction drift, uptime) and why. No alerting infrastructure is wired up — this task calls for the decision to be made and written down, which is what that document is.

## Security note: database credentials in notebooks

An earlier version of `Notebook1.ipynb` and `Notebook5.ipynb` hardcoded the local Postgres connection details (including the password) directly in a code cell. This has been fixed — both notebooks now load credentials from `.env` via `python-dotenv`, consistent with the rest of the project.

Older commits in this repository's history still contain the original hardcoded (local, development-only) password. Since this is a private repository, the password was left as-is rather than rotated; in a real/shared repository, any credential that has ever been committed should be treated as compromised and rotated immediately, regardless of whether the repo is later "cleaned up."

## Project status

All 10 steps of Task 3 are complete:

1. Repository structure & configuration
2. Notebooks refactored into Python modules
3. Logging & error handling
4. Data versioning (DVC) & validation (Great Expectations)
5. Experiment tracking & model registry (MLflow)
6. Testing (pytest — 53 tests)
7. Inference API (FastAPI)
8. Containerization (Docker & Docker Compose)
9. CI/CD (GitHub Actions, ruff, pre-commit)
10. Monitoring (Prometheus metrics, prediction logging, alerting policy)

Run `scripts/check_everything.sh` for a one-command health check covering every step above.