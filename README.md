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
