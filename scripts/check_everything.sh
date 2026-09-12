#!/bin/bash
# Full project health check — verifies every step of Task 3, from
# repo structure through CI/CD, in one pass.

set -uo pipefail
PASS=0
FAIL=0

check() {
  if eval "$2" > /tmp/check_output 2>&1; then
    echo "✅ $1"
    PASS=$((PASS+1))
  else
    echo "❌ $1"
    FAIL=$((FAIL+1))
  fi
}

echo "=================================================="
echo "  STEP 1 — Repo structure & config"
echo "=================================================="
for d in app config data models notebooks src tests requirements logs; do
  check "$d/ exists" "[ -d $d ]"
done
check "config/config.yaml is valid YAML" "python -c \"import yaml; yaml.safe_load(open('config/config.yaml'))\""
check ".env is git-ignored" "git check-ignore -q .env"
check "requirements/base.txt exists" "[ -f requirements/base.txt ]"
check "requirements/dev.txt exists" "[ -f requirements/dev.txt ]"

echo ""
echo "=================================================="
echo "  STEP 2 — Notebooks to Python modules"
echo "=================================================="
for f in config validation preprocessing feature_builder model_loader predictor pipeline exceptions data_loader data_validation; do
  check "src/$f.py exists" "[ -f src/$f.py ]"
done
check "feature pipeline matches Notebook 5 exactly" "python -m pytest tests/model/test_pipeline_matches_notebook.py -q"

echo ""
echo "=================================================="
echo "  STEP 3 — Logging & error handling"
echo "=================================================="
check "src/logger.py exists" "[ -f src/logger.py ]"
check "no bare print() in src/" "! grep -rn 'print(' src/ | grep -v '#'"
check "logs/ directory exists" "[ -d logs ]"

echo ""
echo "=================================================="
echo "  STEP 4 — DVC & Great Expectations"
echo "=================================================="
check "DVC initialized" "[ -d .dvc ]"
check "DVC remote configured" "dvc remote list | grep -q local-storage"
check "src/data_validation.py exists" "[ -f src/data_validation.py ]"
check "at least one .dvc pointer file exists" "find data -name '*.dvc' | grep -q ."

echo ""
echo "=================================================="
echo "  STEP 5 — MLflow"
echo "=================================================="
check "mlflow.db exists" "[ -f mlflow.db ]"
check "model is registered with an alias" "python -c \"
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
c = mlflow.MlflowClient()
c.get_model_version_by_alias('olist-late-delivery-model', 'production')
\""

echo ""
echo "=================================================="
echo "  STEP 6 — Testing"
echo "=================================================="
check "pytest.ini exists" "[ -f pytest.ini ]"
check "full local test suite passes (53 tests)" "python -m pytest -q"

echo ""
echo "=================================================="
echo "  STEP 7 — FastAPI"
echo "=================================================="
check "app/main.py exists" "[ -f app/main.py ]"
check "app/schemas.py exists" "[ -f app/schemas.py ]"
check "API health route works (in-process)" "python -c \"
from fastapi.testclient import TestClient
from app.main import app
r = TestClient(app).get('/health')
assert r.status_code == 200
\""

echo ""
echo "=================================================="
echo "  STEP 8 — Docker"
echo "=================================================="
check "Dockerfile exists" "[ -f Dockerfile ]"
check "docker-compose.yml exists" "[ -f docker-compose.yml ]"
check ".dockerignore excludes notebooks/" "grep -q '^notebooks/' .dockerignore"
check "Docker is installed" "docker --version"

echo ""
echo "=================================================="
echo "  STEP 9 — CI/CD"
echo "=================================================="
check ".github/workflows/ci.yml exists" "[ -f .github/workflows/ci.yml ]"
check ".pre-commit-config.yaml exists" "[ -f .pre-commit-config.yaml ]"
check "ruff lint passes" "ruff check ."
check "ruff format check passes" "ruff format --check ."

echo ""
echo "=================================================="
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "=================================================="

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
