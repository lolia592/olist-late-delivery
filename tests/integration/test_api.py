"""
Integration tests for the FastAPI service, end to end.

Uses FastAPI's TestClient, which runs the app in-process without
needing a live uvicorn server — every request goes through the
real routes, schemas, and the full src/pipeline.py logic.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.requires_mlflow

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_returns_expected_fields():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "olist-late-delivery-model"
    assert body["model_alias"] == "production"
    assert body["model_type"] == "LogisticRegression"


def test_predict_valid_order_returns_200():
    response = client.post(
        "/predict",
        json={
            "total_price": 149.90,
            "total_freight": 18.50,
            "n_items": 2,
            "order_purchase_timestamp": "2026-09-05 14:30:00",
            "customer_state": "SP",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Late", "On-time"}
    assert 0 <= body["probability_late"] <= 1
    assert body["model_version"] == "v1-notebook6"


def test_predict_negative_price_returns_422():
    response = client.post(
        "/predict",
        json={
            "total_price": -10,
            "total_freight": 18.50,
            "n_items": 2,
            "order_purchase_timestamp": "2026-09-05 14:30:00",
            "customer_state": "SP",
        },
    )
    assert response.status_code == 422


def test_predict_missing_field_returns_422():
    response = client.post(
        "/predict",
        json={
            "total_price": 149.90,
            "total_freight": 18.50,
            "n_items": 2,
            "customer_state": "SP",
            # order_purchase_timestamp intentionally missing
        },
    )
    assert response.status_code == 422


def test_predict_unknown_state_returns_422():
    # Passes Pydantic's shape check (2 letters), but fails the
    # deeper Great Expectations / validation logic inside the pipeline.
    response = client.post(
        "/predict",
        json={
            "total_price": 149.90,
            "total_freight": 18.50,
            "n_items": 2,
            "order_purchase_timestamp": "2026-09-05 14:30:00",
            "customer_state": "XX",
        },
    )
    assert response.status_code == 422


def test_predict_batch_returns_one_result_per_order():
    response = client.post(
        "/predict/batch",
        json={
            "orders": [
                {
                    "total_price": 149.90,
                    "total_freight": 18.50,
                    "n_items": 2,
                    "order_purchase_timestamp": "2026-09-05 14:30:00",
                    "customer_state": "SP",
                },
                {
                    "total_price": 85.00,
                    "total_freight": 16.79,
                    "n_items": 1,
                    "order_purchase_timestamp": "2026-06-10 09:00:00",
                    "customer_state": "RJ",
                },
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    for prediction in body["predictions"]:
        assert prediction["prediction"] in {"Late", "On-time"}


def test_predict_batch_empty_list_returns_422():
    response = client.post("/predict/batch", json={"orders": []})
    assert response.status_code == 422
