"""
FastAPI service for the Olist late-delivery prediction pipeline.

Thin routes only — all real logic (validation, feature engineering,
prediction) lives in src/pipeline.py. This file just receives HTTP
requests, validates their shape with Pydantic schemas, calls the
pipeline, and shapes the response.
"""

from fastapi import FastAPI, HTTPException

from app.schemas import (
    BatchOrderRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    OrderRequest,
    PredictionResponse,
)
from src.config import settings
from src.exceptions import PipelineError
from src.pipeline import run_pipeline

app = FastAPI(
    title="Olist Late Delivery Prediction API",
    description="Predicts whether an order will be delivered late or on time.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Simple liveness check — confirms the service is up."""
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    """Returns which model is currently loaded and serving predictions."""
    return ModelInfoResponse(
        model_name=settings.get("mlflow", "model_name"),
        model_alias=settings.get("mlflow", "model_alias"),
        model_type=settings.get("model", "type"),
        model_version=settings.get("model", "version"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(order: OrderRequest):
    """Predict Late/On-time for a single order."""
    try:
        result = run_pipeline(order.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except PipelineError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(batch: BatchOrderRequest):
    """Predict Late/On-time for multiple orders in one request."""
    predictions = []
    for order in batch.orders:
        try:
            result = run_pipeline(order.model_dump())
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Order {order.model_dump()} rejected: {e}")
        except PipelineError as e:
            raise HTTPException(status_code=500, detail=str(e))

        predictions.append(PredictionResponse(**result))

    return BatchPredictionResponse(predictions=predictions)
