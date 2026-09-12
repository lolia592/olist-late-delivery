"""
Request and response schemas for the API.

Pydantic validates every incoming request against these models
automatically — malformed requests are rejected with a clear 422
error before they ever reach our route functions.
"""

from pydantic import BaseModel, Field


class OrderRequest(BaseModel):
    """A single order to predict on."""

    total_price: float = Field(..., gt=0, description="Order total price in BRL")
    total_freight: float = Field(..., ge=0, description="Shipping cost in BRL")
    n_items: int = Field(..., gt=0, description="Number of items in the order")
    order_purchase_timestamp: str = Field(
        ..., description="ISO datetime string, e.g. 2026-09-05 14:30:00"
    )
    customer_state: str = Field(
        ..., min_length=2, max_length=2, description="2-letter Brazilian state code"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_price": 149.90,
                "total_freight": 18.50,
                "n_items": 2,
                "order_purchase_timestamp": "2026-09-05 14:30:00",
                "customer_state": "SP",
            }
        }


class BatchOrderRequest(BaseModel):
    """Multiple orders to predict on in a single request."""

    orders: list[OrderRequest] = Field(..., min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class PredictionResponse(BaseModel):
    """The result of a single prediction."""

    prediction: str
    probability_late: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    """The results of a batch prediction."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_alias: str
    model_type: str
    model_version: str
