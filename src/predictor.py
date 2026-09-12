"""
Prediction module — the final step of the inference pipeline.

Combines validation, feature engineering, and the trained model to
turn a single new order into a clear prediction with a probability,
without ever re-fitting anything.
"""

import pandas as pd

from src.feature_builder import build_features
from src.model_loader import get_model
from src.validation import validate_order

_model = get_model()

# Position of "Late" inside model.classes_ (order comes from the
# model itself, never hardcoded — see model.classes_ check below).
_LATE_INDEX = list(_model.classes_).index("Late")


def predict_order(order: dict) -> dict:
    """
    Predict whether a single new order will be Late or On-time.

    Parameters
    ----------
    order : dict
        Raw order data, e.g.
        {
            "total_price": 149.90,
            "total_freight": 18.50,
            "n_items": 2,
            "order_purchase_timestamp": "2026-09-05 14:30:00",
            "customer_state": "SP",
        }

    Returns
    -------
    dict
        {
            "prediction": "Late" | "On-time",
            "probability_late": float,   # between 0 and 1
        }
    """
    # 1. Validate — raises ValueError if the order is malformed.
    validate_order(order)

    # 2. Build features — exactly what Notebook 5 would produce.
    orders_df = pd.DataFrame([order])
    features = build_features(orders_df)

    # 3. Predict — never re-fit, only ever transform + predict.
    prediction = _model.predict(features)[0]
    probabilities = _model.predict_proba(features)[0]
    probability_late = probabilities[_LATE_INDEX]

    return {
        "prediction": prediction,
        "probability_late": float(probability_late),
    }
