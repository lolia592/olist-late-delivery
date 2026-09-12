"""
Model tests — confirms the trained model loads correctly, predicts
in the expected shape, and behaves sensibly on known inputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_builder import build_features
from src.model_loader import get_model

pytestmark = pytest.mark.requires_mlflow


@pytest.fixture(scope="module")
def model():
    return get_model()


def test_model_loads_successfully(model):
    assert model is not None
    assert type(model).__name__ == "LogisticRegression"


def test_model_has_expected_classes(model):
    assert set(model.classes_) == {"Late", "On-time"}


def test_predict_returns_one_label_per_row(model):
    orders = pd.DataFrame(
        [
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
    )
    features = build_features(orders)
    predictions = model.predict(features)

    assert len(predictions) == 2
    assert set(predictions).issubset({"Late", "On-time"})


def test_predict_proba_returns_valid_probabilities(model):
    orders = pd.DataFrame(
        [
            {
                "total_price": 149.90,
                "total_freight": 18.50,
                "n_items": 2,
                "order_purchase_timestamp": "2026-09-05 14:30:00",
                "customer_state": "SP",
            },
        ]
    )
    features = build_features(orders)
    probabilities = model.predict_proba(features)

    # One row, two classes.
    assert probabilities.shape == (1, 2)
    # Probabilities must be between 0 and 1, and sum to 1.
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.isclose(probabilities.sum(), 1.0)


def test_holiday_season_order_has_higher_late_risk_than_regular(model):
    """
    Sanity check on known behavior: an otherwise identical order
    placed in the holiday season (Nov/Dec) should be scored as at
    least as risky as the same order in a regular month — matching
    the seasonal pattern observed in Notebook 4's EDA.
    """
    base_order = {
        "total_price": 150.0,
        "total_freight": 20.0,
        "n_items": 2,
        "customer_state": "SP",
    }

    regular_month = pd.DataFrame(
        [{**base_order, "order_purchase_timestamp": "2026-06-15 10:00:00"}]
    )
    holiday_month = pd.DataFrame(
        [{**base_order, "order_purchase_timestamp": "2026-11-15 10:00:00"}]
    )

    late_idx = list(model.classes_).index("Late")
    p_regular = model.predict_proba(build_features(regular_month))[0, late_idx]
    p_holiday = model.predict_proba(build_features(holiday_month))[0, late_idx]

    assert p_holiday >= p_regular
