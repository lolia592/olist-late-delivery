"""
Unit tests for src.feature_builder.build_features().
"""

import numpy as np
import pandas as pd

from src.feature_builder import build_features


def _make_order(**overrides):
    """A single valid order, with any field overridable for a specific test."""
    order = {
        "total_price": 149.90,
        "total_freight": 18.50,
        "n_items": 2,
        "order_purchase_timestamp": "2026-09-05 14:30:00",
        "customer_state": "SP",
    }
    order.update(overrides)
    return pd.DataFrame([order])


def test_output_has_exactly_31_columns():
    features = build_features(_make_order())
    assert features.shape == (1, 31)


def test_log_transform_is_correct():
    features = build_features(_make_order(total_price=149.90))
    expected = np.log1p(149.90)
    assert np.isclose(features["total_price_log"].iloc[0], expected)


def test_time_features_extracted_correctly():
    features = build_features(_make_order(order_purchase_timestamp="2026-11-15 10:00:00"))
    assert features["purchase_month"].iloc[0] == 11
    assert features["is_holiday_season"].iloc[0] == 1


def test_non_holiday_month_flagged_correctly():
    features = build_features(_make_order(order_purchase_timestamp="2026-06-15 10:00:00"))
    assert features["purchase_month"].iloc[0] == 6
    assert features["is_holiday_season"].iloc[0] == 0


def test_known_state_one_hot_encoded_correctly():
    features = build_features(_make_order(customer_state="SP"))
    assert features["customer_state_grouped_SP"].iloc[0] == 1.0
    # Every other state column must be 0.
    other_state_cols = [
        c
        for c in features.columns
        if c.startswith("customer_state_grouped_") and c != "customer_state_grouped_SP"
    ]
    assert (features[other_state_cols] == 0.0).all().all()


def test_rare_or_unknown_state_maps_to_other():
    # A state code not in the training data's frequent set should
    # fall into the "Other" bucket rather than crash.
    features = build_features(_make_order(customer_state="AC"))
    # AC is a known rare state grouped into "Other" in Notebook 5.
    assert features["customer_state_grouped_Other"].iloc[0] == 1.0


def test_column_order_matches_saved_feature_names():
    import joblib

    expected_order = joblib.load("models/notebook5_feature_names.pkl")
    features = build_features(_make_order())
    assert list(features.columns) == expected_order
