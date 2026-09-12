"""
Feature engineering pipeline — mirrors Notebook 5 exactly.

Takes validated raw order data and produces the exact feature table
the model expects: same columns, same order, same transformations.
No transformer is ever re-fit here — everything is loaded from the
fitted artifacts saved in Notebook 5.
"""

import joblib
import numpy as np
import pandas as pd

from src.config import settings

# Load fitted artifacts ONCE, when this module is first imported.
# These are never re-fit — only ever used to transform new data.
_encoder = joblib.load(settings.get("paths", "state_encoder"))
_rare_states = joblib.load(settings.get("paths", "rare_states"))
_feature_names = joblib.load(settings.get("paths", "feature_names"))

_HOLIDAY_MONTHS = settings.get("features", "holiday_months")
_LOG_COLUMNS = settings.get("features", "numeric_log_columns")


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors Notebook 5's add_time_features()."""
    df = df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["is_holiday_season"] = (
        df["order_purchase_timestamp"].dt.month.isin(_HOLIDAY_MONTHS).astype(int)
    )
    return df


def _group_states(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors Notebook 5's group_states() — uses the saved rare-state list."""
    df = df.copy()
    df["customer_state_grouped"] = np.where(
        df["customer_state"].isin(_rare_states), "Other", df["customer_state"]
    )
    return df


def _add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors Notebook 5's log1p transform.

    NOTE: no scaling is applied here on purpose — the saved StandardScaler
    was never actually used in the training table built in Notebook 5/6.
    See config.yaml `features.apply_scaler` and the README for details.
    """
    df = df.copy()
    for col in _LOG_COLUMNS:
        df[f"{col}_log"] = np.log1p(df[col])
    return df


def build_features(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Build the exact feature table the model was trained on.

    Parameters
    ----------
    orders : pd.DataFrame
        One row per order, with columns: total_price, total_freight,
        n_items, order_purchase_timestamp, customer_state.

    Returns
    -------
    pd.DataFrame
        Feature table with exactly the 31 columns, in the exact order,
        the model expects.
    """
    df = orders.copy()
    df = _add_time_features(df)
    df = _group_states(df)
    df = _add_log_features(df)

    # Apply the SAVED encoder — transform only, never fit.
    encoded = _encoder.transform(df[["customer_state_grouped"]])
    encoded_df = pd.DataFrame(encoded, columns=_encoder.get_feature_names_out(), index=df.index)

    numeric_part = df[
        [f"{c}_log" for c in _LOG_COLUMNS]
        + ["purchase_month", "purchase_dayofweek", "is_holiday_season"]
    ]

    features = pd.concat([numeric_part, encoded_df], axis=1)

    # Final safety net: force the exact column order the model expects.
    features = features[_feature_names]

    return features
