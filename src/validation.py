"""
Input validation for a new order before it enters the feature pipeline.

Checks that all required fields are present, have the right type,
and hold sensible values — before any feature engineering happens.
Raises a clear ValueError if something is wrong, instead of letting
a bad value silently break the model downstream.
"""

from datetime import datetime

# The exact raw fields the feature pipeline needs — matches the
# columns actually used to build the training table in Notebook 5.
REQUIRED_FIELDS = {
    "total_price": (int, float),
    "total_freight": (int, float),
    "n_items": int,
    "order_purchase_timestamp": (str, datetime),
    "customer_state": str,
}


def validate_order(order: dict) -> None:
    """
    Validate a single new-order record.

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

    Raises
    ------
    ValueError
        If a field is missing, has the wrong type, or an invalid value.
    """
    # 1. Check every required field is present
    missing = [f for f in REQUIRED_FIELDS if f not in order]
    if missing:
        raise ValueError(f"Missing required field(s): {missing}")

    # 2. Check every field has the expected type
    for field, expected_type in REQUIRED_FIELDS.items():
        if not isinstance(order[field], expected_type):
            raise ValueError(
                f"Field '{field}' has invalid type "
                f"{type(order[field]).__name__}, expected {expected_type}"
            )

    # 3. Sanity checks on values (matches ranges seen in Notebook 4's EDA)
    if order["total_price"] <= 0:
        raise ValueError("total_price must be greater than 0")

    if order["total_freight"] < 0:
        raise ValueError("total_freight cannot be negative")

    if order["n_items"] <= 0:
        raise ValueError("n_items must be greater than 0")

    if len(order["customer_state"]) != 2:
        raise ValueError(
            f"customer_state must be a 2-letter state code, "
            f"got '{order['customer_state']}'"
        )

    # 4. If the timestamp is a string, make sure it can actually be parsed
    if isinstance(order["order_purchase_timestamp"], str):
        try:
            datetime.fromisoformat(order["order_purchase_timestamp"])
        except ValueError:
            raise ValueError(
                f"order_purchase_timestamp '{order['order_purchase_timestamp']}' "
                f"is not a valid ISO date/time string"
            )