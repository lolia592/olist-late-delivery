"""
Unit tests for src.validation.validate_order().
"""

import pytest

from src.validation import validate_order


def _valid_order(**overrides):
    order = {
        "total_price": 149.90,
        "total_freight": 18.50,
        "n_items": 2,
        "order_purchase_timestamp": "2026-09-05 14:30:00",
        "customer_state": "SP",
    }
    order.update(overrides)
    return order


def test_valid_order_passes():
    # Should not raise.
    validate_order(_valid_order())


def test_missing_field_is_rejected():
    order = _valid_order()
    del order["customer_state"]
    with pytest.raises(ValueError, match="Missing required field"):
        validate_order(order)


def test_wrong_type_is_rejected():
    order = _valid_order(total_price="not a number")
    with pytest.raises(ValueError, match="invalid type"):
        validate_order(order)


def test_negative_price_is_rejected():
    order = _valid_order(total_price=-10)
    with pytest.raises(ValueError, match="greater than 0"):
        validate_order(order)


def test_negative_freight_is_rejected():
    order = _valid_order(total_freight=-1)
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_order(order)


def test_zero_items_is_rejected():
    order = _valid_order(n_items=0)
    with pytest.raises(ValueError, match="greater than 0"):
        validate_order(order)


def test_invalid_state_code_length_is_rejected():
    order = _valid_order(customer_state="SAO")
    with pytest.raises(ValueError, match="2-letter state code"):
        validate_order(order)


def test_unparseable_timestamp_is_rejected():
    order = _valid_order(order_purchase_timestamp="not-a-date")
    with pytest.raises(ValueError, match="not a valid ISO"):
        validate_order(order)


def test_nan_price_is_rejected():
    order = _valid_order(total_price=float("nan"))
    with pytest.raises(ValueError, match="cannot be NaN"):
        validate_order(order)


def test_nan_freight_is_rejected():
    order = _valid_order(total_freight=float("nan"))
    with pytest.raises(ValueError, match="cannot be NaN"):
        validate_order(order)


def test_zero_freight_is_allowed():
    # Free shipping (0.0) is valid — only negative values are rejected.
    validate_order(_valid_order(total_freight=0.0))