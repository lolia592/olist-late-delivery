"""
Unit tests for src.preprocessing.clean_order().
"""

from src.preprocessing import clean_order


def test_uppercases_lowercase_state():
    order = {"customer_state": "sp"}
    result = clean_order(order)
    assert result["customer_state"] == "SP"


def test_strips_whitespace_from_state():
    order = {"customer_state": "  SP  "}
    result = clean_order(order)
    assert result["customer_state"] == "SP"


def test_converts_numeric_strings_to_floats():
    order = {"total_price": "149.90", "total_freight": "18.50"}
    result = clean_order(order)
    assert result["total_price"] == 149.90
    assert isinstance(result["total_price"], float)
    assert result["total_freight"] == 18.50
    assert isinstance(result["total_freight"], float)


def test_converts_numeric_string_n_items_to_int():
    order = {"n_items": "3"}
    result = clean_order(order)
    assert result["n_items"] == 3
    assert isinstance(result["n_items"], int)


def test_strips_whitespace_from_timestamp():
    order = {"order_purchase_timestamp": "  2026-09-05 14:30:00  "}
    result = clean_order(order)
    assert result["order_purchase_timestamp"] == "2026-09-05 14:30:00"


def test_leaves_already_clean_values_unchanged():
    order = {
        "total_price": 149.90,
        "total_freight": 18.50,
        "n_items": 2,
        "customer_state": "SP",
        "order_purchase_timestamp": "2026-09-05 14:30:00",
    }
    result = clean_order(order)
    assert result == order


def test_does_not_mutate_the_original_dict():
    order = {"customer_state": "sp"}
    clean_order(order)
    # The original dict passed in must remain untouched.
    assert order["customer_state"] == "sp"


def test_ignores_unparseable_numeric_strings():
    # If a string can't be converted to a number, leave it as-is —
    # validation.py is responsible for rejecting it later, not this.
    order = {"total_price": "not-a-number"}
    result = clean_order(order)
    assert result["total_price"] == "not-a-number"