"""
Light input cleaning — runs BEFORE validation.

Fixes small, predictable inconsistencies in incoming data (wrong
case, numeric values sent as strings) so that validation and
feature engineering always receive clean, well-typed data.
This does NOT fix invalid data (e.g. a negative price) — that is
validation's job. This only normalizes formatting.
"""


def clean_order(order: dict) -> dict:
    """
    Normalize a raw order dict before validation.

    Parameters
    ----------
    order : dict
        Raw order data, possibly with minor formatting issues.

    Returns
    -------
    dict
        A new dict with normalized values. The input dict is not
        modified in place.
    """
    cleaned = dict(order)

    # Uppercase the state code: "sp" -> "SP"
    if "customer_state" in cleaned and isinstance(cleaned["customer_state"], str):
        cleaned["customer_state"] = cleaned["customer_state"].strip().upper()

    # Coerce numeric-looking strings to real numbers: "149.90" -> 149.90
    for field in ("total_price", "total_freight"):
        if field in cleaned and isinstance(cleaned[field], str):
            try:
                cleaned[field] = float(cleaned[field])
            except ValueError:
                pass  # leave as-is; validation will catch the bad type

    if "n_items" in cleaned and isinstance(cleaned["n_items"], str):
        try:
            cleaned["n_items"] = int(cleaned["n_items"])
        except ValueError:
            pass

    # Trim whitespace from the timestamp string, if present
    if "order_purchase_timestamp" in cleaned and isinstance(
        cleaned["order_purchase_timestamp"], str
    ):
        cleaned["order_purchase_timestamp"] = cleaned[
            "order_purchase_timestamp"
        ].strip()

    return cleaned