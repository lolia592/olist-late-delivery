"""
Database access — mirrors the joins and aggregations from Notebook 1.

Used for batch scoring: pulling existing orders from the database
and running them through the same inference pipeline used for new
orders. This module only reads from the database — it never writes,
and it never fits or trains anything.
"""

import pandas as pd
from sqlalchemy import create_engine

from src.config import settings

_db = settings.db
_DB_URL = f"postgresql://{_db['user']}:{_db['password']}@{_db['host']}:{_db['port']}/{_db['name']}"

# Created once; SQLAlchemy manages a connection pool internally,
# so we don't need to open/close a raw connection every call.
_engine = create_engine(_DB_URL)


def get_orders(order_ids: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch orders from the database, aggregated to one row per order —
    exactly like Notebook 1's df_ml, but only the columns the feature
    pipeline actually needs.

    Parameters
    ----------
    order_ids : list of str, optional
        If given, only fetch these specific orders. If None, fetch
        every order in the database (use with care — this can be
        a lot of rows).

    Returns
    -------
    pd.DataFrame
        One row per order with columns: order_id, total_price,
        total_freight, n_items, order_purchase_timestamp,
        customer_state.
    """
    query = """
        SELECT
            o.order_id,
            o.order_purchase_timestamp,
            c.customer_state,
            COUNT(oi.order_item_id) AS n_items,
            SUM(oi.price) AS total_price,
            SUM(oi.freight_value) AS total_freight
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_status != 'canceled'
    """

    params = {}
    if order_ids is not None:
        query += " AND o.order_id = ANY(%(order_ids)s)"
        params["order_ids"] = order_ids

    query += """
        GROUP BY o.order_id, o.order_purchase_timestamp, c.customer_state
    """

    return pd.read_sql(query, _engine, params=params or None)
