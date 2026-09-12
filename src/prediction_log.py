"""
Durable storage for every prediction made by the service.

Unlike the text log file (src/logger.py), which is for humans
debugging in real time, this is a structured, queryable record
of every prediction — including a placeholder for the real outcome,
to be filled in later once the order's actual delivery date is known.
This is what lets us measure real-world model accuracy over time,
not just accuracy on the original test set.
"""

import sqlite3
from datetime import datetime, timezone

from src.config import PROJECT_ROOT

_DB_PATH = PROJECT_ROOT / "predictions.db"


def _get_connection():
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT NOT NULL,
            total_price REAL,
            total_freight REAL,
            n_items INTEGER,
            order_purchase_timestamp TEXT,
            customer_state TEXT,
            prediction TEXT NOT NULL,
            probability_late REAL NOT NULL,
            model_version TEXT NOT NULL,
            actual_outcome TEXT
        )
        """
    )
    return conn


def log_prediction(order: dict, result: dict) -> None:
    """
    Persist one prediction to the predictions table.

    `actual_outcome` is left NULL — it gets filled in later, once
    the order's real delivery date is known, via a separate
    evaluation process (out of scope for this task, but this table
    is what makes that evaluation possible).
    """
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO predictions (
                logged_at, total_price, total_freight, n_items,
                order_purchase_timestamp, customer_state,
                prediction, probability_late, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                order.get("total_price"),
                order.get("total_freight"),
                order.get("n_items"),
                order.get("order_purchase_timestamp"),
                order.get("customer_state"),
                result["prediction"],
                result["probability_late"],
                result["model_version"],
            ),
        )
    conn.close()
