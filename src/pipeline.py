"""
Single entry point for the full inference pipeline.

Ties together cleaning, structural data validation (Great
Expectations), field-level validation, feature engineering, and
prediction into one function. This is what the CLI and the API
(built later) both call — neither of them talks to the individual
modules directly.

Logs every request (input, output, latency, model version) and
handles bad input and unexpected failures without letting the
whole service crash on a single bad request.
"""

import logging
import time

import pandas as pd

from src.config import settings
from src.data_validation import validate_dataframe
from src.exceptions import PipelineError
from src.logger import setup_logging
from src.predictor import predict_order
from src.preprocessing import clean_order

setup_logging()
logger = logging.getLogger(__name__)

_MODEL_VERSION = settings.get("model", "version")


def run_pipeline(order: dict) -> dict:
    """
    Run the full inference pipeline on a single raw order.

    Parameters
    ----------
    order : dict
        Raw order data as received from the caller (API, CLI, etc.).

    Returns
    -------
    dict
        {
            "prediction": "Late" | "On-time",
            "probability_late": float,
            "model_version": str,
        }

    Raises
    ------
    ValueError
        If the order is invalid (missing/bad fields, or fails a
        critical Great Expectations check) — the caller's fault,
        safe to report back with the exact message.
    PipelineError
        If something unexpected fails internally — the caller
        should show a generic error message, not these details.
    """
    start_time = time.perf_counter()
    logger.info(f"Received order for prediction | input={order}")

    try:
        cleaned = clean_order(order)

        # Structural data quality check (Great Expectations).
        # Critical issues (nulls, unknown category) reject the
        # request. Statistical outliers (unusual but valid ranges)
        # are logged as warnings and the request proceeds.
        gx_result = validate_dataframe(pd.DataFrame([cleaned]))
        if not gx_result["is_valid"]:
            raise ValueError(f"Order failed data quality checks: {gx_result['critical_failures']}")

        result = predict_order(cleaned)

    except ValueError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(f"Rejected invalid order | reason={e} | latency_ms={latency_ms:.2f}")
        raise

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(f"Unexpected pipeline failure | latency_ms={latency_ms:.2f}")
        raise PipelineError("An unexpected error occurred while processing the order.") from e

    latency_ms = (time.perf_counter() - start_time) * 1000
    result["model_version"] = _MODEL_VERSION

    logger.info(
        f"Prediction complete | output={result} | "
        f"latency_ms={latency_ms:.2f} | model_version={_MODEL_VERSION}"
    )

    return result
