"""
Single entry point for the full inference pipeline.

Ties together cleaning, validation, feature engineering, and
prediction into one function. This is what the CLI and the API
(built later) both call — neither of them talks to the individual
modules directly.

Logs every request (input, output, latency, model version) and
handles bad input and unexpected failures without letting the
whole service crash on a single bad request.
"""

import logging
import time

from src.logger import setup_logging
from src.preprocessing import clean_order
from src.predictor import predict_order
from src.exceptions import PipelineError
from src.config import settings

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
        If the order is invalid (missing/bad fields) — the caller's
        fault, safe to report back with the exact message.
    PipelineError
        If something unexpected fails internally — the caller
        should show a generic error message, not these details.
    """
    start_time = time.perf_counter()
    logger.info(f"Received order for prediction | input={order}")

    try:
        cleaned = clean_order(order)
        result = predict_order(cleaned)

    except ValueError as e:
        # Bad input — expected, log as a warning, not an error,
        # and let the caller see the exact reason.
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(
            f"Rejected invalid order | reason={e} | latency_ms={latency_ms:.2f}"
        )
        raise

    except Exception as e:
        # Anything else is unexpected: log full details internally,
        # but raise a safe, generic error for the caller.
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            f"Unexpected pipeline failure | latency_ms={latency_ms:.2f}"
        )
        raise PipelineError(
            "An unexpected error occurred while processing the order."
        ) from e

    latency_ms = (time.perf_counter() - start_time) * 1000
    result["model_version"] = _MODEL_VERSION

    logger.info(
        f"Prediction complete | output={result} | "
        f"latency_ms={latency_ms:.2f} | model_version={_MODEL_VERSION}"
    )

    return result