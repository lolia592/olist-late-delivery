"""
Single entry point for the full inference pipeline.

Ties together cleaning, validation, feature engineering, and
prediction into one function. This is what the CLI and the API
(built later) both call — neither of them talks to the individual
modules directly.
"""

import logging

from src.preprocessing import clean_order
from src.predictor import predict_order

logger = logging.getLogger(__name__)


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
        {"prediction": "Late" | "On-time", "probability_late": float}

    Raises
    ------
    ValueError
        If the order is invalid after cleaning.
    """
    logger.info("Received order for prediction")

    cleaned = clean_order(order)

    try:
        result = predict_order(cleaned)
    except ValueError as e:
        logger.warning(f"Rejected invalid order: {e}")
        raise

    logger.info(f"Prediction: {result['prediction']} (p_late={result['probability_late']:.3f})")
    return result