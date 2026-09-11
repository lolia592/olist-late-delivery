"""
Data validation using Great Expectations.

Checks that a batch of order data matches the expected schema
(column types, ranges, allowed categories, missing rates) BEFORE
it reaches the feature pipeline and the model.

Expectations are split into two severities:
  - CRITICAL: the data is structurally broken (nulls, unknown
    category) — the request must be rejected, since no valid
    features can be built from it.
  - WARNING: the data is a statistical outlier but still usable
    (e.g. an unusually large but plausible order) — logged, but
    the request proceeds to prediction.

This runs on a pandas DataFrame directly, using an in-memory
("ephemeral") Great Expectations context — no project folder or
config files are created on disk. The expectations themselves are
defined here in plain Python, based on the real ranges observed in
Notebook 4's EDA on the training data.
"""

import logging

import great_expectations as gx

logger = logging.getLogger(__name__)

# The 27 Brazilian state codes actually present in the training data
# (data/processed/notebook3_train.parquet).
VALID_STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
    "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
    "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]

# Structural problems: the data cannot produce valid features at all.
CRITICAL_EXPECTATION_TYPES = {
    "expect_column_values_to_not_be_null",
    "expect_column_values_to_be_in_set",
}

# Statistical outliers: unusual, but the model can still score them.
WARNING_EXPECTATION_TYPES = {
    "expect_column_values_to_be_between",
}


def validate_dataframe(df):
    """
    Validate a DataFrame of raw orders against expected data quality
    rules, before any feature engineering happens.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: total_price, total_freight, n_items,
        customer_state, order_purchase_timestamp.

    Returns
    -------
    dict
        {
            "is_valid": bool,                  # False if any CRITICAL check failed
            "critical_failures": list[str],    # must reject the request
            "warning_failures": list[str],     # logged, request still proceeds
        }
    """
    context = gx.get_context(mode="ephemeral")
    data_source = context.sources.add_pandas("runtime")
    data_asset = data_source.add_dataframe_asset("orders")
    batch = data_asset.build_batch_request(dataframe=df)

    validator = context.get_validator(
        batch_request=batch,
        create_expectation_suite_with_name="order_expectations",
    )

    # --- Column types & missing rates (CRITICAL) ---
    validator.expect_column_values_to_not_be_null("total_price")
    validator.expect_column_values_to_not_be_null("total_freight")
    validator.expect_column_values_to_not_be_null("n_items")
    validator.expect_column_values_to_not_be_null("customer_state")
    validator.expect_column_values_to_not_be_null("order_purchase_timestamp")

    # --- Allowed categories (CRITICAL) ---
    validator.expect_column_values_to_be_in_set("customer_state", VALID_STATES)

    # --- Ranges (WARNING — statistical outliers, based on real
    #     min/max seen in training data, with a wider ceiling) ---
    validator.expect_column_values_to_be_between(
        "total_price", min_value=0.01, max_value=20000
    )
    validator.expect_column_values_to_be_between(
        "total_freight", min_value=0, max_value=1500
    )
    validator.expect_column_values_to_be_between(
        "n_items", min_value=1, max_value=30
    )

    results = validator.validate()

    critical_failures = []
    warning_failures = []

    for r in results["results"]:
        if r["success"]:
            continue

        expectation_type = r["expectation_config"]["expectation_type"]
        column = r["expectation_config"]["kwargs"].get("column", "unknown")
        message = f"{column}: {expectation_type}"

        if expectation_type in CRITICAL_EXPECTATION_TYPES:
            critical_failures.append(message)
        else:
            warning_failures.append(message)

    is_valid = len(critical_failures) == 0

    if critical_failures:
        logger.warning(f"Critical data validation failure | issues={critical_failures}")
    if warning_failures:
        logger.warning(f"Data quality warning (non-blocking) | issues={warning_failures}")
    if is_valid and not warning_failures:
        logger.info("Data validation passed with no issues")

    return {
        "is_valid": is_valid,
        "critical_failures": critical_failures,
        "warning_failures": warning_failures,
    }