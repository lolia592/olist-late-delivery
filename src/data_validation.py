"""
Data validation using Great Expectations.

Checks that a batch of order data matches the expected schema
(column types, ranges, allowed categories, missing rates) BEFORE
it reaches the feature pipeline and the model.

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
            "success": bool,
            "failed_expectations": list[str],  # human-readable reasons
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

    # --- Column types & missing rates ---
    validator.expect_column_values_to_not_be_null("total_price")
    validator.expect_column_values_to_not_be_null("total_freight")
    validator.expect_column_values_to_not_be_null("n_items")
    validator.expect_column_values_to_not_be_null("customer_state")
    validator.expect_column_values_to_not_be_null("order_purchase_timestamp")

    # --- Ranges (based on real min/max seen in training data,
    #     with a wider ceiling to allow for future, larger orders) ---
    validator.expect_column_values_to_be_between(
        "total_price", min_value=0.01, max_value=20000
    )
    validator.expect_column_values_to_be_between(
        "total_freight", min_value=0, max_value=1500
    )
    validator.expect_column_values_to_be_between(
        "n_items", min_value=1, max_value=30
    )

    # --- Allowed categories ---
    validator.expect_column_values_to_be_in_set("customer_state", VALID_STATES)

    results = validator.validate()

    failed_expectations = [
        r["expectation_config"]["kwargs"].get("column", "unknown")
        + ": "
        + r["expectation_config"]["expectation_type"]
        for r in results["results"]
        if not r["success"]
    ]

    if not results["success"]:
        logger.warning(f"Data validation failed | failed_checks={failed_expectations}")
    else:
        logger.info("Data validation passed")

    return {
        "success": results["success"],
        "failed_expectations": failed_expectations,
    }