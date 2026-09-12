"""
Data quality tests for the processed train/val/test splits.

Checks the schema, value ranges, missing values, and — critically —
that there is no leakage of the same order_id across splits.
"""

import pandas as pd
import pytest

EXPECTED_COLUMNS = {
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "n_items",
    "total_price",
    "total_freight",
    "total_payment",
    "n_payments",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
    "label",
}


@pytest.fixture(scope="module")
def train_df():
    return pd.read_parquet("data/processed/notebook3_train.parquet")


@pytest.fixture(scope="module")
def val_df():
    return pd.read_parquet("data/processed/notebook3_val.parquet")


@pytest.fixture(scope="module")
def test_df():
    return pd.read_parquet("data/processed/notebook3_test.parquet")


# --- Schema ---


def test_train_has_expected_columns(train_df):
    assert set(train_df.columns) == EXPECTED_COLUMNS


def test_val_has_expected_columns(val_df):
    assert set(val_df.columns) == EXPECTED_COLUMNS


def test_test_has_expected_columns(test_df):
    assert set(test_df.columns) == EXPECTED_COLUMNS


# --- Ranges ---


def test_total_price_is_always_positive(train_df):
    assert (train_df["total_price"] > 0).all()


def test_n_items_is_always_at_least_one(train_df):
    assert (train_df["n_items"] >= 1).all()


def test_total_freight_is_never_negative(train_df):
    assert (train_df["total_freight"] >= 0).all()


# --- Missing values ---


def test_no_missing_values_in_key_columns(train_df):
    key_columns = [
        "total_price",
        "total_freight",
        "n_items",
        "customer_state",
        "order_purchase_timestamp",
    ]
    assert train_df[key_columns].isnull().sum().sum() == 0


def test_label_has_no_missing_values(train_df):
    assert train_df["label"].isnull().sum() == 0


def test_label_only_has_two_valid_values(train_df):
    assert set(train_df["label"].unique()) == {"Late", "On-time"}


# --- Leakage check: no order_id should appear in more than one split ---


def test_no_order_id_overlap_between_train_and_val(train_df, val_df):
    overlap = set(train_df["order_id"]) & set(val_df["order_id"])
    assert len(overlap) == 0


def test_no_order_id_overlap_between_train_and_test(train_df, test_df):
    overlap = set(train_df["order_id"]) & set(test_df["order_id"])
    assert len(overlap) == 0


def test_no_order_id_overlap_between_val_and_test(val_df, test_df):
    overlap = set(val_df["order_id"]) & set(test_df["order_id"])
    assert len(overlap) == 0


# --- Time-based split sanity: train should end before val/test start ---


def test_time_split_is_chronological(train_df, val_df, test_df):
    train_max = train_df["order_purchase_timestamp"].max()
    val_min = val_df["order_purchase_timestamp"].min()
    test_min = test_df["order_purchase_timestamp"].min()

    assert train_max <= val_min
    assert train_max <= test_min
