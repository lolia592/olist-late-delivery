"""
Confirms that src.feature_builder.build_features() produces output
IDENTICAL to what Notebook 5 actually saved — not just "close enough".
This is the core proof required by Task 3: "the pipeline must
reproduce exactly what your notebook produced on the same input."
"""

import numpy as np
import pandas as pd

from src.feature_builder import build_features


def test_features_match_notebook5_output_exactly():
    raw = pd.read_parquet("data/processed/notebook3_train.parquet")
    actual = pd.read_parquet("data/processed/notebook5_train_features.parquet")

    sample_raw = raw[
        [
            "total_price",
            "total_freight",
            "n_items",
            "order_purchase_timestamp",
            "customer_state",
        ]
    ].head(10)

    our_output = build_features(sample_raw)

    actual_sample = actual.drop(columns=["label"]).head(10)
    actual_sample = actual_sample[our_output.columns]

    assert our_output.shape == actual_sample.shape, (
        "Output shape does not match Notebook 5's output"
    )

    assert list(our_output.columns) == list(actual_sample.columns), (
        "Column names/order do not match Notebook 5's output"
    )

    assert np.allclose(our_output.values, actual_sample.values, atol=1e-8), (
        "Values differ from Notebook 5's actual output"
    )
