
"""

Manual verification script (pre-pytest).



Confirms that src.feature_builder.build_features() produces output

IDENTICAL to what Notebook 5 actually saved — not just "close enough".

This is the core proof required by Task 3: "the pipeline must

reproduce exactly what your notebook produced on the same input."



Run directly with: python tests/verify_pipeline_matches_notebook.py

Will be converted into a proper pytest test in the testing step.

"""


import sys
from pathlib import Path

# Add the project root to Python's search path, so this script
# can find the `src` package regardless of which directory it's
# launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.feature_builder import build_features





def verify():

    raw = pd.read_parquet("data/processed/notebook3_train.parquet")

    actual = pd.read_parquet("data/processed/notebook5_train_features.parquet")



    sample_raw = raw[

        ["total_price", "total_freight", "n_items",

         "order_purchase_timestamp", "customer_state"]

    ].head(10)



    our_output = build_features(sample_raw)



    actual_sample = actual.drop(columns=["label"]).head(10)

    actual_sample = actual_sample[our_output.columns]



    shapes_match = our_output.shape == actual_sample.shape

    columns_match = list(our_output.columns) == list(actual_sample.columns)

    values_match = np.allclose(our_output.values, actual_sample.values, atol=1e-8)



    print("Shapes match:", shapes_match)

    print("Columns match:", columns_match)

    print("Values match exactly:", values_match)



    assert shapes_match, "Output shape does not match Notebook 5's output"

    assert columns_match, "Column names/order do not match Notebook 5's output"

    assert values_match, "Values differ from Notebook 5's actual output"



    print()

    print(" CONFIRMED: pipeline output is IDENTICAL to Notebook 5 output")





if __name__ == "__main__":

    verify()

