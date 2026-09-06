"""
Loads the trained model artifact.

The model is loaded once, from the exact path in config.yaml, and
is never re-trained or re-fit here — this module only ever reads
what Notebook 6 already produced.
"""

import joblib

from src.config import settings

_MODEL_PATH = settings.get("paths", "model")

# Loaded once, when this module is first imported.
_model = joblib.load(_MODEL_PATH)


def get_model():
    """Return the loaded model instance."""
    return _model