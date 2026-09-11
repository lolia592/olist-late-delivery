"""
Loads the trained model artifact from the MLflow Model Registry.

The model is loaded once, from the registry (not a local notebook
folder), using the model's "production" alias — so the service
always uses whichever version has been promoted, without any
code changes when a new version is promoted.
"""

import mlflow
import mlflow.sklearn

from src.config import settings

mlflow.set_tracking_uri(settings.get("mlflow", "tracking_uri"))

_MODEL_NAME = settings.get("mlflow", "model_name")
_MODEL_ALIAS = settings.get("mlflow", "model_alias")

# Loaded once, when this module is first imported.
_model = mlflow.sklearn.load_model(f"models:/{_MODEL_NAME}@{_MODEL_ALIAS}")


def get_model():
    """Return the loaded model instance."""
    return _model
