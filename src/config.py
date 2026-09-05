"""
Central configuration loader.

Reads config/config.yaml and .env once, and exposes a single
`settings` object that every other module in the project imports
from — so there is exactly one source of truth for paths and
parameters, and no hardcoded values anywhere else in the code.
"""

from pathlib import Path
import os

import yaml
from dotenv import load_dotenv

# Resolve paths relative to this file's location, so the project
# works correctly no matter which directory it's launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def load_config() -> dict:
    """Read config.yaml and return it as a plain Python dict."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_env() -> None:
    """Load variables from .env into the process environment."""
    load_dotenv(ENV_PATH)


def get_db_credentials() -> dict:
    """Return database connection details from environment variables."""
    return {
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "name": os.getenv("DB_NAME"),
    }


class Settings:
    """Bundles config.yaml values and .env credentials in one place."""

    def __init__(self):
        load_env()
        self._config = load_config()
        self.db = get_db_credentials()

    def get(self, *keys):
        """
        Fetch a value from config.yaml by key path.
        Example: settings.get("paths", "model")
        """
        value = self._config
        for key in keys:
            value = value[key]
        return value


# Single shared instance — import this everywhere else:
#   from src.config import settings
settings = Settings()