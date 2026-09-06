"""
Central logging setup.

Configures Python's built-in logging module ONCE, based on
config.yaml, so every module in the project logs in the same
format, at the same level, to both the console and a log file.

No other module should call logging.basicConfig() itself —
they should just do:
    import logging
    logger = logging.getLogger(__name__)
and rely on this module having already set things up.
"""

import logging
from pathlib import Path

from src.config import settings, PROJECT_ROOT

_configured = False


def setup_logging() -> None:
    """
    Configure the root logger once.

    Safe to call multiple times — only the first call actually
    does anything, so importing this module from several places
    never creates duplicate log handlers or duplicate log lines.
    """
    global _configured
    if _configured:
        return

    level_name = settings.get("logging", "level")
    log_format = settings.get("logging", "format")
    log_file = settings.get("logging", "file")

    level = getattr(logging, level_name.upper(), logging.INFO)

    log_path = PROJECT_ROOT / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _configured = True