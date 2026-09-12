"""
Custom exceptions for the inference pipeline.

Separating these from Python's built-in ValueError lets calling
code (the API, the CLI) tell the difference between:
  - a bad request from the caller (ValueError — their fault, safe
    to show the exact message back to them)
  - an unexpected internal failure (PipelineError — our fault, log
    the full details internally but never leak them to the caller)
"""


class PipelineError(Exception):
    """Raised when something unexpected fails inside the pipeline
    (not a bad input — a bug, a corrupted artifact, etc.)."""

