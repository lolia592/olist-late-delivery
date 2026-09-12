"""
Prometheus metrics for the inference service.

Exposed at the /metrics endpoint (see app/main.py) so that a real
monitoring stack (Prometheus + Grafana) can scrape them in a
production deployment.

Three kinds of metrics, each answering a different question:
  - Counter: "how many times has X happened, ever?" (requests, errors)
  - Histogram: "what's the distribution of X?" (request latency)
  - Counter (labeled): "how many predictions fell into each class?"
    — this is what lets us detect prediction drift over time.
"""

from prometheus_client import Counter, Histogram

# Total number of prediction requests received.
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received",
)

# Total number of requests that ended in an error, broken down by type.
ERROR_COUNT = Counter(
    "prediction_errors_total",
    "Total number of prediction requests that failed",
    ["error_type"],  # "invalid_input" or "internal_error"
)

# Distribution of end-to-end request latency, in seconds.
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken to process a prediction request",
)

# How many predictions fell into each class ("Late" / "On-time").
# Watching how this ratio shifts over time is exactly how you'd
# spot prediction drift without needing the real outcome yet.
PREDICTION_COUNT = Counter(
    "predictions_by_class_total",
    "Total predictions made, broken down by predicted class",
    ["prediction"],  # "Late" or "On-time"
)
