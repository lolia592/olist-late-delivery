# Alerting policy

This document defines what conditions should trigger an alert to
the on-call engineer, once this service is deployed with a real
monitoring stack (Prometheus + Alertmanager, or equivalent).

No alerting infrastructure is wired up in this task — this is the
decision itself, written down, as required.

## What to alert on, and why

| Condition | Threshold | Why it matters |
|---|---|---|
| **Error rate** | `prediction_errors_total{error_type="internal_error"}` rate > 1% of requests over 5 minutes | A rising rate of *unexpected* failures (not bad input — actual bugs or infra issues) means the service is degrading. `invalid_input` errors are excluded — a spike there usually means a bad upstream caller, not a broken service. |
| **Latency** | p95 of `prediction_latency_seconds` > 1 second over 5 minutes | The pipeline normally responds in well under a second (observed ~100-750ms locally). A sustained p95 above 1s signals a performance regression before it becomes a full outage. |
| **Prediction drift** | Share of `predictions_by_class_total{prediction="Late"}` moves more than 15 percentage points away from its 7-day rolling average | The training data's "Late" rate was roughly 8-15%. A sudden large shift doesn't necessarily mean the model is wrong, but it does mean the input distribution has likely changed - worth a human look before trusting the predictions blindly. |
| **Service down** | `/health` fails to respond for more than 1 minute | The most basic and highest-priority alert: the service isn't reachable at all. |

## What does NOT alert

- Individual `invalid_input` errors (422s) — these are expected, routine rejections of malformed requests, not service problems.
- Single slow requests — only sustained latency (p95 over a window) triggers an alert, to avoid noise from one-off network blips.

## Why these four, and not more

Task 3 asks for a documented decision, not a fully wired alerting
system. These four conditions cover the categories that matter most
for a model-serving API: is it up, is it fast enough, is it failing
internally, and is its output distribution still trustworthy. Each
maps directly to a metric already exposed at `/metrics`.
