# src/api/metrics.py
# This file defines Prometheus metrics for the FastAPI service.
# It tracks total request count, request latency, loaded models count,
# and churn predictions by risk tier.
# The /metrics endpoint is scraped by Prometheus every 15 seconds.

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
import time

router = APIRouter()

# Counter increases only — tracks total number of API requests
REQUEST_COUNT = Counter(
    "neuralretail_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"]
)

# Histogram tracks how long each request takes
REQUEST_LATENCY = Histogram(
    "neuralretail_api_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# Gauge can go up and down — tracks currently loaded models
ACTIVE_MODELS = Gauge(
    "neuralretail_models_loaded",
    "Number of ML models currently loaded"
)

# Counter tracks churn predictions split by risk tier
CHURN_PREDICTIONS = Counter(
    "neuralretail_churn_predictions_total",
    "Total churn predictions made",
    ["risk_tier"]
)


@router.get("/metrics", response_class=PlainTextResponse)
def metrics():
    # Prometheus scrapes this endpoint to collect all metrics
    return generate_latest()


async def metrics_middleware(request, call_next):
    # This middleware runs on every request and records count and latency
    start_time = time.time()
    response   = await call_next(request)
    duration   = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(duration)

    return response