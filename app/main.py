"""
FastAPI churn-prediction service.

Endpoints
---------
POST /predict   – returns churn probability for a customer
GET  /metrics   – Prometheus metrics (request count, latency histogram)
GET  /health    – liveness check
"""
import os
import pickle
import time
from pathlib import Path

import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

MODEL_PATH = os.getenv("MODEL_PATH", "app/model/model.pkl")
FEATURES = ["monthly_spend", "tenure_months", "support_tickets", "login_frequency"]

# ---------------------------------------------------------------------------
# Prometheus instruments
# ---------------------------------------------------------------------------
PREDICT_REQUESTS = Counter(
    "churn_predict_requests_total",
    "Total number of /predict requests",
    ["status"],
)
PREDICT_LATENCY = Histogram(
    "churn_predict_latency_seconds",
    "Latency of /predict requests in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# ---------------------------------------------------------------------------
# App + model loading
# ---------------------------------------------------------------------------
_model = None


def load_model():
    global _model
    model_file = Path(os.getenv("MODEL_PATH", MODEL_PATH))
    if not model_file.exists():
        raise RuntimeError(
            f"Model not found at {model_file}. "
            "Run scripts/train.py first or trigger the Airflow DAG."
        )
    with open(model_file, "rb") as f:
        _model = pickle.load(f)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Serves a RandomForest churn model trained with scikit-learn.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class CustomerFeatures(BaseModel):
    monthly_spend: float = Field(..., gt=0, description="Average monthly spend in USD")
    tenure_months: int = Field(..., ge=0, description="Months since customer joined")
    support_tickets: int = Field(..., ge=0, description="Support tickets opened (last 90 days)")
    login_frequency: int = Field(..., ge=0, description="Logins in the last 30 days")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float
    will_churn: bool
    model_version: str = "1.0"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        X = np.array([[
            customer.monthly_spend,
            customer.tenure_months,
            customer.support_tickets,
            customer.login_frequency,
        ]])
        prob = float(_model.predict_proba(X)[0][1])
        will_churn = prob >= 0.5
        PREDICT_REQUESTS.labels(status="success").inc()
        return PredictionResponse(churn_probability=round(prob, 4), will_churn=will_churn)
    except Exception as exc:
        PREDICT_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        PREDICT_LATENCY.observe(time.time() - start)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
