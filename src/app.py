"""
Heart Disease Prediction API
FastAPI application serving the trained model with monitoring and logging.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ── Logging setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Always use /tmp/logs in containers (writable by any user)
LOG_DIR = "/tmp/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "api_requests.log"), mode="a"),
    ],
)
logger = logging.getLogger("heart-disease-api")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

# ── Load artefacts at startup ─────────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info(f"Model loaded: {type(model).__name__}")
    logger.info(f"Preprocessor loaded: {type(preprocessor).__name__}")
except Exception as exc:
    logger.error(f"Failed to load artefacts: {exc}")
    model = preprocessor = None

# ── Prometheus-style metrics counters ─────────────────────────────────────────
_metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "predictions_positive": 0,
    "predictions_negative": 0,
    "latency_sum_ms": 0.0,
}

# ── Pydantic request schema ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age: float = Field(..., ge=20, le=100, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex: 1=male, 0=female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=80, le=220, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar >120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=60, le=210, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0.0, le=7.0, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=1, le=3, description="Thal: 1=normal, 2=fixed, 3=reversable")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 54,
                "sex": 1,
                "cp": 0,
                "trestbps": 130,
                "chol": 245,
                "fbs": 0,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.4,
                "slope": 1,
                "ca": 0,
                "thal": 2,
            }
        }


class PredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probability_no_disease: float
    probability_disease: float
    model_version: str
    timestamp: str


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "MLOps Assignment — BITS Pilani AIMLCZG523\n\n"
        "Predicts the presence of heart disease from patient health metrics "
        "using a trained XGBoost classifier."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    _metrics["requests_total"] += 1
    try:
        response = await call_next(request)
        elapsed_ms = (time.time() - start) * 1000
        _metrics["latency_sum_ms"] += elapsed_ms
        if response.status_code < 400:
            _metrics["requests_success"] += 1
        else:
            _metrics["requests_error"] += 1
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} [{elapsed_ms:.1f}ms]"
        )
        return response
    except Exception as exc:
        _metrics["requests_error"] += 1
        logger.error(f"Unhandled error: {exc}")
        raise


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Heart Disease Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health():
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": type(model).__name__}


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus-style plain text metrics."""
    total = max(_metrics["requests_total"], 1)
    avg_latency = _metrics["latency_sum_ms"] / total
    lines = [
        "# HELP api_requests_total Total API requests",
        "# TYPE api_requests_total counter",
        f"api_requests_total {_metrics['requests_total']}",
        "",
        "# HELP api_requests_success Successful requests",
        "# TYPE api_requests_success counter",
        f"api_requests_success {_metrics['requests_success']}",
        "",
        "# HELP api_requests_error Error requests",
        "# TYPE api_requests_error counter",
        f"api_requests_error {_metrics['requests_error']}",
        "",
        "# HELP api_avg_latency_ms Average response latency (ms)",
        "# TYPE api_avg_latency_ms gauge",
        f"api_avg_latency_ms {avg_latency:.2f}",
        "",
        "# HELP predictions_positive Positive predictions (disease)",
        "# TYPE predictions_positive counter",
        f"predictions_positive {_metrics['predictions_positive']}",
        "",
        "# HELP predictions_negative Negative predictions (no disease)",
        "# TYPE predictions_negative counter",
        f"predictions_negative {_metrics['predictions_negative']}",
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines))


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(data: PredictRequest):
    """
    Predict heart disease risk from patient features.

    Returns:
    - **prediction**: 0 = No Disease, 1 = Disease
    - **confidence**: model confidence (max class probability)
    - **probability_disease**: probability of heart disease
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build input dict
        raw = data.dict()

        # Transform via saved preprocessor
        import pandas as pd
        from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        df_in = pd.DataFrame([raw])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X = preprocessor.transform(df_in)

        # Predict
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]

        if pred == 1:
            _metrics["predictions_positive"] += 1
        else:
            _metrics["predictions_negative"] += 1

        logger.info(
            f"Prediction: {pred} | Confidence: {float(max(proba)):.3f} | "
            f"Input: {json.dumps(raw)}"
        )

        return PredictResponse(
            prediction=pred,
            prediction_label="Heart Disease Detected" if pred == 1 else "No Heart Disease",
            confidence=round(float(max(proba)), 4),
            probability_no_disease=round(float(proba[0]), 4),
            probability_disease=round(float(proba[1]), 4),
            model_version=type(model).__name__ + "_v1.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Return model metadata and feature descriptions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = {
        "model_type": type(model).__name__,
        "features": {
            "numeric": ["age", "trestbps", "chol", "thalach", "oldpeak"],
            "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        },
        "target": {"0": "No Heart Disease", "1": "Heart Disease Present"},
        "training_metrics": _load_summary(),
    }
    return info


def _load_summary():
    try:
        summary_path = os.path.join(BASE_DIR, "models", "experiment_summary.json")
        with open(summary_path) as f:
            s = json.load(f)
        return s.get("metrics", {})
    except Exception:
        return {}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
