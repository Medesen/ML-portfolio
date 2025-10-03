import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from .constants import MODELS_DIR
from .drift import detect_drift

app = FastAPI(title="Churn Service", version="0.1.0")

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("predictions_total", "Total prediction requests")
PREDICTIONS_ERROR_TOTAL = Counter("predictions_error_total", "Total prediction errors")
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["endpoint", "method", "status_code"]
)

# Very simple token auth (set SERVICE_TOKEN in the env to enable)
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN")

# Global cache for model and metadata (loaded at startup)
_MODEL_CACHE: Optional[Any] = None
_METADATA_CACHE: Optional[Dict[str, Any]] = None


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

    @property
    def num_records(self) -> int:
        return len(self.records)

    def validate_records(self, max_records: int = 1000) -> None:
        """Validate prediction request."""
        if not self.records:
            raise HTTPException(status_code=400, detail="Empty records list")
        if len(self.records) > max_records:
            raise HTTPException(
                status_code=400,
                detail=f"Too many records: {len(self.records)} (max: {max_records})",
            )


class DriftCheckRequest(BaseModel):
    """Optional request body for drift endpoint to analyze batch data."""

    records: Optional[List[Dict[str, Any]]] = None


# Response Models
class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str


class BaselineInfo(BaseModel):
    """Baseline statistics summary."""

    num_numerical_features: int
    num_categorical_features: int
    target_distribution: Dict[str, float]


class DriftSchemaResponse(BaseModel):
    """Response model for GET /drift endpoint."""

    target_column: Optional[str] = None
    feature_columns: List[str] = []
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    decision_threshold: Optional[float] = None
    run_id: Optional[str] = None
    has_baseline_stats: bool
    baseline_info: Optional[BaselineInfo] = None


class PredictionResponse(BaseModel):
    """Response model for /predict endpoint."""

    predictions: List[int]
    probabilities: List[float]


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Record request latency for all endpoints."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    # Record latency with labels
    REQUEST_LATENCY.labels(
        endpoint=request.url.path, method=request.method, status_code=str(response.status_code)
    ).observe(duration)

    return response


@app.on_event("startup")
async def load_model_on_startup():
    """Load model and metadata once at startup for better performance."""
    global _MODEL_CACHE, _METADATA_CACHE

    from joblib import load

    model_path = MODELS_DIR / "churn_model.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"

    # Load model
    if model_path.exists():
        try:
            _MODEL_CACHE = load(model_path)
            logging.getLogger(__name__).info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load model: {e}")
    else:
        logging.getLogger(__name__).warning(f"Model not found at {model_path}")

    # Load metadata
    if meta_path.exists():
        try:
            _METADATA_CACHE = json.loads(meta_path.read_text())
            logging.getLogger(__name__).info("Metadata loaded successfully")
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load metadata: {e}")
            _METADATA_CACHE = {}
    else:
        logging.getLogger(__name__).warning("Metadata not found")
        _METADATA_CACHE = {}


def _auth_check(authorization: str | None) -> None:
    if SERVICE_TOKEN:
        if not authorization or authorization != f"Bearer {SERVICE_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")


def _get_model() -> Any:
    """Get cached model or raise exception if not loaded."""
    if _MODEL_CACHE is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return _MODEL_CACHE


def _get_metadata() -> Dict[str, Any]:
    """Get cached metadata."""
    return _METADATA_CACHE or {}


@app.get("/health", response_model=HealthResponse)
async def health(authorization: str | None = Header(default=None)):
    """Health check endpoint. Returns 'ok' if model and metadata loaded, 'degraded' otherwise."""
    _auth_check(authorization)
    model_ok = _MODEL_CACHE is not None
    meta_ok = _METADATA_CACHE is not None
    status = "ok" if (model_ok and meta_ok) else "degraded"
    return HealthResponse(status=status)


@app.get("/drift", response_model=DriftSchemaResponse)
async def drift_get(authorization: str | None = Header(default=None)):
    """GET /drift - Return schema and baseline information."""
    _auth_check(authorization)
    meta = _get_metadata()

    baseline_info = None
    if "reference_statistics" in meta:
        ref_stats = meta["reference_statistics"]
        baseline_info = BaselineInfo(
            num_numerical_features=len(ref_stats.get("numerical", {})),
            num_categorical_features=len(ref_stats.get("categorical", {})),
            target_distribution=ref_stats.get("target_distribution", {}),
        )

    return DriftSchemaResponse(
        target_column=meta.get("target_column"),
        feature_columns=meta.get("feature_columns", []),
        numeric_columns=meta.get("numeric_columns", []),
        categorical_columns=meta.get("categorical_columns", []),
        decision_threshold=meta.get("decision_threshold"),
        run_id=meta.get("run_id"),
        has_baseline_stats="reference_statistics" in meta,
        baseline_info=baseline_info,
    )


@app.post("/drift")
async def drift_post(
    req: DriftCheckRequest, authorization: str | None = Header(default=None)
) -> Dict[str, Any]:
    """POST /drift - Analyze drift on provided batch data."""
    _auth_check(authorization)
    from .features import align_features_to_training_schema, split_data

    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided for drift analysis")

    try:
        metadata = _get_metadata()

        # Process input data
        df = pd.DataFrame(req.records)
        X = split_data(df, training=False)
        X = align_features_to_training_schema(X, metadata=metadata)

        # Optional: compute predictions for prediction drift
        predictions = None
        try:
            model = _get_model()
            classes = model.named_steps["classifier"].classes_
            if 1 in set(classes):
                pos_index = int(np.where(classes == 1)[0][0])
                probs = model.predict_proba(X)[:, pos_index]
                threshold = float(metadata.get("decision_threshold", 0.5))
                predictions = (probs >= threshold).astype(int)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not compute predictions for drift: {e}")

        # Detect drift
        drift_report = detect_drift(
            data=X,
            metadata=metadata,
            predictions=predictions,
            numerical_threshold=float(os.getenv("DRIFT_THRESHOLD_NUMERICAL", "0.1")),
            categorical_threshold=float(os.getenv("DRIFT_THRESHOLD_CATEGORICAL", "0.1")),
            prediction_threshold=float(os.getenv("DRIFT_THRESHOLD_PREDICTION", "0.1")),
        )

        return drift_report

    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger(__name__).exception(f"Drift analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Drift analysis failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictRequest, authorization: str | None = Header(default=None)):
    """Predict churn for a batch of records. Returns binary predictions and probabilities."""
    _auth_check(authorization)
    from .features import align_features_to_training_schema, split_data

    # Validate request (max 1000 records by default, configurable via env)
    max_records = int(os.getenv("MAX_PREDICTION_RECORDS", "1000"))
    req.validate_records(max_records=max_records)

    try:
        # Use cached model (loaded at startup)
        model = _get_model()
        metadata = _get_metadata()

        df = pd.DataFrame(req.records)
        X = split_data(df, training=False)
        # Align features to training schema (add missing, drop extra, reorder)
        X = align_features_to_training_schema(X, metadata=metadata)

        classes = model.named_steps["classifier"].classes_
        if 1 not in set(classes):
            PREDICTIONS_ERROR_TOTAL.inc()
            raise HTTPException(status_code=500, detail="Model classes not aligned (no class '1')")

        pos_index = int(np.where(classes == 1)[0][0])
        probs = model.predict_proba(X)[:, pos_index]
        preds = (probs >= float(metadata.get("decision_threshold", 0.5))).astype(int)

        PREDICTIONS_TOTAL.inc()
        return PredictionResponse(
            predictions=preds.tolist(),
            probabilities=probs.tolist(),
        )
    except HTTPException:
        raise
    except Exception as e:
        PREDICTIONS_ERROR_TOTAL.inc()
        logging.getLogger(__name__).exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
