"""Prediction service. Every prediction and every piece of feedback is printed as one JSON
line; Azure Container Apps ships these lines to Log Analytics, where monitoring reads them."""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd

# Send request metrics to Application Insights when running in Azure.
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor()

from fastapi import FastAPI  # noqa: E402

from app.schemas import (  # noqa: E402
    Contribution,
    ExplanationResponse,
    FeedbackRequest,
    PredictionRequest,
    PredictionResponse,
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
STATE: dict = {}

# A dedicated logger writes one complete JSON line per event. The handler's lock stops
# lines from different request threads getting mixed together.
events = logging.getLogger("events")
events.setLevel(logging.INFO)
events.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
events.addHandler(_handler)


def log_event(event: dict) -> None:
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    events.info(json.dumps(event))


@asynccontextmanager
async def lifespan(_: FastAPI):
    model = joblib.load(MODEL_PATH)  # a pickle: only load the model file built by the pipeline
    STATE["model"] = model
    STATE["features"] = list(model.feature_names_in_)
    yield
    STATE.clear()


app = FastAPI(title="Sri Lanka export survival API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.get("/version")
def version() -> dict:
    return {"model_version": MODEL_VERSION}


def score(req: PredictionRequest) -> tuple[dict, pd.DataFrame, PredictionResponse]:
    features = req.model_dump(by_alias=True)
    X = pd.DataFrame([features])[STATE["features"]]
    probability = float(STATE["model"].predict_proba(X)[0, 1])
    response = PredictionResponse(
        request_id=str(uuid.uuid4()),
        probability=round(probability, 5),
        prediction=int(probability >= THRESHOLD),
        model_version=MODEL_VERSION,
    )
    return features, X, response


def log_prediction(response: PredictionResponse, features: dict, start: float) -> None:
    log_event(
        {
            "event": "prediction",
            **response.model_dump(include=set(PredictionResponse.model_fields)),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "features": features,
        }
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    start = time.perf_counter()
    features, _, response = score(req)
    log_prediction(response, features, start)
    return response


@app.post("/explain", response_model=ExplanationResponse)
def explain(req: PredictionRequest) -> ExplanationResponse:
    """The prediction plus why: the EBM is additive, so each term's contribution is exact."""
    start = time.perf_counter()
    features, X, response = score(req)
    model = STATE["model"]
    terms = sorted(
        zip(model.term_names_, model.eval_terms(X)[0], strict=True), key=lambda t: -abs(t[1])
    )
    explanation = ExplanationResponse(
        **response.model_dump(),
        baseline=round(float(model.intercept_[0]), 5),
        contributions=[Contribution(term=t, contribution=round(float(c), 5)) for t, c in terms],
    )
    log_prediction(explanation, features, start)
    return explanation


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    log_event({"event": "feedback", **req.model_dump()})
    return {"status": "recorded"}
