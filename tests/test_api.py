import math
import os

import joblib
import pytest
from fastapi.testclient import TestClient
from interpret.glassbox import ExplainableBoostingClassifier

from src.data import CAT_FEATURES, FEATURES
from tests.conftest import make_rows


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Train a tiny model on synthetic data so the API can be tested without the real model.
    df = make_rows(300)
    feature_types = ["nominal" if c in CAT_FEATURES else "auto" for c in FEATURES]
    model = ExplainableBoostingClassifier(
        feature_types=feature_types, interactions=0, outer_bags=1, max_rounds=50
    )
    model.fit(df[FEATURES], df["y"])
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    os.environ["MODEL_PATH"] = str(path)

    import importlib

    import app.main

    importlib.reload(app.main)  # pick up MODEL_PATH
    with TestClient(app.main.app) as test_client:
        yield test_client


@pytest.fixture
def payload():
    return make_rows(1).drop(columns=["y"]).iloc[0].to_dict()


def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"


def test_predict(client, payload):
    body = client.post("/predict", json=payload).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["request_id"]


def test_contract_rejects_share_given_as_percent(client, payload):
    payload["market_share"] = 35.0
    assert client.post("/predict", json=payload).status_code == 422


def test_contract_rejects_bad_destination_code(client, payload):
    payload["destination"] = "Sri Lanka"
    assert client.post("/predict", json=payload).status_code == 422


def test_feedback(client, payload):
    request_id = client.post("/predict", json=payload).json()["request_id"]
    response = client.post("/feedback", json={"request_id": request_id, "actual_outcome": 1})
    assert response.status_code == 200


def test_explain_adds_up_to_the_prediction(client, payload):
    body = client.post("/explain", json=payload).json()
    log_odds = body["baseline"] + sum(c["contribution"] for c in body["contributions"])
    assert abs(1 / (1 + math.exp(-log_odds)) - body["probability"]) < 1e-3
    sizes = [abs(c["contribution"]) for c in body["contributions"]]
    assert sizes == sorted(sizes, reverse=True)


def test_explain_uses_the_same_contract(client, payload):
    payload["market_share"] = 35.0
    assert client.post("/explain", json=payload).status_code == 422
