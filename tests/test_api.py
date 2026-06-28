"""
Integration tests for the FastAPI churn prediction service.

A real (tiny) model is trained in the fixture so tests don't depend on
a pre-existing model file on disk.
"""
import os
import pickle

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Fixture: train a tiny model and point the app at it
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def trained_model(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model")
    model_path = model_dir / "model.pkl"

    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    X = np.array([[100, 12, 2, 10], [50, 1, 15, 1]])
    y = np.array([0, 1])
    clf.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    os.environ["MODEL_PATH"] = str(model_path)
    yield model_path


@pytest.fixture(scope="session")
def client(trained_model):
    # Import after env var is set so MODEL_PATH is resolved correctly on first import
    import app.main as main_module

    main_module.load_model()
    return TestClient(main_module.app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_loaded"] is True


def test_predict_returns_valid_shape(client):
    payload = {
        "monthly_spend": 120.0,
        "tenure_months": 24,
        "support_tickets": 3,
        "login_frequency": 15,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "churn_probability" in data
    assert "will_churn" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["will_churn"], bool)


def test_predict_high_risk_customer(client):
    """Customer with many tickets and few logins should lean toward churn."""
    payload = {
        "monthly_spend": 30.0,
        "tenure_months": 1,
        "support_tickets": 18,
        "login_frequency": 0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["will_churn"] is True


def test_predict_invalid_payload(client):
    resp = client.post("/predict", json={"monthly_spend": -10})
    assert resp.status_code == 422  # Pydantic validation error


def test_metrics_endpoint(client):
    # Hit /predict first so counters are non-zero
    client.post("/predict", json={
        "monthly_spend": 100.0,
        "tenure_months": 12,
        "support_tickets": 1,
        "login_frequency": 10,
    })
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"churn_predict_requests_total" in resp.content
