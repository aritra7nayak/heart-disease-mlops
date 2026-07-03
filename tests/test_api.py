"""
Unit Tests — FastAPI Prediction API
Tests health endpoints, predict endpoint, validation, and error handling.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Check that required models exist before importing app
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_EXISTS = os.path.exists(os.path.join(MODELS_DIR, "best_model.joblib"))

if MODEL_EXISTS:
    from fastapi.testclient import TestClient
    from src.app import app
    client = TestClient(app)

VALID_SAMPLE = {
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

DISEASE_SAMPLE = {
    "age": 67,
    "sex": 1,
    "cp": 0,
    "trestbps": 160,
    "chol": 286,
    "fbs": 0,
    "restecg": 2,
    "thalach": 108,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 3,
    "thal": 2,
}


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
class TestHealthEndpoints:
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_status(self):
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert data["status"] == "running"

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_is_healthy(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_has_model_name(self):
        response = client.get("/health")
        data = response.json()
        assert "model" in data


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
class TestPredictEndpoint:
    def test_predict_returns_200(self):
        response = client.post("/predict", json=VALID_SAMPLE)
        assert response.status_code == 200

    def test_predict_has_required_fields(self):
        response = client.post("/predict", json=VALID_SAMPLE)
        data = response.json()
        required = [
            "prediction", "prediction_label", "confidence",
            "probability_no_disease", "probability_disease",
            "model_version", "timestamp"
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_prediction_is_binary(self):
        response = client.post("/predict", json=VALID_SAMPLE)
        data = response.json()
        assert data["prediction"] in [0, 1]

    def test_probabilities_sum_to_one(self):
        response = client.post("/predict", json=VALID_SAMPLE)
        data = response.json()
        total = data["probability_no_disease"] + data["probability_disease"]
        assert abs(total - 1.0) < 0.01

    def test_confidence_in_range(self):
        response = client.post("/predict", json=VALID_SAMPLE)
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_prediction_label_matches_prediction(self):
        response = client.post("/predict", json=VALID_SAMPLE)
        data = response.json()
        if data["prediction"] == 0:
            assert "No" in data["prediction_label"]
        else:
            assert "Disease" in data["prediction_label"]

    def test_timestamp_format(self):
        response = client.post("/predict", json=VALID_SAMPLE)
        data = response.json()
        assert "T" in data["timestamp"]  # ISO format
        assert "Z" in data["timestamp"]

    def test_disease_sample_prediction(self):
        """Older male with multiple risk factors should often predict disease."""
        response = client.post("/predict", json=DISEASE_SAMPLE)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1]  # just check it runs


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
class TestValidation:
    def test_missing_field_returns_422(self):
        incomplete = {k: v for k, v in VALID_SAMPLE.items() if k != "age"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_age_out_of_range_returns_422(self):
        bad = VALID_SAMPLE.copy()
        bad["age"] = 150  # too high
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_sex_invalid_value_returns_422(self):
        bad = VALID_SAMPLE.copy()
        bad["sex"] = 5  # invalid
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_extra_fields_allowed(self):
        """FastAPI ignores extra fields by default."""
        extra = VALID_SAMPLE.copy()
        extra["unknown_field"] = "whatever"
        response = client.post("/predict", json=extra)
        assert response.status_code == 200


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_counters(self):
        response = client.get("/metrics")
        text = response.text
        assert "api_requests_total" in text
        assert "predictions_positive" in text or "predictions_negative" in text

    def test_model_info_returns_200(self):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_features(self):
        response = client.get("/model-info")
        data = response.json()
        assert "features" in data
        assert "numeric" in data["features"]
        assert "categorical" in data["features"]


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
class TestBatchPredictions:
    """Test repeated calls for consistency."""

    def test_deterministic_predictions(self):
        r1 = client.post("/predict", json=VALID_SAMPLE)
        r2 = client.post("/predict", json=VALID_SAMPLE)
        assert r1.json()["prediction"] == r2.json()["prediction"]
        assert r1.json()["confidence"] == r2.json()["confidence"]

    def test_multiple_patients(self):
        patients = [VALID_SAMPLE, DISEASE_SAMPLE]
        for patient in patients:
            r = client.post("/predict", json=patient)
            assert r.status_code == 200
            data = r.json()
            assert data["prediction"] in [0, 1]
