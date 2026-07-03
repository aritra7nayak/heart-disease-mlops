"""
Unit Tests — Model Training & Evaluation
Tests model building, metric computation, and output shapes.
"""

import os
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from train import compute_metrics, save_roc_curve, save_confusion_matrix, pick_best_model

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


@pytest.fixture
def binary_data():
    """Generate small binary classification dataset."""
    X, y = make_classification(
        n_samples=120, n_features=13, n_informative=8,
        n_redundant=2, random_state=42, n_classes=2
    )
    split = 96
    return X[:split], X[split:], y[:split], y[split:]


@pytest.fixture
def trained_lr(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    return lr, X_test, y_test


class TestComputeMetrics:
    def test_all_keys_present(self, trained_lr):
        model, X_test, y_test = trained_lr
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics

    def test_metrics_in_range(self, trained_lr):
        model, X_test, y_test = trained_lr
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        for val in metrics.values():
            assert 0.0 <= val <= 1.0, f"Metric out of range: {val}"

    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.85])
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_rounding(self, trained_lr):
        model, X_test, y_test = trained_lr
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        for val in metrics.values():
            # Should be rounded to 4 decimal places
            assert round(val, 4) == val


class TestSaveROCCurve:
    def test_creates_file(self, tmp_path, trained_lr):
        model, X_test, y_test = trained_lr
        y_prob = model.predict_proba(X_test)[:, 1]
        out = str(tmp_path / "roc.png")
        save_roc_curve(y_test, y_prob, "LogisticRegression", out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000  # should be a real image

    def test_file_not_empty(self, tmp_path, trained_lr):
        model, X_test, y_test = trained_lr
        y_prob = model.predict_proba(X_test)[:, 1]
        out = str(tmp_path / "roc2.png")
        save_roc_curve(y_test, y_prob, "Test Model", out)
        assert os.path.getsize(out) > 5000


class TestSaveConfusionMatrix:
    def test_creates_file(self, tmp_path, trained_lr):
        model, X_test, y_test = trained_lr
        y_pred = model.predict(X_test)
        out = str(tmp_path / "cm.png")
        save_confusion_matrix(y_test, y_pred, "LogisticRegression", out)
        assert os.path.exists(out)

    def test_file_not_empty(self, tmp_path, trained_lr):
        model, X_test, y_test = trained_lr
        y_pred = model.predict(X_test)
        out = str(tmp_path / "cm2.png")
        save_confusion_matrix(y_test, y_pred, "Test Model", out)
        assert os.path.getsize(out) > 5000


class TestPickBestModel:
    def test_picks_highest_roc_auc(self):
        results = [
            {"model_name": "lr", "roc_auc": 0.85, "accuracy": 0.80},
            {"model_name": "rf", "roc_auc": 0.92, "accuracy": 0.88},
            {"model_name": "xgb", "roc_auc": 0.91, "accuracy": 0.87},
        ]
        best = pick_best_model(results)
        assert best["model_name"] == "rf"
        assert best["roc_auc"] == 0.92

    def test_single_model(self):
        results = [{"model_name": "only", "roc_auc": 0.75, "accuracy": 0.70}]
        best = pick_best_model(results)
        assert best["model_name"] == "only"

    def test_returns_dict(self):
        results = [
            {"model_name": "a", "roc_auc": 0.80, "accuracy": 0.75},
            {"model_name": "b", "roc_auc": 0.90, "accuracy": 0.85},
        ]
        best = pick_best_model(results)
        assert isinstance(best, dict)


class TestModelIntegration:
    """Integration test: full train→predict pipeline."""

    def test_sklearn_model_predict_proba(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_model_output_binary(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_preprocessor_pipeline(self):
        """Test that preprocessor works end-to-end with feature arrays."""
        np.random.seed(1)
        n = 60
        df = pd.DataFrame({
            "age": np.random.randint(30, 75, n),
            "sex": np.random.randint(0, 2, n),
            "cp": np.random.randint(0, 4, n),
            "trestbps": np.random.randint(90, 180, n),
            "chol": np.random.randint(150, 400, n),
            "fbs": np.random.randint(0, 2, n),
            "restecg": np.random.randint(0, 3, n),
            "thalach": np.random.randint(80, 200, n),
            "exang": np.random.randint(0, 2, n),
            "oldpeak": np.round(np.random.uniform(0, 5, n), 1),
            "slope": np.random.randint(0, 3, n),
            "ca": np.random.randint(0, 4, n),
            "thal": np.random.randint(1, 4, n),
        })
        preprocessor = build_preprocessor()
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_t = preprocessor.fit_transform(X)
        assert X_t.shape[0] == n
        assert not np.isnan(X_t).any()

    def test_saved_model_loads_and_predicts(self):
        """Test that the saved best_model.joblib works correctly."""
        import joblib
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")
        if not os.path.exists(model_path):
            pytest.skip("best_model.joblib not found — run train.py first")

        model = joblib.load(model_path)
        dummy_X = np.random.rand(5, 13)  # 13 features
        preds = model.predict(dummy_X)
        assert len(preds) == 5
        assert set(preds).issubset({0, 1})
