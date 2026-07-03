"""
Heart Disease Model Training with MLflow Experiment Tracking
Trains Logistic Regression, Random Forest, and XGBoost classifiers
with hyperparameter tuning and full MLflow logging.
"""

import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import run_preprocessing, NUMERIC_FEATURES, CATEGORICAL_FEATURES

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease_cleaned.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
MLFLOW_DB = os.path.join(MLRUNS_DIR, "mlflow.db")
# Use SQLite backend — avoids file:// URI issues on Windows paths with spaces
MLFLOW_URI = "sqlite:///" + MLFLOW_DB.replace("\\", "/")
EXPERIMENT_NAME = "heart-disease-classification"
RANDOM_STATE = 42
CV_FOLDS = 5

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)


# ── Metric helpers ────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def save_roc_curve(y_true, y_prob, model_name: str, output_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, model_name: str, output_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(importances, feature_names: list, model_name: str, output_path: str):
    idx = np.argsort(importances)[::-1][:13]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(idx)))[::-1]
    bars = ax.barh(range(len(idx)), importances[idx][::-1], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"Feature Importances — {model_name}", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Model definitions & param grids ──────────────────────────────────────────
MODELS = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
        },
    },
}


# ── Training loop ─────────────────────────────────────────────────────────────
def train_and_log(model_name: str, config: dict,
                  X_train, y_train, X_test, y_test,
                  feature_names: list) -> dict:
    """Train, tune, evaluate and log one model with MLflow."""

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name) as run:
        print(f"\n{'='*60}")
        print(f"  Training: {model_name}")
        print(f"{'='*60}")

        # ── GridSearch ──────────────────────────────────────────────
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(
            config["estimator"],
            config["param_grid"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_

        print(f"  Best params: {gs.best_params_}")
        print(f"  CV AUC:      {gs.best_score_:.4f}")

        # ── Cross-validation scores ─────────────────────────────────
        cv_scores = cross_val_score(best_model, X_train, y_train,
                                    cv=cv, scoring="roc_auc")

        # ── Test evaluation ─────────────────────────────────────────
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        print(f"  Test metrics: {metrics}")
        print(classification_report(y_test, y_pred,
                                    target_names=["No Disease", "Disease"]))

        # ── MLflow logging ──────────────────────────────────────────
        # params
        mlflow.log_params(gs.best_params_)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("cv_folds", CV_FOLDS)

        # metrics
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_roc_auc_mean", float(np.mean(cv_scores)))
        mlflow.log_metric("cv_roc_auc_std", float(np.std(cv_scores)))
        mlflow.log_metric("best_cv_roc_auc", gs.best_score_)

        # ── Plots ───────────────────────────────────────────────────
        tmp = os.path.join(MODEL_DIR, "tmp_plots")
        os.makedirs(tmp, exist_ok=True)

        roc_path = os.path.join(tmp, f"{model_name}_roc.png")
        cm_path = os.path.join(tmp, f"{model_name}_cm.png")
        save_roc_curve(y_test, y_prob, model_name, roc_path)
        save_confusion_matrix(y_test, y_pred, model_name, cm_path)
        mlflow.log_artifact(roc_path, "plots")
        mlflow.log_artifact(cm_path, "plots")

        # Feature importance
        if hasattr(best_model, "feature_importances_"):
            fi_path = os.path.join(tmp, f"{model_name}_fi.png")
            save_feature_importance(
                best_model.feature_importances_,
                feature_names, model_name, fi_path
            )
            mlflow.log_artifact(fi_path, "plots")
        elif hasattr(best_model, "coef_"):
            coefs = np.abs(best_model.coef_[0])
            fi_path = os.path.join(tmp, f"{model_name}_coef.png")
            save_feature_importance(coefs, feature_names, model_name, fi_path)
            mlflow.log_artifact(fi_path, "plots")

        # ── Save model ──────────────────────────────────────────────
        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(best_model, model_path)
        # Use xgboost flavour for XGBoost, sklearn flavour for others
        if isinstance(best_model, XGBClassifier):
            import mlflow.xgboost as mlflow_xgb
            mlflow_xgb.log_model(best_model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(best_model, artifact_path="model")
        mlflow.log_artifact(model_path)

        run_id = run.info.run_id

    result = {
        "model_name": model_name,
        "run_id": run_id,
        "best_params": gs.best_params_,
        **metrics,
        "cv_roc_auc_mean": float(np.mean(cv_scores)),
    }
    return result


def pick_best_model(results: list) -> dict:
    """Select best model by ROC-AUC."""
    return max(results, key=lambda r: r["roc_auc"])


def save_best_model_info(best: dict, preprocessor, results: list):
    """Copy best model artefact and save summary."""
    src = os.path.join(MODEL_DIR, f"{best['model_name']}.joblib")
    best_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(joblib.load(src), best_path)

    summary = {
        "best_model": best["model_name"],
        "metrics": {k: v for k, v in best.items()
                    if k not in ("model_name", "run_id", "best_params")},
        "best_params": best["best_params"],
        "all_results": results,
    }
    summary_path = os.path.join(MODEL_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  BEST MODEL : {best['model_name']}")
    print(f"  ROC-AUC    : {best['roc_auc']:.4f}")
    print(f"  Accuracy   : {best['accuracy']:.4f}")
    print(f"  F1-Score   : {best['f1']:.4f}")
    print(f"  Summary    : {summary_path}")
    print(f"{'='*60}\n")


def main():
    # ── Preprocessing ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, preprocessor = run_preprocessing(
        data_path=DATA_PATH,
        output_dir=MODEL_DIR,
        test_size=0.2,
    )
    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # ── Train all models ─────────────────────────────────────────────
    results = []
    for name, cfg in MODELS.items():
        result = train_and_log(
            model_name=name,
            config=cfg,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            feature_names=feature_names,
        )
        results.append(result)

    # ── Pick & save best ─────────────────────────────────────────────
    best = pick_best_model(results)
    save_best_model_info(best, preprocessor, results)

    return results, best


if __name__ == "__main__":
    main()
