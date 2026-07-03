"""
Heart Disease Dataset Preprocessing Pipeline
Handles feature engineering, scaling, encoding, and train/test splits.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# ── Feature definitions ──────────────────────────────────────────────────────
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1=male, 0=female)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mmHg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar >120 mg/dl (1=true)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1=yes)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Number of major vessels coloured (0-3)",
    "thal": "Thal: 1=normal, 2=fixed defect, 3=reversable defect",
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load the cleaned CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"[Preprocessing] Loaded {len(df)} rows × {df.shape[1]} cols from {filepath}")
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Basic validation of the loaded dataframe."""
    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[TARGET].nunique() != 2:
        raise ValueError("Target must be binary (0/1)")
    print(f"[Preprocessing] Validation passed. Target balance: "
          f"{df[TARGET].value_counts().to_dict()}")


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical features are already ordinally encoded integers in this dataset
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """Get ordered feature names after transformation."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets, stratified on target."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Preprocessing] Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    """Persist fitted preprocessor to disk."""
    joblib.dump(preprocessor, path)
    print(f"[Preprocessing] Saved preprocessor to {path}")


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load fitted preprocessor from disk."""
    preprocessor = joblib.load(path)
    print(f"[Preprocessing] Loaded preprocessor from {path}")
    return preprocessor


def preprocess_input(raw_input: dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """Transform a single raw API input dict into model-ready array."""
    df = pd.DataFrame([raw_input])
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return preprocessor.transform(X)


def run_preprocessing(data_path: str, output_dir: str, test_size: float = 0.2):
    """
    Full preprocessing pipeline: load → validate → split → fit → save.
    Returns (X_train, X_test, y_train, y_test, preprocessor).
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_path)
    validate_dataframe(df)

    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    save_preprocessor(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    # Persist splits for reproducibility
    np.save(os.path.join(output_dir, "X_train.npy"), X_train_t)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_t)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test.values)

    print(f"[Preprocessing] All artifacts saved to {output_dir}")
    return X_train_t, X_test_t, y_train.values, y_test.values, preprocessor


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(__file__))
    run_preprocessing(
        data_path=os.path.join(BASE, "data", "heart_disease_cleaned.csv"),
        output_dir=os.path.join(BASE, "models"),
    )
