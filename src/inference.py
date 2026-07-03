"""
Inference module — loads model & preprocessor and runs predictions.
Can be used standalone or imported by the API.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


class HeartDiseasePredictor:
    def __init__(self, model_path: str = MODEL_PATH,
                 preprocessor_path: str = PREPROCESSOR_PATH):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        print(f"[Inference] Loaded {type(self.model).__name__}")

    def predict(self, raw_input: dict) -> dict:
        df = pd.DataFrame([raw_input])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X = self.preprocessor.transform(df)
        pred = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        return {
            "prediction": pred,
            "label": "Heart Disease" if pred == 1 else "No Disease",
            "confidence": round(float(max(proba)), 4),
            "prob_no_disease": round(float(proba[0]), 4),
            "prob_disease": round(float(proba[1]), 4),
        }

    def predict_batch(self, records: list) -> list:
        return [self.predict(r) for r in records]


if __name__ == "__main__":
    predictor = HeartDiseasePredictor()

    # Sample patient
    sample = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    result = predictor.predict(sample)
    print("\nSample prediction:")
    print(json.dumps(result, indent=2))
