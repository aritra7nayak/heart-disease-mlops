#!/usr/bin/env python3
"""
Heart Disease UCI Dataset Download Script
Downloads from UCI Machine Learning Repository and saves locally.
"""

import urllib.request
import os
import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "heart_disease_raw.csv")
CLEANED_PATH = os.path.join(os.path.dirname(__file__), "heart_disease_cleaned.csv")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]


def download_dataset():
    print(f"Downloading Heart Disease UCI Dataset from:\n{DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, OUTPUT_PATH)
    print(f"Raw data saved to: {OUTPUT_PATH}")


def clean_dataset():
    df = pd.read_csv(OUTPUT_PATH, header=None, names=COLUMNS, na_values="?")
    print(f"\nOriginal shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # Drop rows with missing values (only 6 rows affected)
    df.dropna(inplace=True)
    print(f"\nCleaned shape: {df.shape}")

    # Binarize target: 0 = No disease, 1 = Disease
    df["target"] = (df["target"] > 0).astype(int)

    # Cast known integer columns
    int_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    df.to_csv(CLEANED_PATH, index=False)
    print(f"Cleaned data saved to: {CLEANED_PATH}")
    return df


if __name__ == "__main__":
    download_dataset()
    df = clean_dataset()
    print(f"\nSample rows:\n{df.head()}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
