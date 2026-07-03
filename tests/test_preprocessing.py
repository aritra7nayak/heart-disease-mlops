"""
Unit Tests — Data Preprocessing
Tests data loading, validation, pipeline construction, and transformations.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import (
    load_data, validate_dataframe, build_preprocessor,
    split_data, get_feature_names_out,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart_disease_cleaned.csv")


@pytest.fixture
def sample_df():
    """Minimal valid dataframe fixture."""
    np.random.seed(0)
    n = 50
    return pd.DataFrame({
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
        "target": np.random.randint(0, 2, n),
    })


class TestLoadData:
    def test_loads_csv_correctly(self):
        df = load_data(DATA_PATH)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_expected_columns(self):
        df = load_data(DATA_PATH)
        expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_missing_values(self):
        df = load_data(DATA_PATH)
        assert df.isnull().sum().sum() == 0, "Dataset should have no missing values"

    def test_correct_row_count(self):
        df = load_data(DATA_PATH)
        assert 200 <= len(df) <= 400, "Expected ~303 rows"


class TestValidateDataframe:
    def test_valid_df_passes(self, sample_df):
        validate_dataframe(sample_df)  # should not raise

    def test_missing_column_raises(self, sample_df):
        bad_df = sample_df.drop(columns=["age"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_dataframe(bad_df)

    def test_non_binary_target_raises(self, sample_df):
        bad_df = sample_df.copy()
        bad_df["target"] = np.random.randint(0, 5, len(bad_df))
        with pytest.raises(ValueError, match="binary"):
            validate_dataframe(bad_df)

    def test_target_is_binary(self):
        df = load_data(DATA_PATH)
        validate_dataframe(df)
        assert set(df[TARGET].unique()).issubset({0, 1})


class TestBuildPreprocessor:
    def test_returns_column_transformer(self, sample_df):
        from sklearn.compose import ColumnTransformer
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_fit_transform_shape(self, sample_df):
        preprocessor = build_preprocessor()
        X = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_t = preprocessor.fit_transform(X)
        # Output should have len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) columns
        expected_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
        assert X_t.shape == (len(sample_df), expected_cols)

    def test_no_nan_after_transform(self, sample_df):
        preprocessor = build_preprocessor()
        X = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_t = preprocessor.fit_transform(X)
        assert not np.isnan(X_t).any(), "Transformed data should have no NaNs"

    def test_scaler_centers_numeric(self, sample_df):
        preprocessor = build_preprocessor()
        X = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_t = preprocessor.fit_transform(X)
        # First len(NUMERIC_FEATURES) columns should be scaled (~0 mean)
        numeric_cols = X_t[:, :len(NUMERIC_FEATURES)]
        assert abs(numeric_cols.mean()) < 1.0, "Numeric columns should be roughly centered"

    def test_handles_missing_values(self, sample_df):
        df_missing = sample_df.copy()
        df_missing.loc[0, "age"] = np.nan
        df_missing.loc[1, "chol"] = np.nan
        preprocessor = build_preprocessor()
        X = df_missing[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_t = preprocessor.fit_transform(X)
        assert not np.isnan(X_t).any()


class TestSplitData:
    def test_split_sizes(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(sample_df)
        assert abs(len(X_test) / total - 0.2) < 0.1

    def test_stratification(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.2)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.15, "Stratification should preserve class balance"

    def test_no_overlap(self, sample_df):
        sample_df = sample_df.reset_index(drop=True)
        X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.2)
        assert len(X_train) + len(X_test) == len(sample_df)

    def test_reproducibility(self, sample_df):
        r1 = split_data(sample_df, test_size=0.2, random_state=42)
        r2 = split_data(sample_df, test_size=0.2, random_state=42)
        pd.testing.assert_frame_equal(r1[0].reset_index(drop=True),
                                       r2[0].reset_index(drop=True))


class TestFeatureNames:
    def test_feature_names_count(self):
        from sklearn.compose import ColumnTransformer
        preprocessor = build_preprocessor()
        names = get_feature_names_out(preprocessor)
        assert len(names) == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)

    def test_feature_names_content(self):
        preprocessor = build_preprocessor()
        names = get_feature_names_out(preprocessor)
        for f in NUMERIC_FEATURES:
            assert f in names
        for f in CATEGORICAL_FEATURES:
            assert f in names
