"""
data_preprocessing.py
---------------------
Handles all data cleaning, encoding, and scaling steps for the
Student Performance Predictor pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """End-to-end preprocessing pipeline for student performance data."""

    NUMERIC_FEATURES = [
        "age", "study_hours", "attendance", "previous_grade",
        "family_income", "travel_time", "failures",
    ]

    CATEGORICAL_FEATURES = [
        "gender", "ethnicity", "parent_education", "parent_job",
        "extracurricular", "family_support", "internet_access",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_columns: list[str] = []
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load a CSV or Excel file into a DataFrame."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        logger.info("Loaded %d records with %d features from %s", *df.shape, filepath)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values and clip outliers."""
        df = df.copy()

        # Impute numeric columns with median
        for col in self.NUMERIC_FEATURES:
            if col in df.columns:
                median_val = df[col].median()
                missing = df[col].isna().sum()
                if missing:
                    logger.info("Imputing %d missing values in '%s' with median %.2f", missing, col, median_val)
                df[col].fillna(median_val, inplace=True)

        # Impute categorical columns with mode
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                mode_val = df[col].mode()[0]
                missing = df[col].isna().sum()
                if missing:
                    logger.info("Imputing %d missing values in '%s' with mode '%s'", missing, col, mode_val)
                df[col].fillna(mode_val, inplace=True)

        # Clip outliers using 1.5×IQR rule
        for col in self.NUMERIC_FEATURES:
            if col in df.columns:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df[col] = df[col].clip(lower, upper)

        logger.info("Data cleaning complete — shape: %s", df.shape)
        return df

    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """One-hot encode nominal and ordinal-encode ordered categoricals."""
        df = df.copy()

        # Ordinal mappings
        ordinal_maps = {
            "parent_education": {"None": 0, "Primary": 1, "Secondary": 2, "Graduate": 3, "Postgraduate": 4},
            "family_support":   {"None": 0, "Low": 1, "Medium": 2, "High": 3},
        }

        for col, mapping in ordinal_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        # Binary / nominal → one-hot
        one_hot_cols = [c for c in self.CATEGORICAL_FEATURES if c in df.columns and c not in ordinal_maps]
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

        if fit:
            self.feature_columns = [c for c in df.columns if c != "target"]

        return df

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Standardise numeric features (mean=0, std=1)."""
        if fit:
            self.scaler.fit(X)
            self.is_fitted = True
        return self.scaler.transform(X)

    def split_data(self, df: pd.DataFrame, target_col: str = "target", test_size: float = 0.2, random_state: int = 42):
        """Split into train/test sets and apply SMOTE to balance the training set."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

        # Balance training data with SMOTE
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info("After SMOTE — train size: %d", len(X_train_res))

        return X_train_res, X_test, y_train_res, y_test

    def full_pipeline(self, filepath: str, target_col: str = "target"):
        """Convenience method: load → clean → encode → scale → split."""
        df = self.load_data(filepath)
        df = self.clean_data(df)
        df = self.encode_features(df, fit=True)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_scaled = self.scale_features(X, fit=True)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_columns)
        X_scaled_df[target_col] = y.values

        return self.split_data(X_scaled_df, target_col=target_col)


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    print("DataPreprocessor ready. Call full_pipeline(filepath) to start.")
