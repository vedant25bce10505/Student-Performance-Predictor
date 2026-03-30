"""
feature_engineering.py
-----------------------
Creates new features from existing ones and selects the most informative
ones for model training.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates and selects features for the student performance dataset."""

    def __init__(self, k_best: int = 20):
        self.k_best = k_best
        self.selector = None
        self.selected_features: list[str] = []

    # ------------------------------------------------------------------
    # Feature creation
    # ------------------------------------------------------------------

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the DataFrame."""
        df = df.copy()

        # --- Study Efficiency Ratio -----------------------------------
        # study hours per grade point (captures diminishing returns)
        if "study_hours" in df.columns and "previous_grade" in df.columns:
            df["study_efficiency_ratio"] = df["study_hours"] / (df["previous_grade"].replace(0, 1))

        # --- Attendance-Performance Index ----------------------------
        if "attendance" in df.columns and "previous_grade" in df.columns:
            df["attendance_performance_index"] = (
                (df["attendance"] / 100) * df["previous_grade"]
            )

        # --- Parental Involvement Score ------------------------------
        parental_cols = ["parent_education", "parent_job", "family_support"]
        available = [c for c in parental_cols if c in df.columns]
        if available:
            df["parental_involvement_score"] = df[available].sum(axis=1)

        # --- Study-Attendance Interaction ----------------------------
        if "study_hours" in df.columns and "attendance" in df.columns:
            df["study_attendance_interaction"] = df["study_hours"] * (df["attendance"] / 100)

        # --- Risk Score (high failures + low attendance) ------------
        if "failures" in df.columns and "attendance" in df.columns:
            df["risk_score"] = df["failures"] * (1 - df["attendance"] / 100)

        logger.info("Engineered features added. New shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = "mutual_info", fit: bool = True) -> pd.DataFrame:
        """Select the top-k most informative features."""
        score_func = mutual_info_classif if method == "mutual_info" else f_classif

        if fit:
            self.selector = SelectKBest(score_func=score_func, k=min(self.k_best, X.shape[1]))
            self.selector.fit(X, y)
            mask = self.selector.get_support()
            self.selected_features = X.columns[mask].tolist()
            logger.info("Selected %d features via %s", len(self.selected_features), method)

        return X[self.selected_features]

    def get_feature_importance_df(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Return a DataFrame of feature names and their mutual-info scores, sorted descending."""
        scores = mutual_info_classif(X, y, random_state=42)
        importance_df = pd.DataFrame({"feature": X.columns, "importance": scores})
        return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame, target_col: str = "target", fit: bool = True) -> tuple[pd.DataFrame, pd.Series]:
        """Create features then select top-k; return (X_selected, y)."""
        df = self.create_features(df)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        X_selected = self.select_features(X, y, fit=fit)
        return X_selected, y


if __name__ == "__main__":
    print("FeatureEngineer ready.")
