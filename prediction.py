"""
prediction.py
-------------
High-level interface for making student performance predictions using
the best trained model (Gradient Boosting by default).
"""

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
GRADE_LABELS = {0: "F", 1: "D", 2: "C", 3: "B", 4: "A"}


class StudentPerformancePredictor:
    """Load a trained model and predict student performance."""

    def __init__(self, model_name: str = "gradient_boosting"):
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.preprocessor = None  # loaded lazily when needed

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, name: str):
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model '{name}' not found at {path}. "
                "Run model_training.py first to train the models."
            )
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model: %s", name)
        return model

    def switch_model(self, model_name: str):
        """Hot-swap the underlying model without creating a new instance."""
        self.model = self._load_model(model_name)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, student_data: dict) -> dict:
        """
        Predict grade category for a single student.

        Parameters
        ----------
        student_data : dict
            Keys should match the expected feature names.

        Returns
        -------
        dict with keys: prediction (str), confidence (float), grade_probabilities (dict)
        """
        df = pd.DataFrame([student_data])
        df = self._validate_and_fill(df)
        X = self._prepare_features(df)

        pred_idx = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        grade = GRADE_LABELS.get(int(pred_idx), str(pred_idx))
        confidence = round(float(probabilities.max()) * 100, 2)
        grade_probs = {GRADE_LABELS.get(i, str(i)): round(float(p) * 100, 2)
                       for i, p in enumerate(probabilities)}

        result = {
            "prediction": grade,
            "confidence": confidence,
            "grade_probabilities": grade_probs,
            "model_used": self.model_name,
            "pass_fail": "Pass" if grade in ("A", "B", "C") else "Fail",
        }
        logger.info("Prediction: %s (%.1f%% confidence)", grade, confidence)
        return result

    def predict_batch(self, filepath: str) -> pd.DataFrame:
        """Predict grades for all rows in a CSV file."""
        df = pd.read_csv(filepath)
        X = self._prepare_features(df)
        preds = self.model.predict(X)
        proba = self.model.predict_proba(X)

        df["predicted_grade"] = [GRADE_LABELS.get(int(p), str(p)) for p in preds]
        df["confidence"] = (proba.max(axis=1) * 100).round(2)
        df["pass_fail"] = df["predicted_grade"].apply(lambda g: "Pass" if g in ("A", "B", "C") else "Fail")
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_and_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic validation: fill missing expected fields with sensible defaults."""
        defaults = {
            "age": 18,
            "study_hours": 3,
            "attendance": 75,
            "previous_grade": 60,
            "failures": 0,
            "travel_time": 1,
            "family_income": 3,
            "gender": "Male",
            "ethnicity": "Unknown",
            "parent_education": "Secondary",
            "parent_job": "Other",
            "extracurricular": "No",
            "family_support": "Medium",
            "internet_access": "Yes",
        }
        for col, default in defaults.items():
            if col not in df.columns:
                logger.warning("Missing field '%s'; using default: %s", col, default)
                df[col] = default
        return df

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the same preprocessing as during training."""
        # In a production pipeline, load the fitted preprocessor from disk.
        # Here we do a minimal inline transform for demonstration.
        numeric_cols = ["age", "study_hours", "attendance", "previous_grade",
                        "failures", "travel_time", "family_income"]
        ordinal_map = {
            "parent_education": {"None": 0, "Primary": 1, "Secondary": 2, "Graduate": 3, "Postgraduate": 4},
            "family_support":   {"None": 0, "Low": 1, "Medium": 2, "High": 3},
        }
        for col, mapping in ordinal_map.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        df = pd.get_dummies(df, drop_first=True)

        # Align columns with what the model expects
        if hasattr(self.model, "feature_names_in_"):
            expected = list(self.model.feature_names_in_)
            for col in expected:
                if col not in df.columns:
                    df[col] = 0
            df = df[expected]

        return df.values


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict student performance")
    parser.add_argument("--input", required=True, help="Path to CSV file with student data")
    parser.add_argument("--model", default="gradient_boosting", help="Model to use")
    args = parser.parse_args()

    predictor = StudentPerformancePredictor(model_name=args.model)
    results = predictor.predict_batch(args.input)
    print(results[["predicted_grade", "confidence", "pass_fail"]].to_string())
