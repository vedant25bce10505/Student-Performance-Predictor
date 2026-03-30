"""
model_evaluation.py
-------------------
Computes and displays evaluation metrics for all trained models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare multiple classifiers."""

    def __init__(self, class_names: list[str] | None = None):
        self.class_names = class_names
        self.results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, y_true, y_pred, model_name: str = "model") -> dict:
        """Compute accuracy, precision, recall, and F1 for a single model."""
        metrics = {
            "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
            "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
            "recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
            "f1_score":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        }
        self.results[model_name] = metrics
        logger.info("[%s] Accuracy=%.2f%%  F1=%.2f%%", model_name, metrics["accuracy"], metrics["f1_score"])
        return metrics

    def full_report(self, y_true, y_pred, model_name: str = "model") -> str:
        """Return a formatted sklearn classification report."""
        report = classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)
        print(f"\n{'='*50}\n{model_name} Classification Report\n{'='*50}\n{report}")
        return report

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def comparison_table(self) -> pd.DataFrame:
        """Return a DataFrame comparing all evaluated models."""
        if not self.results:
            raise ValueError("No models evaluated yet. Call compute_metrics() first.")
        df = pd.DataFrame(self.results).T
        df.index.name = "Model"
        return df.sort_values("f1_score", ascending=False)

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str = "model",
                               save_path: str | None = None):
        """Plot and optionally save a confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            logger.info("Confusion matrix saved → %s", save_path)
        plt.show()

    def plot_model_comparison(self, save_path: str | None = None):
        """Bar chart comparing accuracy, precision, recall, F1 across models."""
        df = self.comparison_table()
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind="bar", ax=ax, colormap="viridis", width=0.7)
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right")
        ax.set_xticklabels(df.index, rotation=15, ha="right")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()

    def plot_feature_importance(self, model, feature_names: list[str],
                                 top_n: int = 20, save_path: str | None = None):
        """Plot feature importances for tree-based models."""
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model does not have feature_importances_; skipping plot.")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features[::-1], top_importances[::-1], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()


if __name__ == "__main__":
    print("ModelEvaluator ready.")
