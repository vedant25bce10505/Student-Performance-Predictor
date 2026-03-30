"""
visualization.py
----------------
Static and interactive visualisation helpers for EDA and model results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os

logger = logging.getLogger(__name__)
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# EDA helpers
# -----------------------------------------------------------------------

def plot_grade_distribution(df: pd.DataFrame, target_col: str = "target", save: bool = False):
    """Bar chart of grade/class distribution."""
    counts = df[target_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Grade Distribution")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, "grade_distribution.png"), dpi=150)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = False):
    """Heatmap of feature correlations (numeric columns only)."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=False,
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=150)
    plt.show()


def plot_study_hours_vs_grade(df: pd.DataFrame, target_col: str = "target", save: bool = False):
    """Box plot of study hours for each grade category."""
    fig, ax = plt.subplots(figsize=(9, 5))
    df.boxplot(column="study_hours", by=target_col, ax=ax, patch_artist=True)
    ax.set_title("Study Hours by Grade")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Study Hours per Week")
    plt.suptitle("")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, "study_hours_vs_grade.png"), dpi=150)
    plt.show()


def plot_attendance_vs_grade(df: pd.DataFrame, target_col: str = "target", save: bool = False):
    """Violin plot of attendance rate per grade."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=df, x=target_col, y="attendance", palette="muted", ax=ax)
    ax.set_title("Attendance Rate by Grade")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Attendance (%)")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, "attendance_vs_grade.png"), dpi=150)
    plt.show()


# -----------------------------------------------------------------------
# Interactive (Plotly)
# -----------------------------------------------------------------------

def interactive_scatter(df: pd.DataFrame, x: str = "study_hours",
                         y: str = "previous_grade", color: str = "target") -> go.Figure:
    """Interactive scatter coloured by grade."""
    fig = px.scatter(df, x=x, y=y, color=color.astype(str) if color in df else None,
                     title=f"{x} vs {y}", template="plotly_white", opacity=0.7)
    fig.show()
    return fig


def interactive_model_comparison(results: dict) -> go.Figure:
    """
    Grouped bar chart comparing model metrics.

    Parameters
    ----------
    results : dict  e.g. {"Random Forest": {"accuracy": 87.5, ...}, ...}
    """
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    models = list(results.keys())
    fig = make_subplots(rows=1, cols=1)

    for metric in metrics:
        values = [results[m].get(metric, 0) for m in models]
        fig.add_trace(go.Bar(name=metric.replace("_", " ").title(), x=models, y=values))

    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        yaxis_title="Score (%)",
        yaxis_range=[70, 100],
        template="plotly_white",
    )
    fig.show()
    return fig


def interactive_feature_importance(model, feature_names: list[str], top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of feature importances (Plotly)."""
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature_importances_")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
    ))
    fig.update_layout(title=f"Top {top_n} Feature Importances", template="plotly_white")
    fig.show()
    return fig


if __name__ == "__main__":
    print("Visualization helpers loaded. Pass a DataFrame to the plot functions.")
