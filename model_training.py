"""
model_training.py
-----------------
Trains Random Forest, Gradient Boosting, Decision Tree, and Neural Network
models with hyperparameter tuning and cross-validation.
"""

import os
import pickle
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------

def build_random_forest(X_train, y_train, tune: bool = True) -> RandomForestClassifier:
    """Train a Random Forest with optional grid search tuning."""
    if tune:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth":    [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        }
        base = RandomForestClassifier(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(base, param_grid, n_iter=20, cv=5,
                                    scoring="f1_weighted", random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        logger.info("Best RF params: %s", search.best_params_)
        return search.best_estimator_

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


def build_gradient_boosting(X_train, y_train, tune: bool = True) -> GradientBoostingClassifier:
    """Train a Gradient Boosting classifier with optional randomised search."""
    if tune:
        param_dist = {
            "n_estimators":    [100, 200, 300],
            "learning_rate":   [0.05, 0.1, 0.2],
            "max_depth":       [3, 5, 7],
            "subsample":       [0.8, 1.0],
            "min_samples_leaf": [1, 5, 10],
        }
        base = GradientBoostingClassifier(random_state=42)
        search = RandomizedSearchCV(base, param_dist, n_iter=30, cv=5,
                                    scoring="f1_weighted", random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        logger.info("Best GB params: %s", search.best_params_)
        return search.best_estimator_

    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                     max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    return gb


def build_decision_tree(X_train, y_train, tune: bool = True) -> DecisionTreeClassifier:
    """Train a Decision Tree with optional grid search."""
    if tune:
        param_grid = {
            "max_depth":    [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion":    ["gini", "entropy"],
        }
        base = DecisionTreeClassifier(random_state=42)
        search = GridSearchCV(base, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1)
        search.fit(X_train, y_train)
        logger.info("Best DT params: %s", search.best_params_)
        return search.best_estimator_

    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    return dt


def build_neural_network(input_dim: int, num_classes: int) -> keras.Model:
    """Build a 3-hidden-layer neural network with dropout."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_neural_network(X_train, y_train, X_val, y_val) -> keras.Model:
    """Train the neural network with early stopping."""
    num_classes = len(np.unique(y_train))
    model = build_neural_network(X_train.shape[1], num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )
    return model


# -----------------------------------------------------------------------
# Save / load helpers
# -----------------------------------------------------------------------

def save_sklearn_model(model, name: str) -> str:
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved %s → %s", name, path)
    return path


def load_sklearn_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_neural_network(model: keras.Model) -> str:
    path = os.path.join(MODELS_DIR, "neural_network.h5")
    model.save(path)
    logger.info("Saved neural network → %s", path)
    return path


# -----------------------------------------------------------------------
# Cross-validation helper
# -----------------------------------------------------------------------

def evaluate_with_cv(model, X, y, cv: int = 5) -> dict:
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    return {"mean_f1": scores.mean(), "std_f1": scores.std(), "scores": scores.tolist()}


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

def main(args):
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer

    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "student_data.csv")
    X_train, X_test, y_train, y_test = preprocessor.full_pipeline(DATA_PATH)

    X_train_fe, y_train_fe = engineer.transform(
        pd.DataFrame(X_train).assign(target=y_train.values), fit=True
    )
    X_test_fe = engineer.select_features(pd.DataFrame(X_test), y_test, fit=False)

    model_name = args.model.lower()

    if model_name in ("random_forest", "all"):
        rf = build_random_forest(X_train_fe, y_train_fe, tune=True)
        save_sklearn_model(rf, "random_forest")
        print("\nRandom Forest:\n", classification_report(y_test, rf.predict(X_test_fe)))

    if model_name in ("gradient_boosting", "all"):
        gb = build_gradient_boosting(X_train_fe, y_train_fe, tune=True)
        save_sklearn_model(gb, "gradient_boosting")
        print("\nGradient Boosting:\n", classification_report(y_test, gb.predict(X_test_fe)))

    if model_name in ("decision_tree", "all"):
        dt = build_decision_tree(X_train_fe, y_train_fe, tune=True)
        save_sklearn_model(dt, "decision_tree")
        print("\nDecision Tree:\n", classification_report(y_test, dt.predict(X_test_fe)))

    if model_name in ("neural_network", "all"):
        split = int(0.8 * len(X_train_fe))
        nn = train_neural_network(
            X_train_fe.values[:split], y_train_fe.values[:split],
            X_train_fe.values[split:], y_train_fe.values[split:],
        )
        save_neural_network(nn)
        y_pred_nn = np.argmax(nn.predict(X_test_fe.values), axis=1)
        print("\nNeural Network:\n", classification_report(y_test, y_pred_nn))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student performance models")
    parser.add_argument("--model", default="all",
                        choices=["random_forest", "gradient_boosting", "decision_tree", "neural_network", "all"],
                        help="Which model to train")
    main(parser.parse_args())
