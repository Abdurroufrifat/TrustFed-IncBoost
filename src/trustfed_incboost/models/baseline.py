from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def build_classifier(
    backend: str,
    params: dict[str, Any],
    n_classes: int,
    seed: int,
):
    cleaned = dict(params)
    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "XGBoost is not installed. Run: pip install -r requirements.txt"
            ) from exc
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        eval_metric = "logloss" if n_classes == 2 else "mlogloss"
        cleaned.setdefault("objective", objective)
        cleaned.setdefault("eval_metric", eval_metric)
        cleaned.setdefault("random_state", seed)
        if n_classes > 2:
            cleaned.setdefault("num_class", n_classes)
        return XGBClassifier(**cleaned)
    if backend == "sklearn":
        allowed = {
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "max_depth",
            "min_samples_leaf",
            "l2_regularization",
            "early_stopping",
            "validation_fraction",
            "n_iter_no_change",
            "tol",
        }
        cleaned = {key: value for key, value in cleaned.items() if key in allowed}
        cleaned.setdefault("random_state", seed)
        cleaned.setdefault("early_stopping", True)
        return HistGradientBoostingClassifier(**cleaned)
    raise ValueError(f"Unknown model backend: {backend}")


def fit_classifier(
    model,
    backend: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    X_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
):
    if backend == "xgboost" and X_validation is not None and len(X_validation):
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def predict_probabilities(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    decision = np.asarray(model.decision_function(X))
    if decision.ndim == 1:
        positive = 1.0 / (1.0 + np.exp(-decision))
        return np.column_stack([1.0 - positive, positive])
    exp = np.exp(decision - decision.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

