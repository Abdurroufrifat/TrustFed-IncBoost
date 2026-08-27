from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_binary_threshold(y_true: np.ndarray, positive_probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.02, 0.98, 193), positive_probability]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        prediction = (positive_probability >= threshold).astype(int)
        score = f1_score(y_true, prediction, average="macro", zero_division=0)
        if score > best_score or (np.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def predictions_from_probabilities(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    if probabilities.shape[1] == 2:
        return (probabilities[:, 1] >= threshold).astype(int)
    return probabilities.argmax(axis=1)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    class_names: list[str],
) -> dict:
    prediction = predictions_from_probabilities(probabilities, threshold)
    metrics = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, prediction, average="macro", zero_division=0)),
        "threshold": float(threshold),
        "classification_report": classification_report(
            y_true,
            prediction,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
    }
    try:
        if probabilities.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
            precision, recall, _ = precision_recall_curve(y_true, probabilities[:, 1])
            metrics["pr_auc"] = float(abs(np.trapezoid(precision, recall)))
        else:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
            )
    except ValueError:
        pass
    return metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    prediction = predictions_from_probabilities(probabilities, threshold)
    functions: dict[str, Callable] = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "macro_f1": lambda truth, pred: f1_score(
            truth, pred, average="macro", zero_division=0
        ),
    }
    rng = np.random.default_rng(seed)
    values = {name: [] for name in functions}
    n_rows = len(y_true)
    for _ in range(repetitions):
        sample = rng.integers(0, n_rows, size=n_rows)
        if np.unique(y_true[sample]).size < 2:
            continue
        for name, function in functions.items():
            values[name].append(float(function(y_true[sample], prediction[sample])))
    return {
        name: {
            "lower_95": float(np.percentile(scores, 2.5)),
            "median": float(np.percentile(scores, 50)),
            "upper_95": float(np.percentile(scores, 97.5)),
            "successful_resamples": len(scores),
        }
        for name, scores in values.items()
        if scores
    }

