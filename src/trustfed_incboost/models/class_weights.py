from __future__ import annotations

from typing import Hashable

import numpy as np
import pandas as pd


def class_weight_map(
    y: pd.Series | np.ndarray,
    method: str,
    beta: float = 0.9999,
) -> dict[Hashable, float]:
    series = pd.Series(y)
    counts = series.value_counts().sort_index()
    total = float(counts.sum())
    classes = float(len(counts))
    if method == "none":
        weights = pd.Series(1.0, index=counts.index)
    elif method == "paper_printed":
        weights = 2.0 * counts / total
    elif method == "balanced":
        weights = total / (classes * counts)
    elif method == "effective_number":
        effective = (1.0 - np.power(beta, counts.astype(float))) / (1.0 - beta)
        weights = 1.0 / effective
        weights *= classes / weights.sum()
    else:
        raise ValueError(f"Unknown class-weight method: {method}")
    return {key: float(value) for key, value in weights.items()}


def sample_weights(y: pd.Series | np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    mapping = class_weight_map(y, method)
    weights = pd.Series(y).map(mapping)
    if weights.isna().any():
        raise ValueError("A class was missing from the class-weight map.")
    return weights.to_numpy(dtype=float), mapping

