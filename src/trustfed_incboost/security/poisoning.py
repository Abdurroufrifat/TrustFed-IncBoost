from __future__ import annotations

import numpy as np


def flip_binary_labels(
    labels: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flip a deterministic random fraction of binary labels."""
    values = np.asarray(labels, dtype=int)
    if values.ndim != 1 or not set(np.unique(values)).issubset({0, 1}):
        raise ValueError("Label flipping requires a one-dimensional binary array.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("flip_fraction must be between 0 and 1.")
    count = int(round(len(values) * fraction))
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(values), size=count, replace=False))
    poisoned = values.copy()
    poisoned[selected] = 1 - poisoned[selected]
    return poisoned, selected


def select_malicious_clients(
    partitions: dict[int, np.ndarray],
    labels: np.ndarray,
    strategy: str,
    count: int = 1,
    fixed_clients: list[int] | None = None,
) -> list[int]:
    """Select malicious clients without consulting validation or test data."""
    if not 1 <= count <= len(partitions):
        raise ValueError("malicious_count must be between 1 and n_clients.")
    if strategy == "fixed":
        selected = sorted(int(value) for value in (fixed_clients or []))
        if len(selected) != count:
            raise ValueError("fixed strategy requires exactly malicious_count client ids.")
    elif strategy == "largest_client":
        selected = sorted(
            partitions,
            key=lambda client_id: (-len(partitions[client_id]), client_id),
        )[:count]
    elif strategy == "most_attack_samples":
        binary = np.asarray(labels, dtype=int)
        selected = sorted(
            partitions,
            key=lambda client_id: (
                -int(np.sum(binary[partitions[client_id]] == 1)),
                client_id,
            ),
        )[:count]
    else:
        raise ValueError(f"Unknown malicious client strategy: {strategy}")
    if any(client_id not in partitions for client_id in selected):
        raise ValueError("Selected malicious client id is not present.")
    return selected
