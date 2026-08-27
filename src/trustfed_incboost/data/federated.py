from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def dirichlet_label_partition(
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
    require_all_classes: bool = True,
    max_attempts: int = 1000,
) -> dict[int, np.ndarray]:
    """Create a deterministic non-IID label-skew partition.

    Every sample is assigned to exactly one client. Rejection sampling enforces
    the minimum client size and, when requested, representation of every class.
    """
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("y must be a non-empty one-dimensional array.")
    if n_clients < 2:
        raise ValueError("n_clients must be at least 2.")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    if min_samples_per_client * n_clients > len(labels):
        raise ValueError("Minimum client sizes exceed the available samples.")

    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        assignments: list[list[int]] = [[] for _ in range(n_clients)]
        for class_value in classes:
            class_indices = np.flatnonzero(labels == class_value)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(n_clients, alpha))
            counts = rng.multinomial(len(class_indices), proportions)
            starts = np.cumsum(np.r_[0, counts[:-1]])
            for client_id, (start, count) in enumerate(zip(starts, counts, strict=True)):
                assignments[client_id].extend(
                    class_indices[int(start) : int(start + count)].tolist()
                )

        valid_size = all(len(items) >= min_samples_per_client for items in assignments)
        valid_classes = not require_all_classes or all(
            np.unique(labels[np.asarray(items, dtype=int)]).size == classes.size
            for items in assignments
        )
        if valid_size and valid_classes:
            return {
                client_id: np.asarray(rng.permutation(items), dtype=int)
                for client_id, items in enumerate(assignments)
            }
    raise RuntimeError(
        "Could not construct a valid Dirichlet partition. Increase alpha, "
        "reduce the number of clients, or lower min_samples_per_client."
    )


def client_partition_manifest(
    partitions: Mapping[int, np.ndarray], y: np.ndarray
) -> dict[str, object]:
    labels = np.asarray(y)
    all_indices = np.concatenate(list(partitions.values()))
    if len(np.unique(all_indices)) != len(labels) or set(all_indices) != set(range(len(labels))):
        raise AssertionError("Client partitions must cover each training sample exactly once.")
    clients: dict[str, object] = {}
    for client_id, indices in sorted(partitions.items()):
        values, counts = np.unique(labels[indices], return_counts=True)
        clients[str(client_id)] = {
            "rows": int(len(indices)),
            "label_distribution": {
                str(value): int(count) for value, count in zip(values, counts, strict=True)
            },
        }
    return {"n_clients": len(partitions), "clients": clients}
