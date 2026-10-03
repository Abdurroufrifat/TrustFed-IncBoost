"""Keep every site's training rows on exactly one federated client."""
from __future__ import annotations

import numpy as np


def site_group_partition(sites, labels, *, n_clients: int,
                         min_samples_per_client: int = 1,
                         require_all_classes: bool = True) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    sites = np.asarray(sites, dtype=object)
    labels = np.asarray(labels)
    if sites.ndim != 1 or labels.ndim != 1 or len(sites) != len(labels) or len(sites) == 0:
        raise ValueError("site groups and labels must be nonempty aligned vectors")
    if any(value is None or str(value).strip() == "" for value in sites):
        raise ValueError("site identifiers must be present")
    cleaned = np.array([str(value) for value in sites])
    unique = sorted(set(cleaned))
    if len(unique) != n_clients or n_clients < 2:
        raise ValueError(f"site count {len(unique)} differs from configured n_clients {n_clients}")
    classes = set(np.unique(labels))
    groups = {index: np.flatnonzero(cleaned == site) for index, site in enumerate(unique)}
    for index, indices in groups.items():
        if len(indices) < min_samples_per_client:
            raise ValueError(f"site {unique[index]!r} has too few training records")
        if require_all_classes and set(labels[indices]) != classes:
            raise ValueError(f"site {unique[index]!r} lacks a class required for local training")
    return groups, dict(enumerate(unique))
