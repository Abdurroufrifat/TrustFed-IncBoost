from __future__ import annotations

import numpy as np

from trustfed_incboost.data.federated import (
    client_partition_manifest,
    dirichlet_label_partition,
)
from trustfed_incboost.models.federated import (
    aggregate_probabilities,
    aggregation_weights,
)
from trustfed_incboost.security.poisoning import (
    flip_binary_labels,
    select_malicious_clients,
)


def test_dirichlet_partition_is_complete_disjoint_and_deterministic() -> None:
    y = np.asarray([0] * 800 + [1] * 200)
    first = dirichlet_label_partition(y, 5, 0.5, 42, 50, True)
    second = dirichlet_label_partition(y, 5, 0.5, 42, 50, True)
    for client_id in first:
        assert np.array_equal(first[client_id], second[client_id])
        assert np.unique(y[first[client_id]]).size == 2
    manifest = client_partition_manifest(first, y)
    assert manifest["n_clients"] == 5


def test_dynamic_aggregation_is_normalized() -> None:
    weights = aggregation_weights(
        np.asarray([100, 200, 300]),
        np.asarray([0.7, 0.8, 0.9]),
        np.asarray([2.0, 2.0, 2.0]),
        "performance_time",
    )
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0)
    probabilities = [
        np.asarray([[0.8, 0.2], [0.4, 0.6]]),
        np.asarray([[0.7, 0.3], [0.2, 0.8]]),
        np.asarray([[0.6, 0.4], [0.1, 0.9]]),
    ]
    combined = aggregate_probabilities(probabilities, weights)
    assert combined.shape == (2, 2)
    assert np.allclose(combined.sum(axis=1), 1.0)


def test_trust_aggregation_downweights_bad_client_and_caps_weights() -> None:
    weights = aggregation_weights(
        np.asarray([100, 100, 100]),
        np.asarray([0.9, 0.85, 0.2]),
        np.ones(3),
        "trust_aware",
        performance_power=3.0,
        validation_pr_auc=np.asarray([0.95, 0.9, 0.1]),
        validation_balanced_accuracy=np.asarray([0.9, 0.85, 0.2]),
        validation_brier=np.asarray([0.08, 0.10, 0.70]),
        consensus_deviation=np.asarray([0.05, 0.06, 0.5]),
        max_weight=0.45,
    )
    assert np.isclose(weights.sum(), 1.0)
    assert weights[2] < weights[0]
    assert weights.max() <= 0.45 + 1e-12


def test_trust_v2_rejects_seed66_consensus_camouflage() -> None:
    """Regression test for the observed largest-client seed-66 failure."""
    weights = aggregation_weights(
        np.asarray([759, 1733, 2434, 412, 4452]),
        np.asarray([0.37366976, 0.53049050, 0.75610749, 0.37645396, 0.11086897]),
        np.ones(5),
        "trust_aware_v2",
        performance_power=3.0,
        validation_pr_auc=np.asarray(
            [0.54601727, 0.29785244, 0.63450379, 0.51556367, 0.53759433]
        ),
        validation_roc_auc=np.asarray(
            [0.76559883, 0.71316492, 0.84773370, 0.63974107, 0.50705129]
        ),
        validation_balanced_accuracy=np.asarray(
            [0.58228291, 0.53238796, 0.71516106, 0.62237395, 0.49877451]
        ),
        validation_brier=np.asarray(
            [0.25248905, 0.10317928, 0.07349332, 0.25333335, 0.26136936]
        ),
        consensus_deviation=np.asarray(
            [0.00434218, 0.38101777, 0.41493300, 0.00585396, 0.01420241]
        ),
        calibration_strength=2.0,
        consensus_strength=0.10,
        chance_margin=0.02,
        minimum_macro_f1=0.25,
        security_power=2.0,
        consensus_z_cap=3.0,
        minimum_trusted_clients=3,
        max_weight=0.40,
    )
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.40 + 1e-12
    assert weights[4] == 0.0
    assert np.all(weights[:4] > 0.0)


def test_label_flip_is_deterministic_and_binary() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    first, first_indices = flip_binary_labels(labels, 0.5, 9)
    second, second_indices = flip_binary_labels(labels, 0.5, 9)
    assert np.array_equal(first, second)
    assert np.array_equal(first_indices, second_indices)
    assert len(first_indices) == 3
    assert np.all(first[first_indices] == 1 - labels[first_indices])


def test_malicious_selection_uses_training_partition_only() -> None:
    partitions = {
        0: np.asarray([0, 1, 2, 3]),
        1: np.asarray([4, 5, 6]),
        2: np.asarray([7, 8]),
    }
    labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1])
    assert select_malicious_clients(partitions, labels, "largest_client") == [0]
    assert select_malicious_clients(partitions, labels, "most_attack_samples") == [1]
    assert select_malicious_clients(
        partitions, labels, "fixed", fixed_clients=[2]
    ) == [2]
