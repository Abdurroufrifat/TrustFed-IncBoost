from __future__ import annotations

import numpy as np


def aggregation_weights(
    sample_counts: np.ndarray,
    validation_scores: np.ndarray,
    training_seconds: np.ndarray,
    method: str,
    sample_power: float = 0.5,
    performance_power: float = 2.0,
    time_power: float = 0.25,
    validation_pr_auc: np.ndarray | None = None,
    validation_roc_auc: np.ndarray | None = None,
    validation_balanced_accuracy: np.ndarray | None = None,
    validation_brier: np.ndarray | None = None,
    consensus_deviation: np.ndarray | None = None,
    calibration_strength: float = 1.0,
    consensus_strength: float = 0.5,
    chance_margin: float = 0.02,
    minimum_macro_f1: float = 0.25,
    security_power: float = 2.0,
    consensus_z_cap: float = 3.0,
    minimum_trusted_clients: int = 3,
    max_weight: float = 0.45,
) -> np.ndarray:
    counts = np.asarray(sample_counts, dtype=float)
    scores = np.asarray(validation_scores, dtype=float)
    durations = np.asarray(training_seconds, dtype=float)
    if not (counts.shape == scores.shape == durations.shape) or counts.ndim != 1:
        raise ValueError("Aggregation inputs must be aligned one-dimensional arrays.")
    if np.any(counts <= 0) or np.any(durations <= 0):
        raise ValueError("Sample counts and training times must be positive.")
    if method == "uniform":
        raw = np.ones_like(counts)
    elif method == "sample_size":
        raw = counts
    elif method == "performance_time":
        safe_scores = np.clip(scores, 1e-6, 1.0)
        raw = (
            np.power(counts, sample_power)
            * np.power(safe_scores, performance_power)
            / np.power(np.maximum(durations, 1e-6), time_power)
        )
    elif method == "trust_aware":
        if any(
            value is None
            for value in (
                validation_pr_auc,
                validation_balanced_accuracy,
                validation_brier,
                consensus_deviation,
            )
        ):
            raise ValueError("trust_aware aggregation requires all trust diagnostics.")
        pr_auc = np.clip(np.asarray(validation_pr_auc, dtype=float), 1e-6, 1.0)
        balanced = np.clip(
            np.asarray(validation_balanced_accuracy, dtype=float), 1e-6, 1.0
        )
        brier = np.maximum(np.asarray(validation_brier, dtype=float), 0.0)
        deviation = np.maximum(np.asarray(consensus_deviation, dtype=float), 0.0)
        if not all(value.shape == counts.shape for value in (pr_auc, balanced, brier, deviation)):
            raise ValueError("Trust diagnostics must align with the client arrays.")
        quality = np.cbrt(np.clip(scores, 1e-6, 1.0) * pr_auc * balanced)
        deviation_scale = float(np.median(deviation)) + 1e-6
        raw = (
            np.power(quality, performance_power)
            * np.exp(-calibration_strength * brier)
            * np.exp(-consensus_strength * deviation / deviation_scale)
        )
    elif method == "trust_aware_v2":
        if any(
            value is None
            for value in (
                validation_pr_auc,
                validation_roc_auc,
                validation_balanced_accuracy,
                validation_brier,
                consensus_deviation,
            )
        ):
            raise ValueError("trust_aware_v2 aggregation requires all trust diagnostics.")
        components = trust_aware_v2_components(
            validation_macro_f1=scores,
            validation_pr_auc=np.asarray(validation_pr_auc, dtype=float),
            validation_roc_auc=np.asarray(validation_roc_auc, dtype=float),
            validation_balanced_accuracy=np.asarray(
                validation_balanced_accuracy, dtype=float
            ),
            validation_brier=np.asarray(validation_brier, dtype=float),
            consensus_deviation=np.asarray(consensus_deviation, dtype=float),
            performance_power=performance_power,
            calibration_strength=calibration_strength,
            consensus_strength=consensus_strength,
            chance_margin=chance_margin,
            minimum_macro_f1=minimum_macro_f1,
            security_power=security_power,
            consensus_z_cap=consensus_z_cap,
            minimum_trusted_clients=minimum_trusted_clients,
        )
        raw = components["raw_score"]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Aggregation weights are not finite and positive.")
    normalized = raw / total
    if method not in {"trust_aware", "trust_aware_v2"}:
        return normalized
    if not 1.0 / len(normalized) <= max_weight <= 1.0:
        raise ValueError("max_weight must be at least 1/n_clients and at most 1.")
    if method == "trust_aware_v2":
        eligible = components["eligible"].astype(bool)
        effective_cap = max(max_weight, 1.0 / int(eligible.sum()))
        weights = np.zeros_like(raw)
        weights[eligible] = _capped_normalize(raw[eligible], effective_cap)
        return weights
    return _capped_normalize(raw, max_weight)


def trust_aware_v2_components(
    validation_macro_f1: np.ndarray,
    validation_pr_auc: np.ndarray,
    validation_roc_auc: np.ndarray,
    validation_balanced_accuracy: np.ndarray,
    validation_brier: np.ndarray,
    consensus_deviation: np.ndarray,
    performance_power: float = 3.0,
    calibration_strength: float = 2.0,
    consensus_strength: float = 0.10,
    chance_margin: float = 0.02,
    minimum_macro_f1: float = 0.25,
    security_power: float = 2.0,
    consensus_z_cap: float = 3.0,
    minimum_trusted_clients: int = 3,
) -> dict[str, np.ndarray | bool]:
    """Build supervised, chance-aware trust evidence for each client.

    V1 can mistake a poisoned model for a consensus model when several honest
    clients are highly label-skewed. V2 first requires performance above random
    on a clean public validation set. Consensus remains a bounded secondary
    signal, so a collapsed median cannot overwhelm supervised evidence.
    """
    macro_f1 = np.clip(np.asarray(validation_macro_f1, dtype=float), 1e-6, 1.0)
    pr_auc = np.clip(np.asarray(validation_pr_auc, dtype=float), 1e-6, 1.0)
    roc_auc = np.clip(np.asarray(validation_roc_auc, dtype=float), 1e-6, 1.0)
    balanced = np.clip(
        np.asarray(validation_balanced_accuracy, dtype=float), 1e-6, 1.0
    )
    brier = np.maximum(np.asarray(validation_brier, dtype=float), 0.0)
    deviation = np.maximum(np.asarray(consensus_deviation, dtype=float), 0.0)
    arrays = (macro_f1, pr_auc, roc_auc, balanced, brier, deviation)
    if any(value.ndim != 1 or value.shape != macro_f1.shape for value in arrays):
        raise ValueError("Trust diagnostics must be aligned one-dimensional arrays.")
    if not 0.0 <= chance_margin < 0.5:
        raise ValueError("chance_margin must be in [0, 0.5).")
    if not 0.0 <= minimum_macro_f1 <= 1.0:
        raise ValueError("minimum_macro_f1 must be in [0, 1].")
    if minimum_trusted_clients < 1 or minimum_trusted_clients > len(macro_f1):
        raise ValueError("minimum_trusted_clients must be between 1 and n_clients.")

    balanced_signal = np.clip((balanced - 0.5) / 0.5, 0.0, 1.0)
    roc_signal = np.clip((roc_auc - 0.5) / 0.5, 0.0, 1.0)
    security_evidence = np.sqrt(balanced_signal * roc_signal)
    quality = np.power(macro_f1 * pr_auc * balanced * roc_auc, 0.25)
    gate_passed = (
        (balanced >= 0.5 + chance_margin)
        & (roc_auc >= 0.5 + chance_margin)
        & (macro_f1 >= minimum_macro_f1)
    )

    supervised_rank = (
        quality
        * np.maximum(security_evidence, 1e-12)
        * np.exp(-calibration_strength * brier)
    )
    eligible = gate_passed.copy()
    fallback_activated = bool(eligible.sum() < minimum_trusted_clients)
    if fallback_activated:
        top = np.argsort(supervised_rank)[-minimum_trusted_clients:]
        eligible = np.zeros_like(gate_passed, dtype=bool)
        eligible[top] = True

    deviation_scale = float(np.median(deviation)) + 1e-6
    consensus_z = np.clip(deviation / deviation_scale, 0.0, consensus_z_cap)
    raw = (
        np.power(quality, performance_power)
        * np.power(np.maximum(security_evidence, 1e-12), security_power)
        * np.exp(-calibration_strength * brier)
        * np.exp(-consensus_strength * consensus_z)
    )
    raw[~eligible] = 0.0
    return {
        "quality": quality,
        "security_evidence": security_evidence,
        "consensus_z": consensus_z,
        "gate_passed": gate_passed,
        "eligible": eligible,
        "raw_score": raw,
        "fallback_activated": fallback_activated,
    }


def _capped_normalize(raw: np.ndarray, cap: float) -> np.ndarray:
    """Project positive scores onto the simplex with an upper cap."""
    scores = np.maximum(np.asarray(raw, dtype=float), 1e-12)
    low, high = 0.0, 1.0 / scores.min()
    while np.minimum(cap, scores * high).sum() < 1.0:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if np.minimum(cap, scores * middle).sum() < 1.0:
            low = middle
        else:
            high = middle
    weights = np.minimum(cap, scores * high)
    return weights / weights.sum()


def aggregate_probabilities(
    client_probabilities: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    if not client_probabilities:
        raise ValueError("At least one client probability matrix is required.")
    stacked = np.stack(client_probabilities, axis=0)
    normalized = np.asarray(weights, dtype=float)
    if normalized.shape != (stacked.shape[0],):
        raise ValueError("One aggregation weight is required per client.")
    return np.tensordot(normalized, stacked, axes=(0, 0))
