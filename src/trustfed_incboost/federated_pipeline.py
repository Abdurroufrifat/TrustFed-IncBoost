from __future__ import annotations

import copy
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from trustfed_incboost.config import project_root_from_config, resolve_from_project
from trustfed_incboost.data.audit import audit_dataframe, dataframe_fingerprint
from trustfed_incboost.data.federated import (
    client_partition_manifest,
    dirichlet_label_partition,
)
from trustfed_incboost.data.loader import (
    choose_positive_label,
    infer_label_column,
    load_dataset,
    prepare_features,
)
from trustfed_incboost.data.splitting import make_split, split_manifest
from trustfed_incboost.data.site_partition import site_group_partition
from trustfed_incboost.evaluation.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    select_binary_threshold,
)
from trustfed_incboost.evaluation.reporting import save_confusion_matrix
from trustfed_incboost.models.baseline import (
    build_classifier,
    fit_classifier,
    predict_probabilities,
)
from trustfed_incboost.models.class_weights import sample_weights
from trustfed_incboost.models.federated import (
    aggregate_probabilities,
    aggregation_weights,
    trust_aware_v2_components,
)
from trustfed_incboost.models.preprocessing import NumericPreprocessor
from trustfed_incboost.reproducibility import (
    environment_manifest,
    seed_everything,
    utc_run_id,
    write_json,
)
from trustfed_incboost.security.poisoning import (
    flip_binary_labels,
    select_malicious_clients,
)


def _dataset_pattern(config: dict, root: Path) -> str:
    raw = str(config["dataset"]["path"])
    return raw if Path(raw).is_absolute() else str((root / raw).resolve())


def _encode_binary_labels(y_raw: pd.Series, positive_label: object):
    negative = [value for value in y_raw.unique() if value != positive_label]
    if y_raw.nunique() != 2 or len(negative) != 1:
        raise ValueError("The Phase-2 federated baseline currently requires binary labels.")
    ordered = negative + [positive_label]
    mapping = {value: index for index, value in enumerate(ordered)}
    return y_raw.map(mapping).to_numpy(dtype=int), [str(value) for value in ordered], mapping


def run_federated_training(config: dict) -> Path:
    root = project_root_from_config(config)
    seed = int(config["project"].get("seed", 42))
    seed_everything(seed)
    output_base = resolve_from_project(config["output"]["directory"], root)
    run_directory = output_base / utc_run_id(config["project"]["name"])
    run_directory.mkdir(parents=True, exist_ok=False)

    frame, source_paths = load_dataset(
        _dataset_pattern(config, root), config["dataset"].get("max_rows")
    )
    label_column = infer_label_column(frame, config["dataset"].get("label_column", "auto"))
    positive_label = choose_positive_label(
        frame[label_column], config["dataset"].get("positive_label", "auto")
    )
    if positive_label is None:
        raise ValueError("A positive label is required for binary federated training.")
    audit = audit_dataframe(
        frame,
        label_column,
        source_paths,
        excluded_from_fingerprint=config["dataset"].get("drop_columns", []),
    )
    write_json(run_directory / "dataset_audit.json", audit)
    X, y_raw = prepare_features(
        frame, label_column, config["dataset"].get("drop_columns", [])
    )
    y, class_names, raw_mapping = _encode_binary_labels(y_raw, positive_label)
    y_series = pd.Series(y)

    split_config = config["split"]
    split = make_split(
        X,
        y_series,
        method=split_config["method"],
        seed=seed,
        test_size=float(split_config["test_size"]),
        validation_size=float(split_config["validation_size"]),
        group_column=split_config.get("group_column"),
        time_column=split_config.get("time_column"),
    )
    if not len(split.validation):
        raise ValueError("Federated aggregation requires a non-empty public validation split.")
    manifest = split_manifest(split, y_series)
    fingerprint_columns = [
        column
        for column in X.columns
        if column not in {
            "__source_file__",
            split_config.get("group_column"),
            split_config.get("time_column"),
        }
    ]
    fingerprints = dataframe_fingerprint(X[fingerprint_columns])
    manifest["fingerprint_overlap"] = {
        "train_validation": len(set(fingerprints.iloc[split.train]) & set(fingerprints.iloc[split.validation])),
        "train_test": len(set(fingerprints.iloc[split.train]) & set(fingerprints.iloc[split.test])),
        "validation_test": len(set(fingerprints.iloc[split.validation]) & set(fingerprints.iloc[split.test])),
    }
    if split.method == "fingerprint_disjoint" and any(manifest["fingerprint_overlap"].values()):
        raise AssertionError("Fingerprint-disjoint split contains cross-partition duplicates.")
    write_json(run_directory / "split_manifest.json", manifest)

    pre_config = config["preprocessing"]
    preprocessor = NumericPreprocessor(
        use_ica=bool(pre_config.get("use_ica", False)),
        ica_components=pre_config.get("ica_components"),
        seed=seed,
    )
    X_train = preprocessor.fit_transform(X.iloc[split.train])
    X_validation = preprocessor.transform(X.iloc[split.validation])
    X_test = preprocessor.transform(X.iloc[split.test])
    y_train, y_validation, y_test = y[split.train], y[split.validation], y[split.test]

    fed_config = config["federated"]
    partition_config = fed_config["partition"]
    partition_method = partition_config["method"]
    if partition_method == "site_group":
        site_column = partition_config["site_column"]
        if site_column not in X.columns:
            raise ValueError(f"Site column {site_column!r} must be kept in the training features")
        partitions, site_map = site_group_partition(
            X.iloc[split.train][site_column].to_numpy(),
            y_train,
            n_clients=int(fed_config["n_clients"]),
            min_samples_per_client=int(partition_config.get("min_samples_per_client", 1)),
            require_all_classes=bool(partition_config.get("require_all_classes", True)),
        )
    elif partition_method == "dirichlet_label":
        partitions = dirichlet_label_partition(
            y_train,
            n_clients=int(fed_config["n_clients"]),
            alpha=float(partition_config["alpha"]),
            seed=seed + 200,
            min_samples_per_client=int(partition_config.get("min_samples_per_client", 1)),
            require_all_classes=bool(partition_config.get("require_all_classes", True)),
        )
        site_map = {}
    else:
        raise ValueError(f"Unknown federated partition method: {partition_method}")
    partition_report = client_partition_manifest(partitions, y_train)
    partition_report["method"] = partition_method
    if site_map:
        partition_report["site_by_client"] = {str(i): site for i, site in site_map.items()}
    else:
        partition_report["alpha"] = float(partition_config["alpha"])
    write_json(run_directory / "client_partitions.json", partition_report)

    model_config = config["model"]
    clients: list[dict] = []
    validation_probabilities: list[np.ndarray] = []
    test_probabilities: list[np.ndarray] = []
    poison_config = config.get("poisoning", {"enabled": False})
    poisoning_enabled = bool(poison_config.get("enabled", False))
    malicious_strategy = str(poison_config.get("strategy", "fixed"))
    malicious_clients = set(
        select_malicious_clients(
            partitions,
            y_train,
            malicious_strategy,
            count=int(poison_config.get("malicious_count", 1)),
            fixed_clients=[
                int(value) for value in poison_config.get("malicious_clients", [])
            ],
        )
    ) if poisoning_enabled else set()
    poisoning_report: dict[str, object] = {
        "enabled": poisoning_enabled,
        "attack": poison_config.get("attack", "none") if poisoning_enabled else "none",
        "strategy": malicious_strategy if poisoning_enabled else "none",
        "malicious_clients": sorted(malicious_clients),
        "clients": {},
    }
    for client_id, indices in sorted(partitions.items()):
        clean_local_y = y_train[indices]
        local_y = clean_local_y.copy()
        poisoned_positions = np.asarray([], dtype=int)
        if client_id in malicious_clients:
            if poison_config.get("attack") != "label_flip":
                raise ValueError("Phase 3 currently supports poisoning.attack=label_flip.")
            local_y, poisoned_positions = flip_binary_labels(
                clean_local_y,
                float(poison_config.get("flip_fraction", 1.0)),
                seed + 1000 + client_id,
            )
        poisoning_report["clients"][str(client_id)] = {
            "malicious": client_id in malicious_clients,
            "rows": int(len(indices)),
            "flipped_labels": int(len(poisoned_positions)),
            "clean_label_distribution": {
                str(value): int(count)
                for value, count in zip(*np.unique(clean_local_y, return_counts=True), strict=True)
            },
            "training_label_distribution": {
                str(value): int(count)
                for value, count in zip(*np.unique(local_y, return_counts=True), strict=True)
            },
        }
        weights, weight_map = sample_weights(local_y, model_config["class_weight"])
        model = build_classifier(
            model_config["backend"],
            model_config.get("params", {}),
            len(class_names),
            seed + client_id,
        )
        started = time.perf_counter()
        model = fit_classifier(
            model,
            model_config["backend"],
            X_train[indices],
            local_y,
            weights,
            X_validation,
            y_validation,
        )
        elapsed = time.perf_counter() - started
        val_prob = predict_probabilities(model, X_validation)
        test_prob = predict_probabilities(model, X_test)
        val_metrics = classification_metrics(y_validation, val_prob, 0.5, class_names)
        clients.append(
            {
                "client_id": client_id,
                "rows": int(len(indices)),
                "training_seconds": elapsed,
                "class_weight_map": {str(key): value for key, value in weight_map.items()},
                "malicious": client_id in malicious_clients,
                "validation_at_0_5": val_metrics,
                "model": model,
            }
        )
        validation_probabilities.append(val_prob)
        test_probabilities.append(test_prob)

    write_json(run_directory / "poisoning_manifest.json", poisoning_report)

    agg_config = fed_config["aggregation"]
    positive_validation = np.stack(
        [probability[:, 1] for probability in validation_probabilities], axis=0
    )
    consensus = np.median(positive_validation, axis=0)
    brier_scores = np.mean(
        np.square(positive_validation - y_validation[None, :]), axis=1
    )
    consensus_deviations = np.mean(
        np.abs(positive_validation - consensus[None, :]), axis=1
    )
    ensemble_weights = aggregation_weights(
        np.asarray([item["rows"] for item in clients]),
        np.asarray([item["validation_at_0_5"]["macro_f1"] for item in clients]),
        np.asarray([item["training_seconds"] for item in clients]),
        method=agg_config["method"],
        sample_power=float(agg_config.get("sample_power", 0.5)),
        performance_power=float(agg_config.get("performance_power", 2.0)),
        time_power=float(agg_config.get("time_power", 0.25)),
        validation_pr_auc=np.asarray(
            [item["validation_at_0_5"].get("pr_auc", 1e-6) for item in clients]
        ),
        validation_roc_auc=np.asarray(
            [item["validation_at_0_5"].get("roc_auc", 1e-6) for item in clients]
        ),
        validation_balanced_accuracy=np.asarray(
            [item["validation_at_0_5"]["balanced_accuracy"] for item in clients]
        ),
        validation_brier=brier_scores,
        consensus_deviation=consensus_deviations,
        calibration_strength=float(agg_config.get("calibration_strength", 1.0)),
        consensus_strength=float(agg_config.get("consensus_strength", 0.5)),
        chance_margin=float(agg_config.get("chance_margin", 0.02)),
        minimum_macro_f1=float(agg_config.get("minimum_macro_f1", 0.25)),
        security_power=float(agg_config.get("security_power", 2.0)),
        consensus_z_cap=float(agg_config.get("consensus_z_cap", 3.0)),
        minimum_trusted_clients=int(agg_config.get("minimum_trusted_clients", 3)),
        max_weight=float(agg_config.get("max_weight", 0.45)),
    )
    trust_v2 = None
    if agg_config["method"] == "trust_aware_v2":
        trust_v2 = trust_aware_v2_components(
            validation_macro_f1=np.asarray(
                [item["validation_at_0_5"]["macro_f1"] for item in clients]
            ),
            validation_pr_auc=np.asarray(
                [item["validation_at_0_5"].get("pr_auc", 1e-6) for item in clients]
            ),
            validation_roc_auc=np.asarray(
                [item["validation_at_0_5"].get("roc_auc", 1e-6) for item in clients]
            ),
            validation_balanced_accuracy=np.asarray(
                [item["validation_at_0_5"]["balanced_accuracy"] for item in clients]
            ),
            validation_brier=brier_scores,
            consensus_deviation=consensus_deviations,
            performance_power=float(agg_config.get("performance_power", 2.0)),
            calibration_strength=float(agg_config.get("calibration_strength", 1.0)),
            consensus_strength=float(agg_config.get("consensus_strength", 0.5)),
            chance_margin=float(agg_config.get("chance_margin", 0.02)),
            minimum_macro_f1=float(agg_config.get("minimum_macro_f1", 0.25)),
            security_power=float(agg_config.get("security_power", 2.0)),
            consensus_z_cap=float(agg_config.get("consensus_z_cap", 3.0)),
            minimum_trusted_clients=int(agg_config.get("minimum_trusted_clients", 3)),
        )
    ensemble_validation = aggregate_probabilities(validation_probabilities, ensemble_weights)
    threshold = select_binary_threshold(y_validation, ensemble_validation[:, 1])
    validation_metrics = classification_metrics(
        y_validation, ensemble_validation, threshold, class_names
    )
    inference_start = time.perf_counter()
    ensemble_test = aggregate_probabilities(test_probabilities, ensemble_weights)
    inference_seconds = time.perf_counter() - inference_start
    test_metrics = classification_metrics(y_test, ensemble_test, threshold, class_names)
    intervals = bootstrap_confidence_intervals(
        y_test,
        ensemble_test,
        threshold,
        int(config["evaluation"].get("bootstrap_repetitions", 500)),
        seed + 300,
    )

    client_report = []
    models = []
    for index, (item, ensemble_weight) in enumerate(
        zip(clients, ensemble_weights, strict=True)
    ):
        report_item = {key: value for key, value in item.items() if key != "model"} | {
            "aggregation_weight": float(ensemble_weight),
            "validation_brier": float(brier_scores[index]),
            "consensus_deviation": float(consensus_deviations[index]),
        }
        if trust_v2 is not None:
            report_item["trust_v2"] = {
                "quality": float(trust_v2["quality"][index]),
                "security_evidence": float(trust_v2["security_evidence"][index]),
                "consensus_z": float(trust_v2["consensus_z"][index]),
                "gate_passed": bool(trust_v2["gate_passed"][index]),
                "eligible": bool(trust_v2["eligible"][index]),
                "raw_score": float(trust_v2["raw_score"][index]),
                "fallback_activated": bool(trust_v2["fallback_activated"]),
            }
        client_report.append(report_item)
        models.append(item["model"])
    write_json(
        run_directory / "metrics.json",
        {
            "validation": validation_metrics,
            "test": test_metrics,
            "bootstrap_95_ci": intervals,
            "clients": client_report,
            "total_client_training_seconds": float(sum(item["training_seconds"] for item in clients)),
            "ensemble_test_inference_seconds": inference_seconds,
            "ensemble_test_microseconds_per_record": 1e6 * inference_seconds / len(y_test),
        },
    )
    save_confusion_matrix(
        test_metrics["confusion_matrix"], class_names, run_directory / "confusion_matrix.png"
    )
    predictions = (ensemble_test[:, 1] >= threshold).astype(int)
    pd.DataFrame(
        {
            "row_index": split.test,
            "y_true": y_test,
            "y_pred": predictions,
            f"probability_{class_names[0]}": ensemble_test[:, 0],
            f"probability_{class_names[1]}": ensemble_test[:, 1],
        }
    ).to_csv(run_directory / "test_predictions.csv", index=False)

    resolved = copy.deepcopy(config)
    resolved.pop("_config_path", None)
    resolved["resolved"] = {
        "label_column": label_column,
        "positive_label": positive_label,
        "class_names": class_names,
        "label_mapping": {str(key): int(value) for key, value in raw_mapping.items()},
        "numeric_features": preprocessor.columns_,
        "ica_components": preprocessor.resolved_ica_components_,
        "threshold": threshold,
        "aggregation_weights": ensemble_weights.tolist(),
    }
    with (run_directory / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)
    write_json(run_directory / "environment.json", environment_manifest())
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "label_mapping": {str(key): int(value) for key, value in raw_mapping.items()},
            "class_names": class_names,
            "client_models": models,
            "aggregation_weights": ensemble_weights,
            "threshold": threshold,
        },
        run_directory / "federated_ensemble.joblib",
    )
    return run_directory
