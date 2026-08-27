from __future__ import annotations

import copy
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

from trustfed_incboost.config import project_root_from_config, resolve_from_project
from trustfed_incboost.data.audit import audit_dataframe, dataframe_fingerprint
from trustfed_incboost.data.loader import (
    choose_positive_label,
    infer_label_column,
    load_dataset,
    prepare_features,
)
from trustfed_incboost.data.splitting import make_split, split_manifest
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
from trustfed_incboost.models.preprocessing import NumericPreprocessor
from trustfed_incboost.reproducibility import (
    environment_manifest,
    seed_everything,
    utc_run_id,
    write_json,
)


def _resolved_dataset_pattern(config: dict, root: Path) -> str:
    raw = str(config["dataset"]["path"])
    if Path(raw).is_absolute():
        return raw
    return str((root / raw).resolve())


def audit_from_config(config: dict, output_path: Path | None = None) -> dict:
    root = project_root_from_config(config)
    dataset_config = config["dataset"]
    frame, paths = load_dataset(
        _resolved_dataset_pattern(config, root), dataset_config.get("max_rows")
    )
    label = infer_label_column(frame, dataset_config.get("label_column", "auto"))
    audit = audit_dataframe(
        frame,
        label,
        paths,
        excluded_from_fingerprint=dataset_config.get("drop_columns", []),
    )
    if output_path is not None:
        write_json(output_path, audit)
    return audit


def _ensure_each_class(indices: np.ndarray, encoded: np.ndarray, name: str) -> None:
    classes = np.unique(encoded)
    present = np.unique(encoded[indices])
    if len(present) != len(classes):
        raise ValueError(
            f"{name} partition is missing classes. Present={present.tolist()}, "
            f"all={classes.tolist()}. Choose a valid split or collect more data."
        )


def run_training(config: dict) -> Path:
    root = project_root_from_config(config)
    seed = int(config["project"].get("seed", 42))
    seed_everything(seed)

    output_base = resolve_from_project(config["output"]["directory"], root)
    run_directory = output_base / utc_run_id(config["project"]["name"])
    run_directory.mkdir(parents=True, exist_ok=False)

    frame, source_paths = load_dataset(
        _resolved_dataset_pattern(config, root), config["dataset"].get("max_rows")
    )
    label_column = infer_label_column(
        frame, config["dataset"].get("label_column", "auto")
    )
    positive_label = choose_positive_label(
        frame[label_column], config["dataset"].get("positive_label", "auto")
    )
    audit = audit_dataframe(
        frame,
        label_column,
        source_paths,
        excluded_from_fingerprint=config["dataset"].get("drop_columns", []),
    )
    write_json(run_directory / "dataset_audit.json", audit)

    X, y_raw = prepare_features(
        frame,
        label_column,
        config["dataset"].get("drop_columns", []),
    )
    encoder = LabelEncoder()
    label_mapping: dict = {}
    if positive_label is not None and y_raw.nunique() == 2:
        negative_values = [value for value in y_raw.unique() if value != positive_label]
        ordered = negative_values + [positive_label]
        mapping = {value: index for index, value in enumerate(ordered)}
        label_mapping = {str(value): int(index) for value, index in mapping.items()}
        y_encoded = y_raw.map(mapping).to_numpy(dtype=int)
        class_names = [str(value) for value in ordered]
        encoder.fit(y_raw)
    else:
        y_encoded = encoder.fit_transform(y_raw)
        class_names = [str(value) for value in encoder.classes_]
        label_mapping = {
            str(value): int(index) for index, value in enumerate(encoder.classes_)
        }
    y_series = pd.Series(y_encoded)

    split_config = config["split"]
    split = make_split(
        X,
        y_series,
        method=split_config["method"],
        seed=seed,
        test_size=float(split_config["test_size"]),
        validation_size=float(split_config.get("validation_size", 0.0)),
        group_column=split_config.get("group_column"),
        time_column=split_config.get("time_column"),
    )
    for name, indices in [("train", split.train), ("test", split.test)]:
        _ensure_each_class(indices, y_encoded, name)
    if len(split.validation):
        _ensure_each_class(split.validation, y_encoded, "validation")
    manifest = split_manifest(split, y_series)

    fingerprint_columns = [
        column
        for column in X.columns
        if column not in {"__source_file__", split_config.get("group_column"), split_config.get("time_column")}
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
    X_test = preprocessor.transform(X.iloc[split.test])
    X_validation = (
        preprocessor.transform(X.iloc[split.validation]) if len(split.validation) else None
    )
    y_train = y_encoded[split.train]
    y_test = y_encoded[split.test]
    y_validation = y_encoded[split.validation] if len(split.validation) else None

    model_config = config["model"]
    weights, weight_mapping = sample_weights(y_train, model_config["class_weight"])
    model = build_classifier(
        model_config["backend"],
        model_config.get("params", {}),
        len(class_names),
        seed,
    )
    start = time.perf_counter()
    model = fit_classifier(
        model,
        model_config["backend"],
        X_train,
        y_train,
        weights,
        X_validation,
        y_validation,
    )
    training_seconds = time.perf_counter() - start

    threshold = 0.5
    validation_metrics = None
    if X_validation is not None:
        validation_probabilities = predict_probabilities(model, X_validation)
        if len(class_names) == 2 and model_config.get("threshold_metric") == "macro_f1":
            threshold = select_binary_threshold(
                y_validation, validation_probabilities[:, 1]
            )
        validation_metrics = classification_metrics(
            y_validation, validation_probabilities, threshold, class_names
        )

    inference_start = time.perf_counter()
    test_probabilities = predict_probabilities(model, X_test)
    inference_seconds = time.perf_counter() - inference_start
    test_metrics = classification_metrics(
        y_test, test_probabilities, threshold, class_names
    )
    intervals = bootstrap_confidence_intervals(
        y_test,
        test_probabilities,
        threshold,
        int(config["evaluation"].get("bootstrap_repetitions", 500)),
        seed + 100,
    )

    resolved = copy.deepcopy(config)
    resolved.pop("_config_path", None)
    resolved["resolved"] = {
        "label_column": label_column,
        "positive_label": positive_label,
        "class_names": class_names,
        "label_mapping": label_mapping,
        "numeric_features": preprocessor.columns_,
        "ica_components": preprocessor.resolved_ica_components_,
        "class_weight_map": {str(key): value for key, value in weight_mapping.items()},
    }
    with (run_directory / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)
    write_json(run_directory / "environment.json", environment_manifest())
    write_json(
        run_directory / "metrics.json",
        {
            "validation": validation_metrics,
            "test": test_metrics,
            "bootstrap_95_ci": intervals,
            "training_seconds": training_seconds,
            "test_inference_seconds": inference_seconds,
            "test_microseconds_per_record": 1e6 * inference_seconds / len(y_test),
        },
    )
    save_confusion_matrix(
        test_metrics["confusion_matrix"], class_names, run_directory / "confusion_matrix.png"
    )
    pd.DataFrame(
        {
            "row_index": split.test,
            "y_true": y_test,
            "y_pred": (
                (test_probabilities[:, 1] >= threshold).astype(int)
                if len(class_names) == 2
                else test_probabilities.argmax(axis=1)
            ),
            **{
                f"probability_{class_names[index]}": test_probabilities[:, index]
                for index in range(len(class_names))
            },
        }
    ).to_csv(run_directory / "test_predictions.csv", index=False)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "label_encoder": encoder,
            "label_mapping": label_mapping,
            "class_names": class_names,
            "positive_label": positive_label,
            "model": model,
            "threshold": threshold,
        },
        run_directory / "model.joblib",
    )
    return run_directory
