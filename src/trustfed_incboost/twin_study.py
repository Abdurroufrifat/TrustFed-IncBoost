"""Replay a frozen CAG-FE artifact with a site-local device-state surrogate.

New temporal study only: the existing unseen-seed EHMS results are untouched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from trustfed_incboost.data.loader import infer_label_column, load_dataset, prepare_features
from trustfed_incboost.data.splitting import make_split
from trustfed_incboost.device_twin import DeviceStateTwin, fuse_probabilities
from trustfed_incboost.evaluation.metrics import classification_metrics, select_binary_threshold
from trustfed_incboost.models.baseline import predict_probabilities
from trustfed_incboost.models.federated import aggregate_probabilities
from trustfed_incboost.reproducibility import sha256_file, write_json


def validate_temporal_stream(frame: pd.DataFrame, config: dict,
                             device_column: str, time_column: str,
                             feature_columns: list[str]) -> pd.Series:
    if config.get("split", {}).get("method") != "temporal":
        raise ValueError("a new temporal training run is required; old EHMS random splits cannot be replayed")
    if config["split"].get("time_column") != time_column:
        raise ValueError("time_column must match the temporal training split")
    missing = sorted(set([device_column, time_column, *feature_columns]) - set(frame.columns))
    if missing:
        raise ValueError(f"required device/time/twin columns absent: {missing}")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("twin feature columns must be nonempty and distinct")
    if frame[device_column].isna().any() or frame[time_column].isna().any():
        raise ValueError("device identifiers and timestamps must be complete")
    times = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    if times.isna().any():
        raise ValueError("timestamps must be parseable")
    if pd.DataFrame({"device": frame[device_column].astype(str), "time": times}).duplicated().any():
        raise ValueError("duplicate timestamps within a device are not allowed")
    if not frame[feature_columns].apply(pd.to_numeric, errors="coerce").notna().all().all():
        raise ValueError("twin features must be numeric and nonmissing")
    return times


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float,
             class_names: list[str]) -> dict:
    probabilities = np.column_stack([1-scores, scores])
    output = classification_metrics(y, probabilities, threshold, class_names)
    output["attack_f1"] = float(f1_score(y, scores >= threshold, zero_division=0))
    return output


def episode_delays(group: pd.DataFrame, alarm_column: str) -> dict:
    """Count contiguous labeled attack episodes and first-alert delays."""
    if not set(["timestamp", "y_true", alarm_column]).issubset(group.columns):
        raise ValueError("episode table needs timestamps, labels and the specified alarm")
    ordered = group.sort_values("timestamp", kind="stable")
    timestamps = pd.to_datetime(ordered["timestamp"], utc=True, errors="raise")
    starts: list[pd.Timestamp] = []
    delays: list[float] = []
    previous_attack = False
    detected = False
    for time, (_, row) in zip(timestamps, ordered.iterrows(), strict=True):
        attacked = int(row.y_true) == 1
        if attacked and not previous_attack:
            starts.append(time)
            detected = False
        if attacked and bool(row[alarm_column]) and not detected:
            delays.append((time-starts[-1]).total_seconds()/60)
            detected = True
        previous_attack = attacked
    return {"attack_episodes": len(starts), "missed_episodes": len(starts)-len(delays),
            "detected_delay_minutes": delays}


def _probabilities(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    matrix = artifact["preprocessor"].transform(X)
    clients = [predict_probabilities(model, matrix) for model in artifact["client_models"]]
    return aggregate_probabilities(clients, np.asarray(artifact["aggregation_weights"]))[:, 1]


def _ordered(frame: pd.DataFrame, indices: np.ndarray, times: pd.Series) -> pd.DataFrame:
    part = frame.iloc[indices].copy()
    part.index = np.asarray(indices, dtype=int)
    return part.assign(__time__=times.iloc[indices].to_numpy()).sort_values(
        ["__time__"], kind="stable").drop(columns="__time__")


def run_twin_study(run_directory: str | Path, *, root: str | Path,
                   device_column: str, time_column: str,
                   features: list[str], twin_weight: float = .25) -> Path:
    run_directory, root = Path(run_directory).resolve(), Path(root).resolve()
    if not run_directory.is_dir() or not (run_directory / "federated_ensemble.joblib").is_file():
        raise FileNotFoundError("run_directory needs a trained federated_ensemble.joblib")
    with (run_directory / "resolved_config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["federated"]["aggregation"]["method"] != "trust_aware_v2":
        raise ValueError("this study requires a chance-gated TrustFed V2 (CAG-FE) artifact")
    dataset_path = Path(config["dataset"]["path"])
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    frame, files = load_dataset(dataset_path, config["dataset"].get("max_rows"))
    audited = json.loads((run_directory / "dataset_audit.json").read_text(encoding="utf-8"))
    actual_files = {str(path.resolve()): sha256_file(path) for path in files}
    audited_files = {str(Path(item["path"]).resolve()): item["sha256"]
                     for item in audited["source_files"]}
    if actual_files != audited_files or len(frame) != audited["rows"]:
        raise ValueError("dataset bytes or row count changed since the frozen detector was trained")
    times = validate_temporal_stream(frame, config, device_column, time_column, features)
    partition = config["federated"]["partition"]
    if partition.get("method") != "site_group":
        raise ValueError("the twin experiment requires actual site-grouped federated clients")
    site_column = partition["site_column"]
    if site_column not in frame.columns or frame.groupby(device_column)[site_column].nunique().max() != 1:
        raise ValueError("every device must belong to exactly one recorded site")
    label_column = infer_label_column(frame, config["dataset"].get("label_column", "auto"))
    X, raw_labels = prepare_features(frame, label_column, config["dataset"].get("drop_columns", []))
    artifact = joblib.load(run_directory / "federated_ensemble.joblib")
    if set([device_column, time_column, site_column]) & set(artifact["preprocessor"].columns_):
        raise ValueError("device/site/time identifiers leaked into the fitted detector features")
    mapping = artifact["label_mapping"]
    y = raw_labels.map(lambda value: mapping.get(str(value))).to_numpy()
    if pd.isna(y).any() or set(np.unique(y)) != {0, 1}:
        raise ValueError("the replay labels do not match the saved binary model mapping")
    y = y.astype(int)
    split_config = config["split"]
    split = make_split(X, pd.Series(y), "temporal", int(config["project"]["seed"]),
                       float(split_config["test_size"]), float(split_config["validation_size"]),
                       time_column=time_column)
    if not len(split.validation):
        raise ValueError("a nonempty validation interval is required")
    if not times.iloc[split.train].max() < times.iloc[split.validation].min() or not (
        times.iloc[split.validation].max() < times.iloc[split.test].min()
    ):
        raise ValueError("training/validation/test boundary cuts across a shared timestamp")
    training = _ordered(frame, split.train, times)
    validation = _ordered(frame, split.validation, times)
    test = _ordered(frame, split.test, times)
    train_reference = training.assign(__clean_label__=y[training.index])
    model_threshold = float(artifact["threshold"])
    twin = DeviceStateTwin.fit(
        train_reference, device_column=device_column, time_column=time_column,
        feature_columns=features, label_column="__clean_label__",
        min_benign=20, alpha=.05, detector_gate=model_threshold,
        max_normal_deviation=4.,
    )
    probabilities_v = _probabilities(artifact, X.iloc[validation.index])
    probabilities_t = _probabilities(artifact, X.iloc[test.index])
    previous_test = pd.read_csv(run_directory / "test_predictions.csv").set_index("row_index")
    positive_column = f"probability_{artifact['class_names'][1]}"
    if positive_column not in previous_test or set(previous_test.index) != set(test.index):
        raise ValueError("saved test prediction rows do not match reconstructed temporal split")
    expected = previous_test.loc[test.index]
    if not np.allclose(probabilities_t, expected[positive_column].to_numpy(), atol=1e-8, rtol=1e-6):
        raise ValueError("recomputed detector probabilities differ from saved test predictions")
    if not np.array_equal(y[test.index], expected["y_true"].to_numpy(dtype=int)):
        raise ValueError("saved test labels differ from the reconstructed labels")

    validation_risk = twin.replay(validation, probabilities_v)["twin_score"].to_numpy()
    fused_v = fuse_probabilities(probabilities_v, validation_risk, twin_weight=twin_weight)
    hybrid_threshold = select_binary_threshold(y[validation.index], fused_v)
    twin_threshold = select_binary_threshold(y[validation.index], validation_risk)
    start = time.perf_counter()
    test_readings = twin.replay(test, probabilities_t)
    replay_seconds = time.perf_counter() - start
    fused_t = fuse_probabilities(probabilities_t, test_readings["twin_score"].to_numpy(),
                                 twin_weight=twin_weight)
    y_test = y[test.index]
    class_names = list(artifact["class_names"])
    metrics = {
        "detector_only": _metrics(y_test, probabilities_t, model_threshold, class_names),
        "twin_only": _metrics(y_test, test_readings["twin_score"].to_numpy(), twin_threshold, class_names),
        "detector_plus_twin": _metrics(y_test, fused_t, hybrid_threshold, class_names),
    }
    output = run_directory / "twin_study"
    if output.exists():
        raise FileExistsError(f"twin results already exist at {output}; preserve prior analyses")
    output.mkdir()
    per_row = pd.DataFrame({"row_index": test.index, "timestamp": test[time_column].astype(str).to_numpy(),
                            "device_id": test[device_column].astype(str).to_numpy(),
                            "y_true": y_test, "detector_probability": probabilities_t,
                            "twin_score": test_readings["twin_score"].to_numpy(),
                            "hybrid_probability": fused_t,
                            "twin_deviation": test_readings["deviation"].to_numpy(),
                            "state_updated": test_readings["state_updated"].to_numpy(),
                            "detector_alarm": probabilities_t >= model_threshold,
                            "hybrid_alarm": fused_t >= hybrid_threshold})
    per_row.to_csv(output / "test_replay.csv", index=False)
    by_device = []
    for device, group in per_row.groupby("device_id", sort=True):
        normal = group.loc[group.y_true == 0]
        span_hours = (pd.to_datetime(group.timestamp, utc=True).max() -
                      pd.to_datetime(group.timestamp, utc=True).min()).total_seconds()/3600
        detector_episodes = episode_delays(group, "detector_alarm")
        hybrid_episodes = episode_delays(group, "hybrid_alarm")
        by_device.append({"device_id": device, "rows": len(group),
                          "normal_rows": len(normal),
                          "detector_false_alarms": int(normal.detector_alarm.sum()),
                          "hybrid_false_alarms": int(normal.hybrid_alarm.sum()),
                          "detector_false_alarms_per_device_hour": float(normal.detector_alarm.sum()/span_hours) if span_hours else None,
                          "hybrid_false_alarms_per_device_hour": float(normal.hybrid_alarm.sum()/span_hours) if span_hours else None,
                          "detector_episodes": detector_episodes["attack_episodes"],
                          "detector_missed_episodes": detector_episodes["missed_episodes"],
                          "hybrid_missed_episodes": hybrid_episodes["missed_episodes"],
                          "detector_median_delay_minutes": float(np.median(detector_episodes["detected_delay_minutes"]))
                          if detector_episodes["detected_delay_minutes"] else None,
                          "hybrid_median_delay_minutes": float(np.median(hybrid_episodes["detected_delay_minutes"]))
                          if hybrid_episodes["detected_delay_minutes"] else None,
                          "attack_rows": int(group.y_true.sum()),
                          "detector_attack_recall": float(group.loc[group.y_true == 1, "detector_alarm"].mean())
                          if (group.y_true == 1).any() else None,
                          "hybrid_attack_recall": float(group.loc[group.y_true == 1, "hybrid_alarm"].mean())
                          if (group.y_true == 1).any() else None})
    pd.DataFrame(by_device).to_csv(output / "by_device.csv", index=False)
    write_json(output / "summary.json", {"status": "new_temporal_experiment",
        "dataset_sha256": actual_files, "artifact_sha256": sha256_file(run_directory / "federated_ensemble.joblib"),
        "is_synthetic_smoke": bool(config.get("twin_study", {}).get("synthetic_smoke", False)),
        "site_count": len(twin.states), "feature_columns": features,
        "train_rows": len(split.train), "validation_rows": len(split.validation), "test_rows": len(split.test),
        "twin_weight": twin_weight, "thresholds": {"detector": model_threshold,
        "twin": twin_threshold, "hybrid": hybrid_threshold},
        "metrics": metrics, "test_replay_seconds": replay_seconds,
        "test_replay_microseconds_per_record": 1e6*replay_seconds/len(test),
        "limitations": "Training is temporally separated. The detector still reuses its validation interval for early stopping, trust and threshold selection. A stronger study should separate those roles and use external device streams."})
    return output
