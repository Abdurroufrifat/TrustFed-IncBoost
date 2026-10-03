"""Run an exploratory, recording-disjoint simulated-client detector/twin comparison.

Inputs: stage-2 sensor_windows.csv and record_split_manifest.csv, and the
original PhysioNet training.zip. Clinical alarm labels are never prediction targets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

TWIN_COLS = ["pleth_std", "pleth_iqr", "pleth_diff_std", "pleth_flat_fraction",
             "pleth_missing_fraction", "pleth_ecg_diff_std_ratio"]


def load_preparation():
    path = Path(__file__).with_name("14_prepare_physionet_sensor_study.py")
    if not path.is_file():
        raise FileNotFoundError(f"Extract the earlier preparation ZIP first: {path}")
    spec = importlib.util.spec_from_file_location("physionet_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def baseline_features(archive, records, preparation):
    baselines = {}
    for record in records:
        ecg, pleth, rate = preparation.decode(archive, record)
        window = rate * 5
        clean = [preparation.stats(ecg[start:start + window], pleth[start:start + window])
                 for start in range(0, 6 * window, window)]
        baselines[record] = np.median([[d[k] for k in TWIN_COLS] for d in clean], axis=0)
    return baselines


def twin_residuals(frame, baselines, scale):
    reference = np.stack([baselines[record] for record in frame.record])
    current = frame[TWIN_COLS].to_numpy(dtype=float)
    # Per-record, pre-observation reference. No test labels or paired clean rows
    # are used to predict any validation/test window.
    delta = np.abs(current - reference) / scale
    return np.max(delta, axis=1)


def scores(y, prob, threshold):
    pred = prob >= threshold
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return {"f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y, prob)),
            "roc_auc": float(roc_auc_score(y, prob)),
            "confusion_matrix": cm, "threshold": float(threshold)}


def select_threshold(y, prob):
    # Select on validation only, with conservative tie breaking.
    candidates = np.unique(np.quantile(prob, np.linspace(0, 1, 501)))
    ranked = [(f1_score(y, prob >= t, zero_division=0), t) for t in candidates]
    return float(max(ranked)[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, default=Path("data/raw/physionet_2015/training.zip"))
    p.add_argument("--prepared", type=Path, default=Path("outputs/physionet_2015_prepared"))
    p.add_argument("--output", type=Path, default=Path("outputs/physionet_2015_experiment"))
    args = p.parse_args()
    preparation = load_preparation()
    summary = json.loads((args.prepared / "preparation_summary.json").read_text())
    if source_hash(args.zip) != summary["source_sha256"]:
        raise ValueError("training.zip SHA256 does not match the prepared dataset")
    frame = pd.read_csv(args.prepared / "sensor_windows.csv", keep_default_na=True)
    manifest = pd.read_csv(args.prepared / "record_split_manifest.csv")
    if frame.duplicated(["record", "window_start_seconds", "fault"]).any():
        raise ValueError("Repeated window/fault rows")
    if manifest.record.duplicated().any() or set(frame.record) != set(manifest.record):
        raise ValueError("Missing or repeated recording in manifest")
    split_map = manifest.set_index("record").split.to_dict()
    if any(split_map[record] != split for record, split in zip(frame.record, frame.split)):
        raise ValueError("Split mismatch")
    forbidden = {"sensor_fault_label", "fault", "clinical_alarm_true_metadata",
                 "record", "split", "simulated_client", "window_start_seconds"}
    features = sorted(set(frame.columns) - forbidden)
    if not features:
        raise ValueError("Missing feature columns")
    # Stage 2 intentionally uses NaN for injected dropout; aggregate pleth
    # statistics on that row may also be NaN. Encode those as zero, while
    # keeping the explicit missing fraction feature.
    bad = frame[features].isna().any(axis=1)
    if (bad & (frame.fault != "dropout")).any():
        raise ValueError("Unexpected missing features outside dropout injection")
    frame[features] = frame[features].fillna(0.0)
    train = frame[frame.split == "train"]
    val = frame[frame.split == "validation"]
    test = frame[frame.split == "test"]
    yval, ytest = val.sensor_fault_label.to_numpy(), test.sensor_fault_label.to_numpy()
    val_probs, test_probs, counts = [], [], []
    for client in range(5):
        local = train[train.simulated_client == client]
        if local.sensor_fault_label.nunique() != 2:
            raise ValueError(f"Training client {client} lacks one class")
        model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15,
                                               min_samples_leaf=30, random_state=42 + client)
        model.fit(local[features].to_numpy(), local.sensor_fault_label.to_numpy())
        val_probs.append(model.predict_proba(val[features].to_numpy())[:, 1])
        test_probs.append(model.predict_proba(test[features].to_numpy())[:, 1])
        counts.append(len(local))
        print(f"Fitted simulated client {client + 1}/5 ({len(local)} rows)", flush=True)
    weights = np.array(counts, dtype=float) / sum(counts)
    detector_val = np.average(val_probs, axis=0, weights=weights)
    detector_test = np.average(test_probs, axis=0, weights=weights)
    equal_val = np.mean(val_probs, axis=0)
    equal_test = np.mean(test_probs, axis=0)

    # Causal initial 30-second references for each record, including held-out
    # records. References are from raw, unmodified signals before injected faults.
    with ZipFile(args.zip) as archive:
        baselines = baseline_features(archive, manifest.record, preparation)
    train_clean = train[train.sensor_fault_label == 0]
    raw_scale = np.stack([baselines[record] for record in train_clean.record])
    diffs = np.abs(train_clean[TWIN_COLS].to_numpy() - raw_scale)
    scale = np.maximum(np.quantile(diffs, .9, axis=0), .01)
    # Train-only anomaly-score calibration; no attack labels used for twin fit.
    train_resid = twin_residuals(train_clean, baselines, scale)
    reference_distribution = np.sort(train_resid)
    def twin_probability(part):
        residual = twin_residuals(part, baselines, scale)
        return np.searchsorted(reference_distribution, residual, side="right") / len(reference_distribution)
    twin_val, twin_test = twin_probability(val), twin_probability(test)
    # Prespecified fusion weight, threshold chosen on validation only.
    fused_val = 1 - (1 - detector_val) * (1 - .25 * twin_val)
    fused_test = 1 - (1 - detector_test) * (1 - .25 * twin_test)
    options = {"uniform_client_detector": (equal_val, equal_test),
               "sample_weighted_detector": (detector_val, detector_test),
               "signal_twin": (twin_val, twin_test),
               "detector_twin_fusion": (fused_val, fused_test)}
    result = {"source_sha256": summary["source_sha256"], "feature_columns": features,
              "twin_feature_columns": TWIN_COLS, "twin_scale": scale.tolist(),
              "training_client_counts": counts, "client_weights": weights.tolist(),
              "study_type": "simulated clients; controlled synthetic sensor faults; recording-level split",
              "methods": {}, "fault_recall": {}, "limitations": summary["important_limitations"] + [
                  "early clean reference from each held-out recording is assumed available",
                  "no measured cyberattack or real deployment evidence",
                  "this standalone stage does not establish CAG-FE poisoning robustness"]}
    for name, (v, t) in options.items():
        threshold = select_threshold(yval, v)
        result["methods"][name] = {"validation": scores(yval, v, threshold),
                                   "test": scores(ytest, t, threshold)}
        predictions = t >= threshold
        result["fault_recall"][name] = {
            fault: float(np.mean(predictions[test.fault.to_numpy() == fault]))
            for fault in ("flatline", "dropout", "gain", "replay")}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: result["methods"][name]["test"] for name in options}, indent=2))
    print("Saved:", args.output / "results.json")

if __name__ == "__main__":
    main()
