"""Check a stream before training a detector or interpreting twin results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trustfed_incboost.config import load_config, project_root_from_config
from trustfed_incboost.data.loader import load_dataset, prepare_features
from trustfed_incboost.data.splitting import make_split
from trustfed_incboost.twin_study import validate_temporal_stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device-column", required=True)
    parser.add_argument("--time-column", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--min-benign", type=int, default=20)
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root_from_config(config)
    source = Path(config["dataset"]["path"])
    frame, files = load_dataset(source if source.is_absolute() else root/source,
                                config["dataset"].get("max_rows"))
    times = validate_temporal_stream(frame, config, args.device_column,
                                     args.time_column, args.features)
    label_column = config["dataset"]["label_column"]
    positive = config["dataset"]["positive_label"]
    if frame[label_column].nunique() != 2 or positive not in set(frame[label_column]):
        raise ValueError("binary labels and configured positive_label are required")
    X, _ = prepare_features(frame, label_column, config["dataset"].get("drop_columns", []))
    if args.device_column not in X or args.time_column not in X:
        raise ValueError("device ID and time must remain in split features (as nonnumeric metadata)")
    if pd.api.types.is_numeric_dtype(X[args.device_column]) or pd.api.types.is_numeric_dtype(X[args.time_column]):
        raise ValueError("encode device IDs/times as strings so they cannot leak into detector features")
    binary = (frame[label_column] == positive).astype(int)
    split = make_split(X, binary, method="temporal", seed=int(config["project"]["seed"]),
                       test_size=float(config["split"]["test_size"]),
                       validation_size=float(config["split"]["validation_size"]),
                       time_column=args.time_column)
    if not times.iloc[split.train].max() < times.iloc[split.validation].min() or not times.iloc[split.validation].max() < times.iloc[split.test].min():
        raise ValueError("split crosses tied timestamps; revise data or time boundaries")
    train = frame.iloc[split.train]
    devices = set(frame[args.device_column].astype(str))
    train_devices = set(train[args.device_column].astype(str))
    if devices != train_devices:
        raise ValueError(f"test/validation devices lack training references: {sorted(devices-train_devices)}")
    benign_counts = train.loc[train[label_column] != positive].groupby(args.device_column).size()
    insufficient = {str(d): int(benign_counts.get(d, 0)) for d in devices
                    if int(benign_counts.get(d, 0)) < args.min_benign}
    if insufficient:
        raise ValueError(f"insufficient benign training rows per device: {insufficient}")
    print(json.dumps({
        "status": "suitable_for_temporal_smoke_or_experiment",
        "source_files": [str(p) for p in files],
        "devices": len(devices),
        "rows": {name: {"total": len(indices), "attacks": int(binary.iloc[indices].sum())}
                 for name, indices in (("train", split.train), ("validation", split.validation),
                                       ("test", split.test))},
        "earliest_test_timestamp": times.iloc[split.test].min().isoformat(),
        "disclaimer": "Schema and split checks alone do not validate physical fidelity or clinical use.",
    }, indent=2))


if __name__ == "__main__":
    main()
