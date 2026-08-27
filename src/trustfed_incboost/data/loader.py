from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_CANDIDATES = (
    "label",
    "Label",
    "class",
    "Class",
    "attack",
    "Attack",
    "target",
    "Target",
    "binary_label",
)


def _read_one(path: Path, max_rows: int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, nrows=max_rows, low_memory=False)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        if max_rows is not None:
            frame = frame.head(max_rows)
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame["__source_file__"] = path.name
    return frame


def load_dataset(path_pattern: str | Path, max_rows: int | None = None) -> tuple[pd.DataFrame, list[Path]]:
    matches = sorted(Path(item).resolve() for item in glob.glob(str(path_pattern)))
    if not matches and Path(path_pattern).exists():
        matches = [Path(path_pattern).resolve()]
    if not matches:
        raise FileNotFoundError(
            f"No dataset files matched: {path_pattern}. See docs/DATASETS.md."
        )
    frames: list[pd.DataFrame] = []
    remaining = max_rows
    for path in matches:
        if remaining is not None and remaining <= 0:
            break
        frame = _read_one(path, remaining)
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    return combined, matches


def infer_label_column(frame: pd.DataFrame, configured: Any = "auto") -> str:
    if configured not in {None, "auto"}:
        if configured not in frame.columns:
            raise KeyError(
                f"Configured label column '{configured}' was not found. "
                f"Available columns: {list(frame.columns)}"
            )
        return str(configured)
    for candidate in LABEL_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    low_cardinality = [
        column
        for column in frame.columns
        if column != "__source_file__" and 1 < frame[column].nunique(dropna=False) <= 20
    ]
    raise ValueError(
        "Could not infer the label column. Set dataset.label_column in the YAML. "
        f"Possible low-cardinality columns: {low_cardinality}"
    )


def choose_positive_label(y: pd.Series, configured: Any = "auto") -> Any:
    values = list(y.dropna().unique())
    if configured not in {None, "auto"}:
        if configured not in values:
            configured_str = str(configured).lower()
            for value in values:
                if str(value).lower() == configured_str:
                    return value
            raise ValueError(f"Positive label {configured!r} not present in {values!r}")
        return configured
    if len(values) != 2:
        return None
    preferred = {"attack", "abnormal", "anomaly", "malicious", "1", "true", "yes"}
    for value in values:
        if str(value).strip().lower() in preferred:
            return value
    counts = y.value_counts()
    return counts.index[-1]


def prepare_features(
    frame: pd.DataFrame,
    label_column: str,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    drop = set(drop_columns or [])
    drop.add(label_column)
    missing_drops = sorted(column for column in drop if column not in frame.columns)
    drop = {column for column in drop if column in frame.columns}
    if missing_drops:
        print(f"Warning: configured drop columns not present: {missing_drops}")
    X = frame.drop(columns=sorted(drop)).copy()
    y = frame[label_column].copy()
    if y.isna().any():
        raise ValueError(f"Label column '{label_column}' contains missing values.")
    return X, y

