from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trustfed_incboost.reproducibility import sha256_file


def dataframe_fingerprint(frame: pd.DataFrame) -> pd.Series:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_float_dtype(normalized[column]):
            normalized[column] = normalized[column].round(12)
        elif pd.api.types.is_object_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("string").fillna("<NA>")
    return pd.util.hash_pandas_object(normalized, index=False).astype("uint64")


def audit_dataframe(
    frame: pd.DataFrame,
    label_column: str,
    source_paths: list[Path],
    excluded_from_fingerprint: list[str] | None = None,
) -> dict[str, Any]:
    excluded = set(excluded_from_fingerprint or []) | {label_column, "__source_file__"}
    feature_columns = [column for column in frame.columns if column not in excluded]
    fingerprints = dataframe_fingerprint(frame[feature_columns])
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_features = int(fingerprints.duplicated(keep=False).sum())
    conflicting = (
        pd.DataFrame({"fingerprint": fingerprints, "label": frame[label_column]})
        .groupby("fingerprint", observed=True)["label"]
        .nunique()
    )
    conflicting_fingerprints = int((conflicting > 1).sum())
    missing_by_column = frame.isna().sum().sort_values(ascending=False)
    nunique = frame.nunique(dropna=False)
    near_constant = [
        str(column)
        for column in frame.columns
        if len(frame) and nunique[column] <= 1
    ]
    numeric_columns = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
    non_numeric_columns = [column for column in frame.columns if column not in numeric_columns]
    files = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_paths
    ]
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "label_column": label_column,
        "label_distribution": {
            str(key): int(value) for key, value in frame[label_column].value_counts(dropna=False).items()
        },
        "duplicate_complete_rows": duplicate_rows,
        "rows_in_duplicate_feature_groups": duplicate_features,
        "conflicting_feature_fingerprints": conflicting_fingerprints,
        "numeric_column_count": len(numeric_columns),
        "non_numeric_columns": [str(column) for column in non_numeric_columns],
        "constant_columns": near_constant,
        "missing_total": int(frame.isna().sum().sum()),
        "missing_top_columns": {
            str(key): int(value)
            for key, value in missing_by_column.head(20).items()
            if value > 0
        },
        "source_files": files,
    }

