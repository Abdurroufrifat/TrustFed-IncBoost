from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from trustfed_incboost.data.audit import dataframe_fingerprint


@dataclass(frozen=True)
class SplitResult:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    method: str


def _group_split(indices: np.ndarray, groups: pd.Series, test_size: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pos, test_pos = next(splitter.split(indices, groups=groups.iloc[indices]))
    return indices[train_pos], indices[test_pos]


def _validate_partitions(result: SplitResult, n_rows: int) -> None:
    sets = [set(result.train), set(result.validation), set(result.test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("Split partitions overlap.")
    if len(set.union(*sets)) != n_rows:
        raise AssertionError("Split partitions do not cover every row exactly once.")
    if len(result.train) == 0 or len(result.test) == 0:
        raise ValueError("Training and test partitions must be non-empty.")


def make_split(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    seed: int,
    test_size: float,
    validation_size: float,
    group_column: str | None = None,
    time_column: str | None = None,
) -> SplitResult:
    n_rows = len(X)
    indices = np.arange(n_rows)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 <= validation_size < 1 - test_size:
        raise ValueError("validation_size must be non-negative and leave training data.")

    if method == "temporal":
        if not time_column or time_column not in X.columns:
            raise ValueError("Temporal splitting requires split.time_column in the features.")
        ordered = X.assign(__row_index__=indices).sort_values(time_column)["__row_index__"].to_numpy()
        test_start = int(round(n_rows * (1 - test_size)))
        val_start = int(round(n_rows * (1 - test_size - validation_size)))
        result = SplitResult(ordered[:val_start], ordered[val_start:test_start], ordered[test_start:], method)
    elif method in {"group", "fingerprint_disjoint"}:
        if method == "group":
            if not group_column or group_column not in X.columns:
                raise ValueError("Group splitting requires split.group_column in the features.")
            groups = X[group_column].astype("string").fillna("<NA>")
        else:
            fingerprint_columns = [
                column for column in X.columns if column not in {group_column, time_column, "__source_file__"}
            ]
            groups = dataframe_fingerprint(X[fingerprint_columns]).astype("string")
        train_val, test = _group_split(indices, groups, test_size, seed)
        if validation_size > 0:
            relative_val = validation_size / (1 - test_size)
            train, validation = _group_split(train_val, groups, relative_val, seed + 1)
        else:
            train, validation = train_val, np.array([], dtype=int)
        result = SplitResult(train, validation, test, method)
    elif method == "random_stratified":
        train_val, test = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=y
        )
        if validation_size > 0:
            relative_val = validation_size / (1 - test_size)
            train, validation = train_test_split(
                train_val,
                test_size=relative_val,
                random_state=seed + 1,
                stratify=y.iloc[train_val],
            )
        else:
            train, validation = train_val, np.array([], dtype=int)
        result = SplitResult(np.asarray(train), np.asarray(validation), np.asarray(test), method)
    else:
        raise ValueError(f"Unknown split method: {method}")

    _validate_partitions(result, n_rows)
    return result


def split_manifest(result: SplitResult, y: pd.Series) -> dict:
    def describe(indices: np.ndarray) -> dict:
        return {
            "rows": int(len(indices)),
            "label_distribution": {
                str(key): int(value) for key, value in y.iloc[indices].value_counts().items()
            },
        }

    return {
        "method": result.method,
        "train": describe(result.train),
        "validation": describe(result.validation),
        "test": describe(result.test),
    }
