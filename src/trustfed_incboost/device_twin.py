"""Causal, site-local device-state surrogate for a new temporal experiment.

This module is independent of the frozen EHMS confirmation. Its reference is
fitted on benign, training-side observations only; validation/test labels are
never passed to the replay method.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def fuse_probabilities(detector: np.ndarray, twin: np.ndarray,
                       *, twin_weight: float = .25) -> np.ndarray:
    """Noisy-OR blend of a frozen detector and a bounded twin risk score."""
    detector = np.asarray(detector, dtype=float)
    twin = np.asarray(twin, dtype=float)
    if detector.shape != twin.shape:
        raise ValueError("detector and twin scores need the same shape")
    if not 0 <= twin_weight <= 1:
        raise ValueError("twin_weight must be in [0, 1]")
    if not np.isfinite(detector).all() or not np.isfinite(twin).all():
        raise ValueError("scores must be finite")
    if ((detector < 0) | (detector > 1)).any() or ((twin < 0) | (twin > 1)).any():
        raise ValueError("scores must be probabilities in [0, 1]")
    return 1 - (1 - detector) * (1 - twin_weight * twin)


@dataclass
class DeviceStateTwin:
    device_column: str
    time_column: str
    feature_columns: tuple[str, ...]
    states: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    risk_scales: dict[str, float]
    last_times: dict[str, pd.Timestamp]
    alpha: float
    detector_gate: float
    max_normal_deviation: float

    @classmethod
    def fit(cls, train: pd.DataFrame, *, device_column: str, time_column: str,
            feature_columns: list[str], label_column: str,
            min_benign: int = 20, alpha: float = .05,
            detector_gate: float = .5, max_normal_deviation: float = 4.) -> "DeviceStateTwin":
        if not feature_columns or len(set(feature_columns)) != len(feature_columns):
            raise ValueError("nonempty, distinct twin feature columns are required")
        required = [device_column, time_column, label_column, *feature_columns]
        missing = sorted(set(required) - set(train.columns))
        if missing:
            raise ValueError(f"missing required twin columns: {missing}")
        if not (0 < alpha <= 1 and 0 < detector_gate < 1 and max_normal_deviation > 0):
            raise ValueError("invalid twin parameters")
        times = pd.to_datetime(train[time_column], utc=True, errors="coerce")
        if times.isna().any() or train[device_column].isna().any():
            raise ValueError("training device IDs and timestamps must be present and parseable")
        clean = train.assign(__time__=times, __device__=train[device_column].astype(str))
        if clean.duplicated(["__device__", "__time__"]).any():
            raise ValueError("duplicate timestamps within a training device")
        states: dict[str, np.ndarray] = {}
        scales: dict[str, np.ndarray] = {}
        risk_scales: dict[str, float] = {}
        last_times: dict[str, pd.Timestamp] = {}
        for device, group in clean.groupby("__device__", sort=True):
            benign = group.loc[group[label_column] == 0, feature_columns]
            if len(benign) < min_benign:
                raise ValueError(f"device {device!r} has fewer than {min_benign} benign training rows")
            values = benign.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"missing/nonfinite benign twin features for device {device!r}")
            center = np.median(values, axis=0)
            mad = np.median(np.abs(values - center), axis=0) * 1.4826
            spread = np.maximum.reduce([mad, .1 * values.std(axis=0),
                                        np.full(len(feature_columns), 1e-3)])
            deviations = np.sqrt(np.mean(((values-center)/spread)**2, axis=1))
            states[device] = center.copy()
            scales[device] = spread
            risk_scales[device] = max(float(np.quantile(deviations, .95)), .1)
            last_times[device] = group["__time__"].max()
        return cls(device_column, time_column, tuple(feature_columns), states,
                   scales, risk_scales, last_times, alpha, detector_gate, max_normal_deviation)

    def observe(self, device: str, timestamp: object, measurement: object,
                detector_probability: float) -> dict[str, float | bool]:
        device = str(device)
        if device not in self.states:
            raise ValueError(f"unknown device {device!r}; no training-only reference exists")
        time = pd.to_datetime(timestamp, utc=True, errors="coerce")
        if pd.isna(time) or time <= self.last_times[device]:
            raise ValueError(f"timestamps must be strictly increasing for device {device!r}")
        vector = np.asarray(measurement, dtype=float)
        if vector.shape != self.states[device].shape or not np.isfinite(vector).all():
            raise ValueError("measurement shape/values do not match the fitted device")
        p = float(detector_probability)
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("detector probability must be in [0, 1]")
        deviation = float(np.sqrt(np.mean(((vector-self.states[device])/self.scales[device])**2)))
        twin_score = float(-np.expm1(-deviation/self.risk_scales[device]))
        safe = p < self.detector_gate and deviation < self.max_normal_deviation
        if safe:
            self.states[device] = (1-self.alpha)*self.states[device] + self.alpha*vector
        self.last_times[device] = time
        return {"twin_score": twin_score, "deviation": deviation, "state_updated": bool(safe)}

    def replay(self, frame: pd.DataFrame, detector_probabilities: np.ndarray) -> pd.DataFrame:
        """Replay only chronologically sorted, previously unseen observations."""
        if len(frame) != len(detector_probabilities):
            raise ValueError("frame and detector probabilities must have equal length")
        if frame.index.has_duplicates:
            raise ValueError("frame row indices must be unique")
        times = pd.to_datetime(frame[self.time_column], utc=True, errors="coerce")
        if times.isna().any() or not times.is_monotonic_increasing:
            raise ValueError("replay rows must have valid, globally ordered timestamps")
        records = []
        for position, (_, row) in enumerate(frame.iterrows()):
            result = self.observe(row[self.device_column], times.iloc[position],
                                  row.loc[list(self.feature_columns)].to_numpy(dtype=float),
                                  float(detector_probabilities[position]))
            records.append(result)
        return pd.DataFrame.from_records(records, index=frame.index)
