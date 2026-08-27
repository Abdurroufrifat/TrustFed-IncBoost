from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def main() -> None:
    destination = Path("data/raw/synthetic/synthetic_iot.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    X, y = make_classification(
        n_samples=6000,
        n_features=24,
        n_informative=14,
        n_redundant=5,
        weights=[0.86, 0.14],
        class_sep=1.7,
        flip_y=0.01,
        random_state=42,
    )
    frame = pd.DataFrame(X, columns=[f"feature_{index:02d}" for index in range(X.shape[1])])
    rng = np.random.default_rng(42)
    frame["device_id"] = rng.choice([f"device_{index:02d}" for index in range(12)], len(frame))
    frame["timestamp"] = pd.date_range("2025-01-01", periods=len(frame), freq="min")
    frame["label"] = np.where(y == 1, "attack", "normal")
    duplicates = frame.sample(60, random_state=42)
    frame = pd.concat([frame, duplicates], ignore_index=True)
    frame.to_csv(destination, index=False)
    print(f"Synthetic smoke dataset saved: {destination} ({len(frame)} rows)")


if __name__ == "__main__":
    main()

