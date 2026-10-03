"""Create an ordered synthetic stream for checking the twin workflow only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/twin_streams/synthetic_devices.csv")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--devices", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 100 or args.devices < 3:
        raise ValueError("smoke run needs at least 100 steps and 3 simulated devices")
    rng = np.random.default_rng(20260923)
    rows = []
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    for t in range(args.steps):
        for device in range(args.devices):
            phase = (t + device*17) % 97
            attacked = 35 <= phase < 47
            physiological = attacked and device % 2 == 0
            network = attacked and not physiological
            rows.append({
                "timestamp": (first + pd.Timedelta(minutes=t)).isoformat(),
                "device_id": f"device_{device:02d}",
                "pulse_bpm": round(72 + device*1.8 + 2*np.sin(t/50)
                                   + rng.normal(0, 1.5) + (19 if physiological else 0), 3),
                "spo2_pct": round(97.5 - device*.15 + rng.normal(0, .45)
                                  - (6 if physiological else 0), 3),
                "skin_temp_c": round(36.5 + device*.06 + rng.normal(0, .10)
                                      + (1.5 if physiological else 0), 3),
                "network_rate": round(26 + rng.normal(0, 3.5)
                                      + (65 if network else 0), 3),
                "label": "attack" if attacked else "normal",
            })
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination, index=False)
    print(f"SYNTHETIC SMOKE ONLY: {destination} ({len(frame)} rows, {args.devices} device streams)")


if __name__ == "__main__":
    main()
