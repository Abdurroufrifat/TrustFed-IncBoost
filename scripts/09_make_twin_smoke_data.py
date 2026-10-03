"""Create a labeled synthetic device stream; never report it as clinical data."""
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(302)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for tick in range(400):
        for index in range(5):
            # Four short device episodes, including validation and test periods.
            episode = any(begin <= tick < begin + 16 for begin in (54+index*4,
                          173+index*3, 258+index*2, 341+index*3))
            physiological = (index + tick // 120) % 2 == 0
            pulse = 73 + index*1.5 + 2.5*np.sin(tick/23) + rng.normal(0, 1.7)
            oxygen = 98 - .15*index + rng.normal(0, .22)
            temperature = 36.8 + .07*index + rng.normal(0, .08)
            network = max(0., 90 + 12*np.sin(tick/17) + rng.normal(0, 11))
            if episode and physiological:
                pulse += 16
                oxygen -= 3.0
                temperature += .55
            elif episode:
                network += 160
            rows.append({"timestamp": (start + pd.Timedelta(minutes=tick)).isoformat(),
                         "site_id": f"site_{index:02d}", "device_id": f"device_{index:02d}", "pulse_bpm": pulse,
                         "spo2_pct": oxygen, "skin_temp_c": temperature,
                         "network_rate": network, "label": "attack" if episode else "normal"})
    destination = Path("data/raw/twin_streams/synthetic_device_stream.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"Synthetic smoke data only: {destination} ({len(rows)} records)")


if __name__ == "__main__":
    main()
