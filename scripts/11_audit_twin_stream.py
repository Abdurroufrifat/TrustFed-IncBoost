"""Preflight a real, ordered multi-site device CSV before expensive training."""
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--site-column", default="site_id")
    parser.add_argument("--device-column", default="device_id")
    parser.add_argument("--time-column", default="timestamp")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--features", nargs="+", required=True)
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    required = [args.site_column, args.device_column, args.time_column, args.label_column, *args.features]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("required columns contain missing values")
    if frame[args.device_column].astype(str).str.strip().eq("").any():
        raise ValueError("device identifiers cannot be blank")
    times = pd.to_datetime(frame[args.time_column], utc=True, errors="coerce")
    if times.isna().any():
        raise ValueError("invalid device timestamps")
    if pd.DataFrame({"device": frame[args.device_column], "time": times}).duplicated().any():
        raise ValueError("duplicate time at a device")
    if (frame.groupby(args.device_column)[args.site_column].nunique() != 1).any():
        raise ValueError("some devices occur at multiple sites")
    for feature in args.features:
        if not pd.to_numeric(frame[feature], errors="coerce").notna().all():
            raise ValueError(f"{feature!r} contains nonnumeric measurements")
    if set(frame[args.label_column].astype(str)) != {"normal", "attack"}:
        raise ValueError("expected labels 'normal' and 'attack'; change the real config if using other labels")
    ordered = frame.assign(__time__=times).sort_values("__time__", kind="stable")
    train_size = round(.60*len(ordered))
    val_end = round(.80*len(ordered))
    for name, part in (("train", ordered.iloc[:train_size]),
                       ("validation", ordered.iloc[train_size:val_end]),
                       ("test", ordered.iloc[val_end:])):
        print(f"{name}: {len(part)} rows; labels {part[args.label_column].value_counts().to_dict()}; "
              f"{part.__time__.min()} to {part.__time__.max()}")
        if set(part[args.label_column]) != {"normal", "attack"}:
            raise ValueError(f"{name} interval needs both labels")
    training = ordered.iloc[:train_size]
    site_counts = training.groupby(args.site_column)[args.label_column].value_counts().unstack(fill_value=0)
    print(f"sites: {frame[args.site_column].nunique()}; devices: {frame[args.device_column].nunique()}")
    print(site_counts.to_string())
    if (site_counts["normal"] < 20).any() or (site_counts["attack"] < 1).any():
        raise ValueError("each site needs >=20 normal and >=1 attack training rows")
    if times.iloc[ordered.index[train_size-1]] >= times.iloc[ordered.index[train_size]] or (
        times.iloc[ordered.index[val_end-1]] >= times.iloc[ordered.index[val_end]]
    ):
        raise ValueError("time split cuts a repeated timestamp; choose a dataset with safe boundaries")
    print("Audit passed. Verify dataset rights, device IDs, units and clinical meaning separately.")


if __name__ == "__main__":
    main()
