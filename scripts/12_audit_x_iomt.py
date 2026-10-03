"""Audit X-IoMT capture chronology without fabricating hospital sites."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


FEATURES = ("Memory", "In_Temperature", "CPU", "RSSI")


def iso(microseconds: int | None) -> str | None:
    if microseconds is None:
        return None
    return datetime.fromtimestamp(microseconds / 1_000_000, timezone.utc).isoformat()


def analyze(reader, name: str, device_rows, totals: Counter) -> None:
    category = (
        "normal" if "NormalTraffic" in name else
        "physical" if "PhysicalAnomalies" in name else
        "network" if "NetworkAnomalies" in name else "application"
    )
    for row in csv.DictReader(reader):
        totals[f"{category}_rows"] += 1
        device = (row.get("MAC_extracted") or "").strip().lower()
        fields = [row.get(k, "").strip() for k in FEATURES]
        if not device:
            totals[f"{category}_missing_device"] += 1
        if not all(fields):
            totals[f"{category}_missing_twin_features"] += 1
        if category not in ("normal", "physical") or not device or not all(fields):
            continue
        try:
            timestamp = int(row["frame.time"].strip())
            for field in fields:
                float(field)
            label = float(row["anomaly.label"])
        except (ValueError, TypeError, KeyError):
            totals["candidate_invalid_numeric_or_time"] += 1
            continue
        if label not in (0.0, 1.0):
            totals["candidate_invalid_label"] += 1
            continue
        device_rows[device].append((timestamp, int(label)))


def make_report(files) -> dict:
    totals = Counter()
    device_rows = defaultdict(list)
    for name, reader in files:
        analyze(reader, name, device_rows, totals)
    devices = []
    global_rows = []
    for device, rows in sorted(device_rows.items()):
        rows.sort()
        normal = [t for t, label in rows if label == 0]
        attacks = [t for t, label in rows if label == 1]
        devices.append({
            "device_id": device,
            "normal_rows": len(normal),
            "physical_anomaly_rows": len(attacks),
            "first_normal": iso(normal[0] if normal else None),
            "last_normal": iso(normal[-1] if normal else None),
            "first_anomaly": iso(attacks[0] if attacks else None),
            "last_anomaly": iso(attacks[-1] if attacks else None),
            "normal_recorded_before_first_anomaly": bool(normal and attacks and normal[0] < attacks[0]),
            "normal_recorded_after_last_anomaly": bool(normal and attacks and normal[-1] > attacks[-1]),
        })
        global_rows.extend(rows)
    global_rows.sort()
    split_report = {}
    for name, chunk in (
        ("train", global_rows[:round(.6 * len(global_rows))]),
        ("validation", global_rows[round(.6 * len(global_rows)):round(.8 * len(global_rows))]),
        ("test", global_rows[round(.8 * len(global_rows)):]),
    ):
        split_report[name] = {
            "rows": len(chunk), "normal": sum(label == 0 for _, label in chunk),
            "physical_anomaly": sum(label == 1 for _, label in chunk),
            "first": iso(chunk[0][0] if chunk else None),
            "last": iso(chunk[-1][0] if chunk else None),
        }
    split_ok = all(part["normal"] > 0 and part["physical_anomaly"] > 0
                   for part in split_report.values())
    return {
        "scope": "normal and physical anomaly captures with device MAC and four complete numeric device-state features",
        "original_site_ids": False,
        "network_attacks_eligible_for_device_twin": False,
        "features": list(FEATURES),
        "totals": dict(sorted(totals.items())),
        "devices": devices,
        "global_chronological_60_20_20": split_report,
        "three_intervals_contain_both_labels": split_ok,
        "conclusion": (
            "Candidate for further leakage and episode review only; no original hospital sites or valid network-attack twin."
            if split_ok else
            "STOP: at least one chronological interval lacks a label. Do not run the existing twin study on these files."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/x_iomt/OriginalDatasetUpdated"))
    parser.add_argument("--output", type=Path, default=Path("outputs/x_iomt_capture_audit.json"))
    args = parser.parse_args()
    if args.input.is_dir():
        paths = sorted(args.input.glob("*.csv"))
        if not paths:
            parser.error(f"no CSV files in {args.input}")
        from contextlib import ExitStack
        with ExitStack() as stack:
            files = ((p.name, stack.enter_context(p.open(encoding="utf-8-sig", newline=""))) for p in paths)
            report = make_report(files)
    elif args.input.is_file() and args.input.suffix.lower() == ".zip":
        import io
        with ZipFile(args.input) as archive:
            files = ((n, io.TextIOWrapper(archive.open(n), encoding="utf-8-sig", newline=""))
                     for n in sorted(archive.namelist()) if n.endswith(".csv"))
            report = make_report(files)
    else:
        parser.error(f"input must be the extracted CSV directory or source ZIP: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["conclusion"])
    for name, part in report["global_chronological_60_20_20"].items():
        print(f"{name}: {part['rows']} rows; normal={part['normal']}; physical_anomaly={part['physical_anomaly']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
