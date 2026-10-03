"""Audit PhysioNet/CinC 2015 training.zip without treating alarm labels as attacks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


def audit(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        rows = list(csv.reader(io.StringIO(archive.read("training/ALARMS").decode("utf-8"))))
        records = archive.read("training/RECORDS").decode("utf-8").splitlines()
        if len(rows) != len(records) or set(r[0] for r in rows) != set(records):
            raise ValueError("ALARMS and RECORDS disagree")
        alarm_types = Counter()
        alarm_verdicts = Counter()
        channels = Counter()
        formats = Counter()
        candidate = 0
        manifest = []
        for record, alarm_type, verdict in rows:
            if verdict not in {"0", "1"}:
                raise ValueError(f"unexpected alarm label in {record}: {verdict}")
            header_name = f"training/{record}.hea"
            signal_name = f"training/{record}.mat"
            if header_name not in names or signal_name not in names:
                raise ValueError(f"missing waveform or header for {record}")
            head = archive.read(header_name).decode("utf-8").splitlines()
            first = head[0].split()
            if first[0] != record or len(first) < 4:
                raise ValueError(f"bad waveform header: {record}")
            count, sample_rate, samples = int(first[1]), int(first[2]), int(first[3])
            signal_lines = [line for line in head[1:] if line and not line.startswith("#")]
            if count != len(signal_lines):
                raise ValueError(f"channel count mismatch: {record}")
            labels = [line.split()[-1] for line in signal_lines]
            if len(labels) != len(set(labels)):
                raise ValueError(f"duplicate channel names: {record}")
            channels.update(labels)
            alarm_types[alarm_type] += 1
            alarm_verdicts[verdict] += 1
            formats[f"{sample_rate}Hz_{samples}samples"] += 1
            candidate += "PLETH" in labels and any(c in labels for c in ("II", "I", "III", "V", "MCL", "aVF", "aVL", "aVR"))
            manifest.append({"record": record, "alarm_type": alarm_type,
                             "alarm_verdict": "true" if verdict == "1" else "false",
                             "sample_rate_hz": sample_rate, "samples": samples, "channels": labels})
    return {
        "source_sha256": digest.hexdigest(),
        "source": "PhysioNet/CinC Challenge 2015 public training recordings",
        "original_labels_mean": "true/false clinical alarms; NOT cyberattacks",
        "original_hospital_id_available": False,
        "patient_group_id_verified": False,
        "warning": "Do not claim hospital federation or patient-disjoint splits from these files. Inspect waveform identity before splitting; injected attacks must be separately labelled.",
        "record_count": len(rows), "alarm_verdict_counts": dict(alarm_verdicts),
        "alarm_type_counts": dict(alarm_types), "channel_counts": dict(channels),
        "record_lengths": dict(formats), "ecg_and_pleth_candidates": candidate,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("data/raw/physionet_2015/training.zip"))
    parser.add_argument("--output", type=Path, default=Path("outputs/physionet_2015_audit.json"))
    args = parser.parse_args()
    report = audit(args.zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Records: {report['record_count']}; ECG + PLETH: {report['ecg_and_pleth_candidates']}")
    print(f"Alarm verdicts (not cyberattacks): {report['alarm_verdict_counts']}")
    print(f"Audit: {args.output}")


if __name__ == "__main__":
    main()
