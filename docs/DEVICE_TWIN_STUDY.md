# CAG-FE with a device-state twin: runnable research add-on

This add-on targets the special issue's adversarial-resilience topic by
linking CAG-FE's site-trained predictors to a causal state surrogate for each
medical device. The twin estimates a benign device reference from training
history, compares new sensor readings with its prior state, and combines the
deviation with the frozen CAG-FE probability. An alerting reading does not
update the normal state. It also compares the original detector, the twin
alone, and their combination on the same future records.

The uploaded paper's 90 WUSTL-EHMS runs are **unchanged**. Their client groups
are simulated with label-Dirichlet partitions, and their split is not
temporal. They are not digital-twin results. This code starts a new experiment
and refuses to replay that old non-temporal artifact as a twin study.

## Install on Windows / VS Code

The ZIP is **flat**: extract its contents directly into `D:\TrustFed-IncBoost`
at the same level as the existing `scripts`, `src`, `configs` and `tests`
directories. Do not extract into `D:\TrustFed-IncBoost\TrustFed...`.

The patch modifies the existing training pipeline to keep each `site_id`
entirely on one client. It applies to the inspected public repository commit
`536f224` (2026-08-27). If `git apply --check` fails, stop; your local pipeline
differs and needs a review before applying this patch.

```powershell
cd D:\TrustFed-IncBoost
git apply --check patches\enable_site_group_partition.patch
git apply patches\enable_site_group_partition.patch
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

Keep the project at this root. The commands below use the same folder.

## Run a complete software check (synthetic data)

```powershell
cd D:\TrustFed-IncBoost
.\.venv\Scripts\python.exe scripts\09_make_twin_smoke_data.py
.\.venv\Scripts\python.exe scripts\11_audit_twin_stream.py --csv data\raw\twin_streams\synthetic_device_stream.csv --features pulse_bpm spo2_pct skin_temp_c
.\.venv\Scripts\python.exe scripts\05_train_trustfed.py --config configs\synthetic_device_twin.yaml
$clean = Get-ChildItem .\outputs\synthetic_device_twin -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
.\.venv\Scripts\python.exe scripts\10_run_twin_study.py --run-dir $clean.FullName --features pulse_bpm spo2_pct skin_temp_c
Get-Content (Join-Path $clean.FullName 'twin_study\summary.json')
```

For a paired, single-client complete label-flip software check:

```powershell
cd D:\TrustFed-IncBoost
.\.venv\Scripts\python.exe scripts\05_train_trustfed.py --config configs\synthetic_device_twin_flip.yaml
$flip = Get-ChildItem .\outputs\synthetic_device_twin_flip -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
.\.venv\Scripts\python.exe scripts\10_run_twin_study.py --run-dir $flip.FullName --features pulse_bpm spo2_pct skin_temp_c
Get-Content (Join-Path $flip.FullName 'twin_study\summary.json')
```

Results are in the existing run folder under `twin_study/`:
`summary.json` (accuracy, macro-F1, attack F1, PR-AUC, ROC-AUC, replay time),
`test_replay.csv` (one scored record per row), and `by_device.csv` (false
alarms per device-hour, missed episodes, delay to first alert). The synthetic
generator is a software check only. It is not research evidence or a medical
dataset.

## Use a real, chronologically recorded device stream

Obtain licensed records with globally unique `device_id`, a real `site_id`,
parseable `timestamp`, binary `label` (`normal` / `attack`), and numeric,
device-level measurements. `pulse_bpm`, `spo2_pct`, and `skin_temp_c` are
example twin features; update `--features` for the actual sensor columns.
Each device must stay at one site. Each site must have both labels in the
training interval. Do not infer times or patients from row numbers.

Save the real CSV as `data\raw\twin_streams\medical_device_stream.csv`.
Copy `configs\real_device_twin_template.yaml` to
`configs\real_device_twin.yaml` and set the real column names, number of
sites, and XGBoost settings. Then:

```powershell
cd D:\TrustFed-IncBoost
Copy-Item configs\real_device_twin_template.yaml configs\real_device_twin.yaml
.\.venv\Scripts\python.exe scripts\11_audit_twin_stream.py --csv data\raw\twin_streams\medical_device_stream.csv --features pulse_bpm spo2_pct skin_temp_c
.\.venv\Scripts\python.exe scripts\05_train_trustfed.py --config configs\real_device_twin.yaml
$real = Get-ChildItem .\outputs\real_device_twin -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
.\.venv\Scripts\python.exe scripts\10_run_twin_study.py --run-dir $real.FullName --features pulse_bpm spo2_pct skin_temp_c
Get-Content (Join-Path $real.FullName 'twin_study\summary.json')
```

For a defensible paper, freeze the protocol before viewing test results:
sensor units and reference windows, sites, train/validation/test dates, twin
weight, attacks, endpoints and uncertainty estimates. Repeat on independent
devices or a second real dataset. Compare under clean and label-flip conditions
and report negative or neutral effects. The existing training pipeline uses
the same validation split for early stopping, trust scoring, and threshold
selection; separate these roles in a stronger follow-up. This runner reads
site streams together for an **offline replay**; it does not implement
network transport, secure aggregation, real-time operation, or formal privacy.
