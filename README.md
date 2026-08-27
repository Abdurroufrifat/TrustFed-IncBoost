# TrustFed-IncBoost

Phase 6 adds a frozen 90-run unseen-seed confirmation comparing Uniform,
TrustFed V1, and TrustFed V2 under clean, largest-client label-flip, and
attack-rich-client label-flip conditions. See
`docs/PHASE6_UNSEEN_CONFIRMATION.md` before running the confirmation.

Poisoning-resilient, few-shot incremental federated boosting for zero-day
Consumer IoT intrusion detection.

This repository includes the reproducible centralized foundation and the
Phase-2 non-IID federated ensemble baseline: dataset auditing, leakage-safe
splitting, published and intended class-weight formulas, XGBoost baselines,
Dirichlet client partitioning, validation/performance/time aggregation,
threshold selection, confidence intervals, and automated tests.

Phase 3 adds multi-signal trust-aware aggregation and deterministic binary
label-flip poisoning experiments. Client trust combines validation macro-F1,
balanced accuracy, PR-AUC, calibration error, and robust prediction agreement;
an explicit weight cap prevents one client from dominating the ensemble.

Phase 4 adds leakage-safe automatic attacker selection and a paired five-seed
robustness matrix. It compares uniform and trust-aware aggregation under clean,
largest-client label-flip, and attack-rich-client label-flip conditions and
exports paper-ready run-level, summary, and paired-bootstrap statistics.

## Research rule

Never optimize on the test set. The test set is opened only after preprocessing,
model settings, class weights, and the decision threshold have been selected on
training/validation data.

## Windows quick start

Open this folder in VS Code, open a PowerShell terminal, and run:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
python scripts\00_check_environment.py
python scripts\01_make_synthetic_data.py
python scripts\02_audit_dataset.py --config configs\synthetic.yaml
python scripts\03_train_baseline.py --config configs\synthetic.yaml
python scripts\04_train_federated_baseline.py --config configs\synthetic_federated.yaml
python scripts\05_train_trustfed.py --config configs\ehms_trustfed_clean.yaml
python scripts\06_run_robustness_matrix.py --matrix configs\ehms_robustness_matrix.yaml
pytest -q
```

Expected final smoke-test output includes `status: PASS` and a new timestamped
folder under `outputs/synthetic/`.

## Project stages

1. Environment and synthetic smoke test.
2. WUSTL-EHMS-2020 acquisition and audit.
3. Original-paper baseline reconstruction.
4. ToN_IoT acquisition and natural client partitioning.
5. Non-IID federated baseline (implemented in Phase 2).
6. Poisoning/backdoor attack engine.
7. Trust-aware robust aggregation.
8. Open-set rejection and few-shot incremental adaptation.
9. Ablation, statistics, efficiency, figures, and paper.

Read `docs/WINDOWS_SETUP.md`, then `docs/DATASETS.md`.

## Phase-2 EHMS federated run

After the corrected EHMS centralized baseline is frozen, run:

```powershell
python scripts\04_train_federated_baseline.py --config configs\ehms_federated.yaml
```

The run uses five Dirichlet label-skewed clients (`alpha=0.30`). Client models
are combined using weights determined only from client sample size, public
validation macro-F1, and training time. The held-out test partition is not used
to choose clients, aggregation weights, or the decision threshold.

## Important reproduction note

Equation (8) in the reference paper prints
`2 * class_count / total_count`, although the prose says minority classes should
receive larger weights. This project implements both interpretations:

- `paper_printed`: exactly as printed.
- `balanced`: `total_count / (number_of_classes * class_count)`.

Both will be reported. The published formula will never be silently changed.

## Outputs

Each run saves its resolved configuration, audit, split manifest, metrics,
classification report, confusion matrix, bootstrap confidence intervals, and
trained artifact. Dataset files and trained outputs are excluded from Git by
default.
