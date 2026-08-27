# Phase 6: frozen unseen-seed confirmation

## Status of the Phase-5 results

The Phase-5 replay is a development ablation because TrustFed V2 was designed
after inspecting the seed-66 V1 failure. Across the ten development seeds, V2
assigned zero aggregation weight to all 20 poisoned clients, including seed 66.
It also improved PR-AUC and ROC-AUC over uniform aggregation under both
poisoning strategies without harming clean performance. These findings justify
freezing V2, but they are not independent confirmatory evidence.

## Frozen confirmation protocol

The confirmation uses ten seeds that do not overlap development:

`121, 132, 143, 154, 165, 176, 187, 198, 209, 220`

It is a complete paired 3x3 design:

- scenarios: clean, largest-client label flip, and attack-rich-client label flip;
- methods: Uniform, original TrustFed V1, and frozen TrustFed V2;
- ten identical seeds per condition, for 90 runs total.

The primary poisoning endpoints are PR-AUC and ROC-AUC. V2 must have paired
95% confidence intervals above zero versus Uniform under both poisoning
strategies. Clean macro-F1 uses a predeclared non-inferiority margin of one
percentage point. Mean V2 malicious-client weight must not exceed 2% in either
attack. All other metrics are secondary and must be reported without selective
omission.

No trust parameters, seeds, endpoints, or margins may be changed after viewing
confirmation results. The runner writes SHA-256 hashes of the frozen matrix,
base configuration, aggregation implementation, and runner into
`freeze_manifest.json`.

## Install and validate

After extracting the update into `D:\TrustFed-IncBoost`, run:

```powershell
cd D:\TrustFed-IncBoost
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pytest -q
python scripts\08_run_unseen_confirmation.py `
    --matrix configs\ehms_unseen_confirmation.yaml `
    --validate-only
```

The validation message must confirm ten unseen seeds, no development-seed
overlap, and 90 frozen runs.

## Run the confirmation

```powershell
python scripts\08_run_unseen_confirmation.py `
    --matrix configs\ehms_unseen_confirmation.yaml
```

Do not interrupt the terminal and do not modify the configuration after the
run begins.

## Files for final analysis

The result directory contains:

- `all_runs.csv`
- `summary_by_condition.csv`
- `paired_comparisons.json`
- `client_diagnostics.csv`
- `confirmation_decisions.json`
- `freeze_manifest.json`
- `matrix_config.yaml`

The first six files are sufficient for the final confirmatory analysis. The
per-run folders remain the full reproducibility record.
