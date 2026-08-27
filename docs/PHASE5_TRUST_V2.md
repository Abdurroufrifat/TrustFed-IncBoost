# Phase 5: chance-aware TrustFed V2

## Why V2 is required

The ten-seed Phase-4 matrix exposed a failure in the largest-client attack at
seed 66. Dirichlet partitioning produced two honest clients dominated by class
1 and two honest clients dominated by class 0. The poisoned largest client was
also dominated by class 1 after flipping. Therefore, three models produced
similar validation probabilities and the median-consensus term treated the
poisoned model as normal.

The V1 cap then amplified the problem. Two clients reached the 0.40 cap, so
most of the remaining 0.20 mass was assigned to the poisoned client. Its final
weight was 0.19596 even though its validation macro-F1 was 0.11087 and its
balanced accuracy was below random at 0.49877.

## V2 defense

V2 keeps the public validation set clean and uses it before consensus:

1. A client passes the gate only when balanced accuracy and ROC-AUC exceed
   random performance by `chance_margin`, and macro-F1 exceeds a catastrophic
   performance floor.
2. The continuous security evidence is the geometric mean of the positive
   balanced-accuracy and ROC-AUC margins above 0.5.
3. Consensus deviation is clipped and used only as a bounded secondary term.
4. Rejected clients receive zero weight. If too few clients pass, a documented
   supervised fallback selects the best `minimum_trusted_clients`.
5. The cap is applied only across eligible clients, preventing rejected
   clients from receiving leftover probability mass.

For the recorded seed-66 diagnostics, the regression test gives the malicious
client exactly zero aggregation weight while retaining all four honest clients.
This is a regression result, not a publication claim. The complete paired
ten-seed V2 matrix must be run next.

Because V2 was designed after inspecting seed 66, these 30 replay runs are a
development ablation. They must not be presented as independent confirmatory
evidence. If the ablation succeeds, the next phase will freeze V2 and evaluate
uniform, V1, and V2 on unseen seeds before journal claims are finalized.

## Windows installation

Extract the Phase-5 update ZIP directly into the project root
`D:\TrustFed-IncBoost`. Allow Windows to replace files when prompted. The ZIP
contains only new or intentionally updated project files; it does not contain
or overwrite the dataset or earlier output directories.

Activate the existing environment and install the updated editable package:

```powershell
cd D:\TrustFed-IncBoost
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pytest -q
```

Expected result: all tests pass, including
`test_trust_v2_rejects_seed66_consensus_camouflage`.

## Locate the completed Phase-4 directory

```powershell
$phase4 = Get-ChildItem .\outputs\ehms_robustness_matrix_10seeds -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$phase4.FullName
```

Confirm that `$phase4` contains `all_runs.csv` with 60 rows before continuing.

Run the built-in validation without starting training:

```powershell
python scripts\07_run_trust_v2_matrix.py `
    --matrix configs\ehms_trustfed_v2_matrix.yaml `
    --phase4-results $phase4.FullName `
    --validate-only
```

Expected output begins with `Validation passed`.

## Run only the 30 new V2 experiments

```powershell
python scripts\07_run_trust_v2_matrix.py `
    --matrix configs\ehms_trustfed_v2_matrix.yaml `
    --phase4-results $phase4.FullName
```

The script reuses the completed Phase-4 table and trains only three V2
conditions over ten paired seeds. It does not retrain the 60 V1 and uniform
runs.

## Files to send for analysis

After completion, locate the newest V2 result directory:

```powershell
$v2 = Get-ChildItem .\outputs\ehms_trustfed_v2_matrix -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

explorer $v2.FullName
```

Send these four files:

1. `all_runs_combined.csv`
2. `summary_by_condition.csv`
3. `paired_comparisons.json`
4. `v2_client_diagnostics.csv`

The primary acceptance checks are:

- the largest-client seed-66 malicious weight is strongly suppressed;
- mean malicious weight is reduced under both attack strategies;
- V2 versus uniform intervals for PR-AUC, ROC-AUC, and attack F1 are reported;
- V2 does not materially damage clean performance;
- fallback activation and gate errors are disclosed rather than hidden.
