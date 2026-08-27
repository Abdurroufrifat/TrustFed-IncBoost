# Phase 4: Paired multi-seed robustness matrix

The matrix runs clean, largest-client label-flip, and attack-rich-client
label-flip conditions for uniform and trust-aware aggregation across five
paired seeds. Malicious clients are selected from training partitions only;
validation and test information is never used to select the attacker.

Run:

```powershell
python scripts\06_run_robustness_matrix.py --matrix configs\ehms_robustness_matrix.yaml
```

The final directory contains `all_runs.csv`, `summary_by_condition.csv`, and
`paired_comparisons.json`. Paired bootstrap intervals quantify TrustFed minus
uniform differences for each metric. Five seeds are a minimum robustness
screen, not a substitute for broader external-dataset validation.
