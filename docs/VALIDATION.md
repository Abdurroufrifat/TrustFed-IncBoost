# Package validation

Validation date: 2026-08-26

## Automated checks

- Python source compilation: passed.
- Unit and pipeline tests: 10 passed.
- Environment check with Python 3.12: passed.
- Synthetic dataset audit: passed.
- Scikit-learn end-to-end baseline: passed.
- XGBoost 3.4 end-to-end baseline with early stopping: passed.
- Non-IID Dirichlet partition invariants and dynamic aggregation tests: passed.
- Synthetic four-client federated ensemble smoke run: passed.
- Trust-aware capped aggregation and label-flip determinism tests: passed.
- Clean and poisoned synthetic TrustFed end-to-end runs: passed.
- Automatic training-only attacker selection tests: passed.
- Paired multi-seed robustness matrix synthetic smoke run: passed (8 runs).

The robustness smoke run produced `all_runs.csv`,
`summary_by_condition.csv`, and `paired_comparisons.json` for clean and
attack-rich label-flip conditions under uniform and trust-aware aggregation.
It verifies execution and reporting only; its synthetic numbers must not be
cited as research results.

## XGBoost synthetic smoke result

The smoke dataset is generated solely to verify execution and must never be
cited as a research result.

- Accuracy: 0.9810
- Balanced accuracy: 0.9592
- Macro-F1: 0.9604
- ROC-AUC: 0.9862
- PR-AUC: 0.9690

The run confirmed that validation-selected thresholds, fingerprint-disjoint
splits, class weights, confidence intervals, figures, predictions, and model
serialization work together.
