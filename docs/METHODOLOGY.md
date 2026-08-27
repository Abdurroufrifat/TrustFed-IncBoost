# Methodology contract

## Baselines

1. Central XGBoost without class weighting.
2. Central XGBoost with the paper-printed weight.
3. Central XGBoost with balanced inverse-frequency weight.
4. Paper-configuration reconstruction.
5. Strong tuned configuration selected using validation data only.

## Splits

- `random_stratified`: reproduction-only comparison with the paper.
- `fingerprint_disjoint`: exact feature duplicates cannot cross partitions.
- `group`: device/capture/source separation.
- `temporal`: chronological generalization.
- Later: leave-one-attack-family-out zero-day protocol.

## Primary metrics

Macro-F1, balanced accuracy, minority recall, per-class F1, PR-AUC, ROC-AUC,
false-positive rate, and bootstrap 95% confidence intervals. Accuracy is
secondary on imbalanced data.

## Reproducibility assumptions

The reference paper does not specify client count, rounds, local epochs,
boosting-tree count, ICA dimension, aggregation thresholds, random seed, or
regularization values. Every reconstructed value must be marked as an
assumption and subjected to sensitivity analysis.

## Claims policy

- A random 80/20 test cannot support a zero-day claim.
- Keeping data local is not, by itself, proof of privacy.
- High accuracy is not proof of robustness to poisoned federated clients.
- No claimed result is written into the manuscript until produced by a saved,
  versioned configuration and verified output file.

