# Phase 2: Non-IID federated baseline

This stage simulates five data-owning clients with label distributions sampled
from a Dirichlet distribution. Lower alpha values create stronger statistical
heterogeneity. Every training record belongs to exactly one client, while the
validation and test partitions remain global and fingerprint-disjoint.

Each client trains its own boosting classifier. The server combines probability
outputs rather than averaging incompatible tree parameters. Aggregation weights
are computed from client sample count, public-validation macro-F1, and measured
training time. The test partition is used once for final evaluation.

This is the clean non-IID baseline. It does not yet claim secure aggregation,
differential privacy, poisoning resistance, or communication confidentiality.
Those mechanisms belong to the subsequent adversarial and TrustFed stages.

Run on Windows PowerShell:

```powershell
python scripts\04_train_federated_baseline.py --config configs\ehms_federated.yaml
```

Important outputs are `client_partitions.json`, `metrics.json`,
`split_manifest.json`, `confusion_matrix.png`, and
`federated_ensemble.joblib`.
