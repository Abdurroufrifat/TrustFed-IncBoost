# Phase 3: Trust-aware aggregation and poisoning

The trust-aware server computes client weights exclusively from the clean
validation partition. The score combines macro-F1, balanced accuracy, PR-AUC,
Brier calibration error, and deviation from the median client prediction. A
maximum weight cap limits dominance by one client under severe non-IID skew.

The label-flip experiment poisons the local training labels of configured
malicious clients. The global validation and test labels remain untouched.
`poisoning_manifest.json` records the selected client, clean class counts,
poisoned class counts, and number of flipped labels.

Run the clean method first:

```powershell
python scripts\05_train_trustfed.py --config configs\ehms_trustfed_clean.yaml
```

After freezing the clean result, run the attack configuration:

```powershell
python scripts\05_train_trustfed.py --config configs\ehms_trustfed_label_flip.yaml
```

These are simulated federated experiments. They do not claim cryptographic
secure aggregation or differential privacy.
