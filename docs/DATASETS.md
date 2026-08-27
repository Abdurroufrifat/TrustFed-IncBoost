# Dataset acquisition and integrity rules

Datasets are downloaded manually from their official owners so that access
conditions and citations remain visible. Never commit dataset files to GitHub.

## Stage 1: WUSTL-EHMS-2020

Official project page:
<https://www.cse.wustl.edu/~jain/ehms/index.html>

Official-author Kaggle mirror, if the university page is unavailable:
<https://www.kaggle.com/datasets/aghubaish/wustl-ehms-2020-dataset/data>

1. Download the CSV archive.
2. Extract the CSV into `data/raw/ehms/`.
3. Do not rename columns or edit the original file.
4. Update `dataset.path` in `configs/ehms.yaml` if its filename differs.
5. Run the audit before training.

```powershell
python scripts\02_audit_dataset.py --config configs\ehms.yaml
```

Expected scale from the reference paper: 16,318 rows, 14,272 normal and 2,046
abnormal. A mismatch is not automatically wrong, but it must be explained.

## Stage 2: ToN_IoT

Official UNSW page:
<https://research.unsw.edu.au/projects/toniot-datasets>

Use the official page's **HERE** download link. Initially download only the
processed telemetry/train-test CSV material, not every raw PCAP, Windows, and
Linux archive.

Keep device CSVs separate under `data/raw/ton_iot/`. Separation by device is
valuable because each device can become a natural federated client. We will
create a reviewed manifest after the EHMS baseline is verified.

## Future external validation

CICIoT2023 official page:
<https://www.unb.ca/cic/datasets/iotdataset-2023.html>

This dataset is not required for the first baseline. It will be added only
after the feature and split protocol are frozen.

## Non-negotiable data rules

- Preserve raw files unchanged.
- Record filename, byte size, and SHA-256 hash.
- Fit imputation, normalization, ICA, and feature selection only on training
  data.
- Never use random row splitting as the only journal result.
- Never balance or oversample before splitting.
- Never inspect final-test performance during hyperparameter selection.

