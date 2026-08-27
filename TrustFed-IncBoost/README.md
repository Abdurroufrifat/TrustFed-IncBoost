# TrustFed-IncBoost

[![Tests](https://github.com/Abdurroufrifat/TrustFed-IncBoost/actions/workflows/tests.yml/badge.svg)](https://github.com/Abdurroufrifat/TrustFed-IncBoost/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Chance-aware trust-gated federated XGBoost ensembles for poisoning-resilient Internet of Medical Things (IoMT) intrusion detection.

## Why this project exists

Federated intrusion detection keeps raw client records at their source, but a malicious participant can still corrupt the shared decision rule. The problem is harder when honest clients have strongly non-IID traffic: an unusual specialist site may look like an attacker.

TrustFed-IncBoost evaluates client models on a clean server-held validation set and combines four signals:

- validation quality;
- chance-aware balanced-accuracy and ROC-AUC margins;
- probability calibration; and
- bounded prediction consensus.

A hard gate rejects catastrophically weak clients. Capped-simplex normalization then limits every accepted client to at most 40% aggregation weight.

## Confirmatory result

The frozen study contains 90 runs: three aggregation profiles, three scenarios, and ten previously unseen paired seeds.

| Scenario and endpoint | TrustFed V2 minus Uniform | Paired bootstrap 95% CI |
|---|---:|---:|
| Clean macro-F1 | +1.075 percentage points | +0.542 to +1.677 |
| Largest-client flip PR-AUC | +0.776 points | +0.368 to +1.205 |
| Largest-client flip ROC-AUC | +0.267 points | +0.150 to +0.396 |
| Attack-rich flip PR-AUC | +0.725 points | +0.370 to +1.102 |
| Attack-rich flip ROC-AUC | +0.393 points | +0.200 to +0.665 |

TrustFed V2 assigned exactly 0% aggregation weight to the malicious client in all 20 attacked confirmation runs. Accuracy and attack-F1 differences under poisoning were not statistically supported; this repository does not claim universal Byzantine robustness.

The complete run-level evidence is in [`results/ehms_unseen_confirmation`](results/ehms_unseen_confirmation). The IEEE manuscript and vector figures are in [`paper`](paper).

## Repository structure

```text
configs/                         Experiment configurations
data/raw/README.md               Dataset acquisition instructions
docs/                            Methodology and reproduction notes
paper/                           IEEE LaTeX source, vector figures, and PDF
results/ehms_unseen_confirmation Frozen 90-run analytical evidence
scripts/                         Dataset, training, and matrix runners
src/trustfed_incboost/           Python package
tests/                           Automated tests
```

Raw EHMS data, virtual environments, caches, temporary outputs, and trained model binaries are intentionally excluded.

## Installation

Python 3.12 is recommended.

### Windows PowerShell

```powershell
git clone https://github.com/Abdurroufrifat/TrustFed-IncBoost.git
cd TrustFed-IncBoost
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

Expected test result:

```text
12 passed
```

## Synthetic smoke test

The synthetic dataset is generated locally and requires no external download.

```powershell
python scripts\00_check_environment.py
python scripts\01_make_synthetic_data.py
python scripts\02_audit_dataset.py --config configs\synthetic.yaml
python scripts\03_train_baseline.py --config configs\synthetic.yaml
python scripts\04_train_federated_baseline.py --config configs\synthetic_federated.yaml
```

## WUSTL-EHMS experiment

The WUSTL-EHMS dataset is not redistributed. Follow [`docs/DATASETS.md`](docs/DATASETS.md), place the CSV at the configured local path, and audit it before training.

Validate the frozen confirmation design without training:

```powershell
python scripts\08_run_unseen_confirmation.py `
  --matrix configs\ehms_unseen_confirmation.yaml `
  --validate-only
```

Run all 90 frozen confirmation experiments:

```powershell
python scripts\08_run_unseen_confirmation.py `
  --matrix configs\ehms_unseen_confirmation.yaml
```

The protocol separates development seeds from confirmation seeds and prohibits post-result tuning.

## Reproducibility rules

- Duplicate row fingerprints remain within one data split.
- The test set is not used for client weighting or threshold selection.
- Poisoning changes only the configured client's local training labels.
- Validation and test labels remain clean.
- Uniform, TrustFed V1, and TrustFed V2 use paired seeds.
- Confirmation configurations and implementation hashes are recorded in `freeze_manifest.json`.

## Paper

The compiled draft is [`paper/TrustFed-IncBoost_IEEE_Draft.pdf`](paper/TrustFed-IncBoost_IEEE_Draft.pdf). To rebuild it:

```bash
cd paper
python make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Citation

If this repository supports your research, cite the software using [`CITATION.cff`](CITATION.cff). Replace the placeholder manuscript status with the final DOI after publication.

## License and data

The source code is released under the MIT License. Dataset files are governed by their original providers' terms and are not covered by this repository's license.
