# Windows and VS Code setup

## Install once

1. Install 64-bit Python 3.12 from <https://www.python.org/downloads/windows/>.
   Select **Add python.exe to PATH** during installation.
2. Install VS Code from <https://code.visualstudio.com/>.
3. In VS Code, install the Microsoft **Python** and **Pylance** extensions.
4. Install Git from <https://git-scm.com/download/win> if you will use GitHub.

## Create the environment

Open the extracted `TrustFed-IncBoost` folder in VS Code. Select **Terminal >
New Terminal**, make sure it is PowerShell, and run:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Select the interpreter shown as `.venv` using **Ctrl+Shift+P > Python: Select
Interpreter**.

## Verify before downloading research data

```powershell
python scripts\00_check_environment.py
python scripts\01_make_synthetic_data.py
python scripts\02_audit_dataset.py --config configs\synthetic.yaml
python scripts\03_train_baseline.py --config configs\synthetic.yaml
pytest -q
```

Do not continue to real experiments if any command fails. Save the complete
terminal output so the error can be diagnosed without guessing.

## Hardware guidance

- Minimum: 16 GB RAM and 20 GB free disk space.
- Recommended: 32 GB RAM and at least 80 GB free disk space for ToN_IoT work.
- A GPU is optional for the tree baseline. CPU results remain scientifically
  valid and are easier to reproduce.
- Do not download raw PCAP archives for the first experiment. Use the processed
  CSV/telemetry files identified in the dataset guide.

