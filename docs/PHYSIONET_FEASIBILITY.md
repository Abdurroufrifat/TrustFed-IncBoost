# PhysioNet signal-state feasibility study

The scripts in `scripts/13_*.py` through `scripts/27_*.py` reproduce the
recording-disjoint PhysioNet/Computing in Cardiology 2015 feasibility study.
Place the original `training.zip` at `data/raw/physionet_2015/training.zip`.
From the repository root on Windows, install dependencies and run:

```powershell
cd D:\TrustFed-IncBoost
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\27_finish_physionet_project.py --dry-run
.\.venv\Scripts\python.exe scripts\27_finish_physionet_project.py
Get-Content outputs\physionet_project_final\manifest.json
```

The runner checks completed stages and resumes missing stages. `--force`
reruns every experiment; `--verify-only` checks existing outputs. Source data
and generated outputs stay local. `figures/` contains the vector methodology
diagrams and their Python drawing source.

Five simulated clients are used; they are not verified hospital sites. Faults
are injected into recorded ECG/PLETH signals and do not represent observed
cyberattacks. The recording-specific state is a physiological signal proxy,
not a clinically validated patient digital twin. The WUSTL-EHMS IDS study and
PhysioNet signal-state study have different datasets and threat models.
