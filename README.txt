CAUSAL SIGNAL-STATE FEASIBILITY AUDIT

Extract this ZIP directly into D:\TrustFed-IncBoost (merge scripts folder).
Requires data\raw\physionet_2015\training.zip and the stage-6 prepared data
in outputs\physionet_2015_prepared_v3.

VS Code PowerShell at D:\TrustFed-IncBoost:
.\.venv\Scripts\python.exe .\scripts\22_audit_causal_signal_state.py
Get-Content .\outputs\physionet_causal_signal_audit\summary.json

This compares a frozen 30-second physiological signal reference with a
per-record causal updating state on held-out CLEAN windows. Training clean
windows set a feature scale and update gate. Validation recordings select an
update rate; the test recordings are held out. It saves every record's
prediction error to per_record.csv, including extreme outliers.

It does not test attack detection or clinical outcomes and does not validate
a patient digital twin. The evaluation knows the audited stream contains
clean windows. New work is needed before allowing updates during live mixed
clean/attacked streams. The data have no verified patient/hospital identities.
