Extract this ZIP directly into D:\TrustFed-IncBoost. It adds scripts\12_audit_x_iomt.py and one reference audit report. It does not alter existing code or input data.

In VS Code PowerShell:
cd D:\TrustFed-IncBoost
.\.venv\Scripts\python.exe scripts\12_audit_x_iomt.py

Read outputs\x_iomt_capture_audit.json. The script accepts --input <ZIP or extracted CSV folder> and --output <JSON>. It deliberately does not invent site identifiers, join incompatible time periods or train a model. Its candidate scope is device telemetry in normal and physical-anomaly captures; network attacks have no device state. Output is an eligibility audit, not a result for a paper.
