# Frozen results

`ehms_unseen_confirmation/` contains the complete analytical evidence exported by the frozen 90-run confirmation matrix:

- `all_runs.csv`: one row per condition and unseen seed;
- `summary_by_condition.csv`: means and standard deviations;
- `paired_comparisons.json`: paired bootstrap effects and confidence intervals;
- `client_diagnostics.csv`: client-level validation, gate, and weight diagnostics;
- `confirmation_decisions.json`: pre-specified acceptance checks;
- `freeze_manifest.json`: development/confirmation seeds and implementation hashes.
- `evidence_hashes_sha256.csv`: checksums for the public evidence files.

The timestamped training folders and serialized models are not committed. They are reproducible from the frozen configuration and are unsuitable for ordinary Git storage.
