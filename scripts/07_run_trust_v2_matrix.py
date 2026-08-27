from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trustfed_incboost.config import load_config, project_root_from_config
from trustfed_incboost.federated_pipeline import run_federated_training
from trustfed_incboost.reproducibility import utc_run_id, write_json


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "pr_auc",
    "roc_auc",
    "attack_precision",
    "attack_recall",
    "attack_f1",
    "false_positives",
    "false_negatives",
)


def _condition_config(base: dict, condition: dict, seed: int, run_root: str) -> dict:
    config = copy.deepcopy(base)
    config["project"]["seed"] = int(seed)
    config["project"]["name"] = f"{condition['name']}_seed{seed}"
    config["output"]["directory"] = run_root
    config["evaluation"]["bootstrap_repetitions"] = int(
        condition.get("bootstrap_repetitions", 100)
    )
    config["federated"]["aggregation"]["method"] = condition["aggregation"]
    attack = condition["attack"]
    config["poisoning"] = {
        "enabled": attack != "none",
        "attack": "none" if attack == "none" else "label_flip",
        "strategy": condition.get("strategy", "fixed"),
        "malicious_count": int(condition.get("malicious_count", 1)),
        "malicious_clients": condition.get("malicious_clients", []),
        "flip_fraction": float(condition.get("flip_fraction", 0.0)),
    }
    return config


def _flatten_metrics(condition: str, seed: int, run_directory: Path) -> dict:
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    poison = json.loads(
        (run_directory / "poisoning_manifest.json").read_text(encoding="utf-8")
    )
    test = metrics["test"]
    attack_report = test["classification_report"].get(
        "attack", test["classification_report"].get("1")
    )
    if attack_report is None:
        raise KeyError("Classification report has neither an 'attack' nor a '1' class.")
    matrix = test["confusion_matrix"]
    malicious = [item for item in metrics["clients"] if item["malicious"]]
    fallback = any(
        item.get("trust_v2", {}).get("fallback_activated", False)
        for item in metrics["clients"]
    )
    return {
        "condition": condition,
        "seed": seed,
        "run_directory": str(run_directory),
        "accuracy": test["accuracy"],
        "balanced_accuracy": test["balanced_accuracy"],
        "macro_f1": test["macro_f1"],
        "pr_auc": test.get("pr_auc", np.nan),
        "roc_auc": test.get("roc_auc", np.nan),
        "attack_precision": attack_report["precision"],
        "attack_recall": attack_report["recall"],
        "attack_f1": attack_report["f1-score"],
        "false_positives": matrix[0][1],
        "false_negatives": matrix[1][0],
        "malicious_clients": ",".join(
            str(value) for value in poison["malicious_clients"]
        ),
        "malicious_weight_sum": float(
            sum(item["aggregation_weight"] for item in malicious)
        ),
        "malicious_gate_passed": (
            bool(any(item["trust_v2"]["gate_passed"] for item in malicious))
            if malicious
            else np.nan
        ),
        "trust_fallback_activated": fallback,
        "eligible_clients": int(
            sum(
                item.get("trust_v2", {}).get("eligible", False)
                for item in metrics["clients"]
            )
        ),
    }


def _client_diagnostics(condition: str, seed: int, run_directory: Path) -> list[dict]:
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    rows = []
    for item in metrics["clients"]:
        trust = item.get("trust_v2", {})
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "client_id": item["client_id"],
                "malicious": item["malicious"],
                "rows": item["rows"],
                "aggregation_weight": item["aggregation_weight"],
                "macro_f1": item["validation_at_0_5"]["macro_f1"],
                "balanced_accuracy": item["validation_at_0_5"]["balanced_accuracy"],
                "pr_auc": item["validation_at_0_5"].get("pr_auc", np.nan),
                "roc_auc": item["validation_at_0_5"].get("roc_auc", np.nan),
                "validation_brier": item["validation_brier"],
                "consensus_deviation": item["consensus_deviation"],
                "quality": trust.get("quality", np.nan),
                "security_evidence": trust.get("security_evidence", np.nan),
                "consensus_z": trust.get("consensus_z", np.nan),
                "gate_passed": trust.get("gate_passed", np.nan),
                "eligible": trust.get("eligible", np.nan),
                "raw_score": trust.get("raw_score", np.nan),
                "fallback_activated": trust.get("fallback_activated", np.nan),
                "run_directory": str(run_directory),
            }
        )
    return rows


def _paired_bootstrap(values: np.ndarray, seed: int, repetitions: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True)
    means = draws.mean(axis=1)
    return {
        "mean_difference": float(values.mean()),
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
        "n_pairs": int(len(values)),
    }


def _validate_phase4(frame: pd.DataFrame, matrix: dict) -> None:
    required_columns = {"condition", "seed", *METRICS, "malicious_weight_sum"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Phase-4 all_runs.csv is missing columns: {missing_columns}")
    seeds = {int(value) for value in matrix["seeds"]}
    required_conditions = set(matrix["required_phase4_conditions"])
    observed_conditions = set(frame["condition"].astype(str))
    missing_conditions = sorted(required_conditions - observed_conditions)
    if missing_conditions:
        raise ValueError(f"Phase-4 results are missing conditions: {missing_conditions}")
    subset = frame[
        frame["condition"].isin(required_conditions)
        & frame["seed"].astype(int).isin(seeds)
    ].copy()
    counts = subset.groupby("condition")["seed"].nunique().to_dict()
    incomplete = {
        condition: counts.get(condition, 0)
        for condition in sorted(required_conditions)
        if counts.get(condition, 0) != len(seeds)
    }
    if incomplete:
        raise ValueError(f"Phase-4 results are not complete for the ten seeds: {incomplete}")
    duplicates = subset.duplicated(["condition", "seed"]).sum()
    if duplicates:
        raise ValueError(f"Phase-4 results contain {duplicates} duplicate condition-seed rows.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument(
        "--phase4-results",
        required=True,
        help="Path to the completed Phase-4 result directory or its all_runs.csv.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the Phase-4 table and configuration without training.",
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix).resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    base_path = (matrix_path.parent.parent / matrix["base_config"]).resolve()
    base = load_config(base_path)
    root = project_root_from_config(base)
    phase4_path = Path(args.phase4_results).resolve()
    phase4_csv = phase4_path if phase4_path.is_file() else phase4_path / "all_runs.csv"
    phase4 = pd.read_csv(phase4_csv)
    _validate_phase4(phase4, matrix)
    if args.validate_only:
        print(
            "Validation passed: Phase-4 contains six complete conditions "
            "across ten paired seeds (60 unique runs)."
        )
        return

    result_directory = root / matrix["output_directory"] / utc_run_id(
        "trust_v2_matrix"
    )
    result_directory.mkdir(parents=True, exist_ok=False)
    run_root = str((result_directory / "runs").relative_to(root))

    new_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for condition in matrix["conditions"]:
        for seed in matrix["seeds"]:
            print(f"Running {condition['name']} seed={seed} ...", flush=True)
            config = _condition_config(base, condition, int(seed), run_root)
            run_directory = run_federated_training(config)
            new_rows.append(
                _flatten_metrics(condition["name"], int(seed), run_directory)
            )
            diagnostic_rows.extend(
                _client_diagnostics(condition["name"], int(seed), run_directory)
            )

    new_frame = pd.DataFrame(new_rows)
    new_frame.to_csv(result_directory / "v2_runs.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        result_directory / "v2_client_diagnostics.csv", index=False
    )
    required_phase4 = phase4[
        phase4["condition"].isin(matrix["required_phase4_conditions"])
        & phase4["seed"].astype(int).isin({int(value) for value in matrix["seeds"]})
    ].copy()
    combined = pd.concat([required_phase4, new_frame], ignore_index=True, sort=False)
    combined.to_csv(result_directory / "all_runs_combined.csv", index=False)

    summary_rows: list[dict] = []
    for condition, group in combined.groupby("condition", sort=False):
        row: dict[str, object] = {"condition": condition, "runs": len(group)}
        for metric in METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        row["malicious_weight_sum_mean"] = float(
            group["malicious_weight_sum"].astype(float).mean()
        )
        if "malicious_gate_passed" in group and group["malicious_gate_passed"].notna().any():
            row["malicious_gate_pass_rate"] = float(
                group["malicious_gate_passed"].dropna().astype(bool).mean()
            )
        if "trust_fallback_activated" in group and group["trust_fallback_activated"].notna().any():
            row["fallback_activation_rate"] = float(
                group["trust_fallback_activated"].dropna().astype(bool).mean()
            )
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        result_directory / "summary_by_condition.csv", index=False
    )

    paired: dict[str, object] = {}
    for comparison_index, comparison in enumerate(matrix["paired_comparisons"]):
        left = combined[combined["condition"] == comparison["trust"]].set_index("seed")
        right = combined[combined["condition"] == comparison["baseline"]].set_index("seed")
        common = left.index.intersection(right.index)
        report: dict[str, object] = {}
        for metric_index, metric in enumerate(METRICS):
            differences = (
                left.loc[common, metric].to_numpy(float)
                - right.loc[common, metric].to_numpy(float)
            )
            report[metric] = _paired_bootstrap(
                differences,
                seed=12000 + 100 * comparison_index + metric_index,
            )
        paired[comparison["name"]] = report
    write_json(result_directory / "paired_comparisons.json", paired)
    with (result_directory / "matrix_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(matrix, stream, sort_keys=False)
    print(f"TrustFed V2 matrix completed. Results: {result_directory}")


if __name__ == "__main__":
    main()
