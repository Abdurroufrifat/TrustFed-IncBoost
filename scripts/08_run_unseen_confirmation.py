from __future__ import annotations

import argparse
import copy
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_matrix(matrix: dict) -> int:
    seeds = [int(value) for value in matrix["seeds"]]
    development_seeds = {
        int(value) for value in matrix["protocol"]["development_seeds"]
    }
    if len(seeds) != len(set(seeds)):
        raise ValueError("Confirmation seeds must be unique.")
    overlap = sorted(set(seeds) & development_seeds)
    if overlap:
        raise ValueError(
            f"Confirmation seeds overlap the V2 development seeds: {overlap}"
        )

    profiles = matrix["aggregation_profiles"]
    expected_profiles = {"uniform", "trust_v1", "trust_v2"}
    if set(profiles) != expected_profiles:
        raise ValueError(
            "The frozen confirmation requires uniform, trust_v1, and trust_v2 profiles."
        )
    expected_methods = {
        "uniform": "uniform",
        "trust_v1": "trust_aware",
        "trust_v2": "trust_aware_v2",
    }
    observed_methods = {
        name: str(profile["method"]) for name, profile in profiles.items()
    }
    if observed_methods != expected_methods:
        raise ValueError(f"Aggregation profiles are not frozen correctly: {observed_methods}")

    conditions = matrix["conditions"]
    names = [str(item["name"]) for item in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Condition names must be unique.")
    observed_cells = {
        (str(item["scenario"]), str(item["profile"])) for item in conditions
    }
    expected_cells = {
        (scenario, profile)
        for scenario in ("clean", "largest_flip", "attack_rich_flip")
        for profile in expected_profiles
    }
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ValueError(
            f"The confirmation matrix must be a complete 3x3 design; "
            f"missing={missing}, extra={extra}."
        )
    planned_runs = len(seeds) * len(conditions)
    if planned_runs != int(matrix["protocol"]["planned_runs"]):
        raise ValueError(
            f"planned_runs must equal {planned_runs}, not "
            f"{matrix['protocol']['planned_runs']}."
        )
    return planned_runs


def _condition_config(
    base: dict, matrix: dict, condition: dict, seed: int, run_root: str
) -> dict:
    config = copy.deepcopy(base)
    config["project"]["seed"] = int(seed)
    config["project"]["name"] = f"{condition['name']}_seed{seed}"
    config["output"]["directory"] = run_root
    config["evaluation"]["bootstrap_repetitions"] = int(
        condition.get("bootstrap_repetitions", 100)
    )
    profile = matrix["aggregation_profiles"][condition["profile"]]
    # Replace the complete mapping so V1 cannot inherit V2-only parameters and
    # V2 cannot inherit the original V1 consensus coefficient.
    config["federated"]["aggregation"] = copy.deepcopy(profile)
    attack = str(condition["attack"])
    config["poisoning"] = {
        "enabled": attack != "none",
        "attack": "none" if attack == "none" else "label_flip",
        "strategy": condition.get("strategy", "fixed"),
        "malicious_count": int(condition.get("malicious_count", 1)),
        "malicious_clients": condition.get("malicious_clients", []),
        "flip_fraction": float(condition.get("flip_fraction", 0.0)),
    }
    return config


def _flatten_metrics(
    condition: dict, seed: int, run_directory: Path
) -> tuple[dict, list[dict]]:
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
    aggregation_method = {
        "uniform": "uniform",
        "trust_v1": "trust_aware",
        "trust_v2": "trust_aware_v2",
    }[condition["profile"]]
    row = {
        "condition": condition["name"],
        "scenario": condition["scenario"],
        "profile": condition["profile"],
        "aggregation_method": aggregation_method,
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
            if malicious and condition["profile"] == "trust_v2"
            else np.nan
        ),
        "trust_fallback_activated": (
            fallback if condition["profile"] == "trust_v2" else np.nan
        ),
        "eligible_clients": (
            int(
                sum(
                    item.get("trust_v2", {}).get("eligible", False)
                    for item in metrics["clients"]
                )
            )
            if condition["profile"] == "trust_v2"
            else np.nan
        ),
    }
    diagnostics: list[dict] = []
    for item in metrics["clients"]:
        trust = item.get("trust_v2", {})
        diagnostics.append(
            {
                "condition": condition["name"],
                "scenario": condition["scenario"],
                "profile": condition["profile"],
                "seed": seed,
                "client_id": item["client_id"],
                "malicious": item["malicious"],
                "rows": item["rows"],
                "aggregation_weight": item["aggregation_weight"],
                "macro_f1": item["validation_at_0_5"]["macro_f1"],
                "balanced_accuracy": item["validation_at_0_5"][
                    "balanced_accuracy"
                ],
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
    return row, diagnostics


def _paired_bootstrap(
    values: np.ndarray, seed: int, repetitions: int = 10000
) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True)
    means = draws.mean(axis=1)
    return {
        "mean_difference": float(values.mean()),
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
        "n_pairs": int(len(values)),
    }


def _build_decisions(
    frame: pd.DataFrame, paired: dict[str, dict], protocol: dict
) -> dict:
    clean_margin = float(protocol["clean_macro_f1_noninferiority_margin"])
    weight_limit = float(protocol["max_mean_malicious_weight_v2"])

    clean = paired["clean_v2_minus_uniform"]["macro_f1"]
    attack_reports = {
        "largest_flip": paired["largest_v2_minus_uniform"],
        "attack_rich_flip": paired["attack_rich_v2_minus_uniform"],
    }
    clean_check = {
        "criterion": f"paired macro-F1 lower 95% bound > -{clean_margin:.4f}",
        "mean_difference": clean["mean_difference"],
        "lower_95": clean["lower_95"],
        "upper_95": clean["upper_95"],
        "passed": bool(clean["lower_95"] > -clean_margin),
    }
    attack_checks: dict[str, dict] = {}
    for scenario, report in attack_reports.items():
        pr_auc = report["pr_auc"]
        roc_auc = report["roc_auc"]
        attack_checks[scenario] = {
            "criterion": "PR-AUC and ROC-AUC paired lower 95% bounds > 0",
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "passed": bool(pr_auc["lower_95"] > 0 and roc_auc["lower_95"] > 0),
        }

    weight_checks: dict[str, dict] = {}
    for scenario in ("largest_flip", "attack_rich_flip"):
        values = frame[
            (frame["scenario"] == scenario) & (frame["profile"] == "trust_v2")
        ]["malicious_weight_sum"].astype(float)
        mean_weight = float(values.mean())
        weight_checks[scenario] = {
            "criterion": f"mean malicious weight <= {weight_limit:.4f}",
            "mean_malicious_weight": mean_weight,
            "maximum_malicious_weight": float(values.max()),
            "passed": bool(mean_weight <= weight_limit),
        }

    v2 = frame[frame["profile"] == "trust_v2"]
    fallback_rate = float(v2["trust_fallback_activated"].astype(bool).mean())
    primary_passed = bool(
        clean_check["passed"]
        and all(item["passed"] for item in attack_checks.values())
        and all(item["passed"] for item in weight_checks.values())
    )
    return {
        "analysis_status": "confirmatory_unseen_seeds",
        "no_post_result_tuning_permitted": True,
        "clean_noninferiority": clean_check,
        "poisoning_auc_superiority": attack_checks,
        "malicious_weight_control": weight_checks,
        "v2_fallback_activation_rate_all_conditions": fallback_rate,
        "all_primary_acceptance_checks_passed": primary_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen unseen-seed protocol without training.",
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix).resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    planned_runs = _validate_matrix(matrix)
    if args.validate_only:
        print(
            f"Validation passed: Phase-6 uses {len(matrix['seeds'])} unseen paired "
            "seeds, has no "
            f"development-seed overlap, and defines {planned_runs} frozen runs."
        )
        return

    base_path = (matrix_path.parent.parent / matrix["base_config"]).resolve()
    base = load_config(base_path)
    root = project_root_from_config(base)
    result_directory = root / matrix["output_directory"] / utc_run_id(
        "unseen_confirmation"
    )
    result_directory.mkdir(parents=True, exist_ok=False)
    run_root = str((result_directory / "runs").relative_to(root))

    freeze_files = {
        "matrix_config": matrix_path,
        "base_config": base_path,
        "aggregation_implementation": root
        / "src"
        / "trustfed_incboost"
        / "models"
        / "federated.py",
        "confirmation_runner": Path(__file__).resolve(),
    }
    freeze_manifest = {
        "analysis_status": "confirmatory_unseen_seeds",
        "development_seeds": matrix["protocol"]["development_seeds"],
        "confirmation_seeds": matrix["seeds"],
        "planned_runs": planned_runs,
        "no_post_result_tuning_permitted": True,
        "sha256": {name: _sha256(path) for name, path in freeze_files.items()},
    }
    write_json(result_directory / "freeze_manifest.json", freeze_manifest)

    rows: list[dict] = []
    diagnostics: list[dict] = []
    for condition in matrix["conditions"]:
        for seed in matrix["seeds"]:
            print(f"Running {condition['name']} seed={seed} ...", flush=True)
            config = _condition_config(base, matrix, condition, int(seed), run_root)
            run_directory = run_federated_training(config)
            row, client_rows = _flatten_metrics(condition, int(seed), run_directory)
            rows.append(row)
            diagnostics.extend(client_rows)

    frame = pd.DataFrame(rows)
    if len(frame) != planned_runs or frame.duplicated(["condition", "seed"]).any():
        raise RuntimeError("The completed confirmation table is incomplete or duplicated.")
    frame.to_csv(result_directory / "all_runs.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        result_directory / "client_diagnostics.csv", index=False
    )

    summary_rows: list[dict] = []
    for condition, group in frame.groupby("condition", sort=False):
        row: dict[str, object] = {
            "condition": condition,
            "scenario": group["scenario"].iloc[0],
            "profile": group["profile"].iloc[0],
            "runs": len(group),
        }
        for metric in METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        row["malicious_weight_sum_mean"] = float(
            group["malicious_weight_sum"].astype(float).mean()
        )
        if group["malicious_gate_passed"].notna().any():
            row["malicious_gate_pass_rate"] = float(
                group["malicious_gate_passed"].dropna().astype(bool).mean()
            )
        if group["trust_fallback_activated"].notna().any():
            row["fallback_activation_rate"] = float(
                group["trust_fallback_activated"].dropna().astype(bool).mean()
            )
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        result_directory / "summary_by_condition.csv", index=False
    )

    expected_seeds = {int(value) for value in matrix["seeds"]}
    paired: dict[str, dict] = {}
    for comparison_index, comparison in enumerate(matrix["paired_comparisons"]):
        left = frame[frame["condition"] == comparison["left"]].set_index("seed")
        right = frame[frame["condition"] == comparison["right"]].set_index("seed")
        common = left.index.intersection(right.index)
        if set(int(value) for value in common) != expected_seeds:
            raise RuntimeError(
                f"Comparison {comparison['name']} does not contain all paired seeds."
            )
        report: dict[str, dict] = {}
        for metric_index, metric in enumerate(METRICS):
            differences = (
                left.loc[sorted(common), metric].to_numpy(float)
                - right.loc[sorted(common), metric].to_numpy(float)
            )
            report[metric] = _paired_bootstrap(
                differences,
                seed=15000 + 100 * comparison_index + metric_index,
            )
        paired[comparison["name"]] = report
    write_json(result_directory / "paired_comparisons.json", paired)
    write_json(
        result_directory / "confirmation_decisions.json",
        _build_decisions(frame, paired, matrix["protocol"]),
    )
    with (result_directory / "matrix_config.yaml").open(
        "w", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(matrix, stream, sort_keys=False)
    print(f"Unseen confirmation completed. Results: {result_directory}")


if __name__ == "__main__":
    main()
