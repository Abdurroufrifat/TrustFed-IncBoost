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
    report = test["classification_report"]
    attack_report = report.get("attack", report.get("1"))
    if attack_report is None:
        raise KeyError(
            "Classification report has neither an 'attack' nor a '1' class entry."
        )
    matrix = test["confusion_matrix"]
    malicious_weights = [
        item["aggregation_weight"] for item in metrics["clients"] if item["malicious"]
    ]
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
        "malicious_clients": ",".join(str(value) for value in poison["malicious_clients"]),
        "malicious_weight_sum": float(sum(malicious_weights)),
    }


def _paired_bootstrap(values: np.ndarray, seed: int, repetitions: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "mean_difference": float(values.mean()),
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
        "n_pairs": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    args = parser.parse_args()
    matrix_path = Path(args.matrix).resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    base_path = (matrix_path.parent.parent / matrix["base_config"]).resolve()
    base = load_config(base_path)
    root = project_root_from_config(base)
    result_directory = root / matrix["output_directory"] / utc_run_id("robustness_matrix")
    result_directory.mkdir(parents=True, exist_ok=False)
    run_root = str((result_directory / "runs").relative_to(root))

    rows: list[dict] = []
    for condition in matrix["conditions"]:
        for seed in matrix["seeds"]:
            print(f"Running {condition['name']} seed={seed} ...", flush=True)
            config = _condition_config(base, condition, int(seed), run_root)
            run_directory = run_federated_training(config)
            rows.append(_flatten_metrics(condition["name"], int(seed), run_directory))

    frame = pd.DataFrame(rows)
    frame.to_csv(result_directory / "all_runs.csv", index=False)
    summary_rows: list[dict] = []
    for condition, group in frame.groupby("condition", sort=False):
        row: dict[str, object] = {"condition": condition, "runs": len(group)}
        for metric in METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        row["malicious_weight_sum_mean"] = float(group["malicious_weight_sum"].mean())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(result_directory / "summary_by_condition.csv", index=False)

    paired: dict[str, object] = {}
    for comparison in matrix["paired_comparisons"]:
        left = frame[frame["condition"] == comparison["trust"]].set_index("seed")
        right = frame[frame["condition"] == comparison["baseline"]].set_index("seed")
        common = left.index.intersection(right.index)
        report: dict[str, object] = {}
        for metric_index, metric in enumerate(METRICS):
            differences = left.loc[common, metric].to_numpy(float) - right.loc[common, metric].to_numpy(float)
            report[metric] = _paired_bootstrap(
                differences, seed=9000 + metric_index
            )
        paired[comparison["name"]] = report
    write_json(result_directory / "paired_comparisons.json", paired)
    with (result_directory / "matrix_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(matrix, stream, sort_keys=False)
    print(f"Robustness matrix completed. Results: {result_directory}")


if __name__ == "__main__":
    main()
