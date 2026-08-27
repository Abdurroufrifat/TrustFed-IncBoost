from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_classification

from trustfed_incboost.config import load_config
from trustfed_incboost.pipeline import run_training


def test_training_pipeline_with_sklearn_backend(tmp_path: Path) -> None:
    X, y = make_classification(
        n_samples=700,
        n_features=12,
        n_informative=8,
        weights=[0.8, 0.2],
        class_sep=1.5,
        random_state=11,
    )
    frame = pd.DataFrame(X, columns=[f"f{index}" for index in range(X.shape[1])])
    frame["label"] = np.where(y, "attack", "normal")
    data_path = tmp_path / "smoke.csv"
    frame.to_csv(data_path, index=False)
    config = {
        "project": {"name": "pytest_smoke", "seed": 42},
        "dataset": {
            "path": str(data_path),
            "label_column": "label",
            "positive_label": "attack",
            "drop_columns": [],
            "max_rows": None,
        },
        "split": {
            "method": "fingerprint_disjoint",
            "test_size": 0.2,
            "validation_size": 0.2,
            "group_column": None,
            "time_column": None,
        },
        "preprocessing": {"use_ica": False, "ica_components": None},
        "model": {
            "backend": "sklearn",
            "class_weight": "balanced",
            "threshold_metric": "macro_f1",
            "params": {"max_iter": 80, "learning_rate": 0.1},
        },
        "evaluation": {"bootstrap_repetitions": 20},
        "output": {"directory": str(tmp_path / "outputs")},
    }
    project_root = tmp_path / "project"
    config_directory = project_root / "configs"
    config_directory.mkdir(parents=True)
    config_path = config_directory / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = run_training(load_config(config_path))
    assert (output / "metrics.json").exists()
    assert (output / "model.joblib").exists()
    assert (output / "confusion_matrix.png").exists()
