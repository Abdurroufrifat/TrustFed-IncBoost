from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def resolve_from_project(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def project_root_from_config(config: dict[str, Any]) -> Path:
    config_path = Path(config["_config_path"])
    return config_path.parent.parent.resolve()

