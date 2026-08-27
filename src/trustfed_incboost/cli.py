from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustfed_incboost.config import load_config, project_root_from_config
from trustfed_incboost.pipeline import audit_from_config, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trustfed")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["audit", "train"]:
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "audit":
        root = project_root_from_config(config)
        output = root / "outputs" / "latest_dataset_audit.json"
        audit = audit_from_config(config, output)
        print(json.dumps(audit, indent=2))
        print(f"Audit saved: {output}")
    elif args.command == "train":
        output = run_training(config)
        print(f"Training completed. Results: {output}")


if __name__ == "__main__":
    main()

