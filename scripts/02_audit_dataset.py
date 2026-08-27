from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustfed_incboost.config import load_config
from trustfed_incboost.pipeline import audit_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path("outputs/latest_dataset_audit.json").resolve()
    audit = audit_from_config(config, output)
    print(json.dumps(audit, indent=2))
    print(f"Audit saved: {output}")


if __name__ == "__main__":
    main()

