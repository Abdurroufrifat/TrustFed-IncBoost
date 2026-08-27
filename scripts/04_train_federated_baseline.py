from __future__ import annotations

import argparse

from trustfed_incboost.config import load_config
from trustfed_incboost.federated_pipeline import run_federated_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output = run_federated_training(load_config(args.config))
    print(f"Federated training completed. Results: {output}")


if __name__ == "__main__":
    main()
