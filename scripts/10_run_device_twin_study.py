"""Compare frozen CAG-FE with a local device twin on chronological data."""
from __future__ import annotations

import argparse
from pathlib import Path

from trustfed_incboost.twin_study import run_twin_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Folder from scripts/05_train_trustfed.py containing the saved ensemble")
    parser.add_argument("--device-column", required=True)
    parser.add_argument("--time-column", required=True)
    parser.add_argument("--features", nargs="+", required=True,
                        help="Numeric device-state features observed in time order")
    parser.add_argument("--twin-weight", type=float, default=.25)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--max-normal-deviation", type=float, default=4.)
    parser.add_argument("--min-benign", type=int, default=20)
    parser.add_argument("--synthetic", action="store_true",
                        help="Mark all outputs as synthetic checks, not research findings")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = run_twin_study(args.run_dir, root,
                            device_column=args.device_column,
                            time_column=args.time_column,
                            feature_columns=args.features,
                            twin_weight=args.twin_weight, alpha=args.alpha,
                            max_normal_deviation=args.max_normal_deviation,
                            min_benign=args.min_benign, synthetic=args.synthetic)
    print(f"New temporal twin comparison saved: {output}")


if __name__ == "__main__":
    main()
