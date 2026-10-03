"""Replay a new, temporally trained CAG-FE artifact with local device twins."""
import argparse
from pathlib import Path

from trustfed_incboost.twin_study import run_twin_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Folder created by scripts/05_train_trustfed.py")
    parser.add_argument("--device-column", default="device_id")
    parser.add_argument("--time-column", default="timestamp")
    parser.add_argument("--features", nargs="+", required=True,
                        help="Numeric, device-local measurements; use the same units as training")
    parser.add_argument("--twin-weight", type=float, default=.25)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = run_twin_study(args.run_dir, root=root, device_column=args.device_column,
                            time_column=args.time_column, features=args.features,
                            twin_weight=args.twin_weight)
    print(f"New temporal twin study: {output}")
    print(f"Read: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
