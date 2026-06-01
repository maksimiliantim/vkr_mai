from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing import preprocess_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CIC-IDS2017 raw CSV files for the PINN experiment."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory with raw CIC-IDS2017 MachineLearningCVE CSV files.",
    )
    parser.add_argument(
        "--output",
        default="data/cic2017_preprocessed.npz",
        help="Output NPZ path consumed by run_experiment.py.",
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        default=60.0,
        help="Aggregation window size in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preprocess_pipeline(
        data_dir=args.data_dir,
        delta_t_sec=args.delta_t,
        save_path=str(output),
    )
    print(f"Prepared CIC-IDS2017 NPZ: {output}")


if __name__ == "__main__":
    main()
