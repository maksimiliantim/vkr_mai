from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing_cse2018 import DEFAULT_N_WINDOWS, preprocess_cse2018_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CSE-CIC-IDS2018 raw CSV files for the PINN experiment."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory or single CSV file with raw CSE-CIC-IDS2018 data.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/cse2018_preprocessed",
        help="Directory where cse2018_preprocessed.npz and preprocessing artifacts are written.",
    )
    parser.add_argument(
        "--n_windows",
        type=int,
        default=DEFAULT_N_WINDOWS,
        help="Number of sequential aggregation windows.",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--file_limit", type=int, default=None)
    parser.add_argument("--exclude_large_mb", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_cse2018_pipeline(
        data_dir=args.data_dir,
        output_dir=output_dir,
        n_windows=args.n_windows,
        chunksize=args.chunksize,
        file_limit=args.file_limit,
        exclude_large_mb=args.exclude_large_mb,
    )
    print(f"Prepared CSE-CIC-IDS2018 NPZ: {output_dir / 'cse2018_preprocessed.npz'}")


if __name__ == "__main__":
    main()
