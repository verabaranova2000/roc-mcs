# src/roc_mcs/cli.py

import argparse
from pathlib import Path

from roc_mcs.pipeline.build import run_experiment


def main():
    parser = argparse.ArgumentParser(description="ROC-MCS experiment runner")

    parser.add_argument("folder", type=str, help="Path to experiment folder")
    parser.add_argument("--amplitude", type=float, default=400)
    parser.add_argument("--ref_amplitude", type=float, default=400)
    parser.add_argument("--ref_angle", type=float, default=180)
    parser.add_argument("--results", type=str, default=None)

    args = parser.parse_args()

    folder = Path(args.folder)

    results_folder = Path(args.results) if args.results else folder / "results"

    print("Running ROC-MCS pipeline...")

    roc_map, fig = run_experiment(
        folder=folder,
        amplitude=args.amplitude,
        reference_amplitude=args.ref_amplitude,
        reference_angle=args.ref_angle,
        results_folder=results_folder,
    )

    print("Done")
    print(f"Results saved to: {results_folder}")


if __name__ == "__main__":
    main()