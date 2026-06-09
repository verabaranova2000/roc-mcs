from __future__ import annotations

import argparse
from pathlib import Path

from roc_mcs.pipeline.build import run_experiment
from roc_mcs.export import export_roc_map_excel
from roc_mcs.plots import save_figure


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="roc-mcs",
        description="Build ROC map from MCS files."
    )
    parser.add_argument("folder", type=Path, 
                        help="Folder with .mcs files")
    parser.add_argument("--amplitude", type=float, required=True,
                        help="Current piezo amplitude, mVpp")
    parser.add_argument("--reference-amplitude", type=float, default=400.0,
                        help="Reference amplitude used for calibration, mVpp")
    parser.add_argument("--reference-angle", type=float, default=180.0,
                        help="Reference angle used for calibration, arcsec",)
    parser.add_argument("--results-folder", type=Path, default=None,
                        help="Folder for output files")

    args = parser.parse_args()

    results_folder = args.results_folder or (args.folder / "results")

    roc_map, fig = run_experiment(
        folder=args.folder,
        amplitude=args.amplitude,
        reference_amplitude=args.reference_amplitude,
        reference_angle=args.reference_angle,
    )

    save_figure(fig, results_folder, "roc_map.png")

    export_roc_map_excel(
        roc_map,
        folder=results_folder,
        filename="rocking_curve_dynamics.xlsx",
    )

    print(f"Done. Results saved to: {results_folder.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())