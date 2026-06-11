from __future__ import annotations

import argparse
import yaml
from pathlib import Path

from roc_mcs.io.config import load_config
from roc_mcs.pipeline.build import run_experiment
from roc_mcs.export import export_roc_map_excel
from roc_mcs.plots import save_figure


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="roc-mcs",
        description="Build ROC map from MCS files."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, 
                       help="Path to YAML config file")
    group.add_argument("--input-folder", 
                       type=Path, 
                        help="Folder with .mcs files")
    parser.add_argument("--amplitude", type=float,
                        help="Current piezo amplitude, mVpp")
    parser.add_argument("--reference-amplitude", type=float, default=400.0,
                        help="Reference amplitude used for calibration, mVpp")
    parser.add_argument("--reference-angle", type=float, default=180.0,
                        help="Reference angle used for calibration, arcsec",)
    parser.add_argument("--output-folder", type=Path, default=None,
                        help="Folder for output files")

    args = parser.parse_args()

    if args.config is not None:
        cfg = load_config(args.config)

        input_folder = Path(cfg["input_folder"])
        amplitude = cfg["amplitude"]
        reference_amplitude = cfg.get("reference_amplitude", 400.0)
        reference_angle = cfg.get("reference_angle", 180.0)
        output_folder = cfg.get("output_folder")
    else:
        if args.amplitude is None:
            parser.error("--amplitude is required when --config is not used")        
        input_folder = args.input_folder
        amplitude = args.amplitude
        reference_amplitude = args.reference_amplitude
        reference_angle = args.reference_angle
        output_folder = args.output_folder

    output_folder = output_folder or (input_folder / "results")

    roc_map, fig = run_experiment(
        folder=input_folder,
        amplitude=amplitude,
        reference_amplitude=reference_amplitude,
        reference_angle=reference_angle,
    )

    save_figure(fig, output_folder, "roc_map.png")

    export_roc_map_excel(
        roc_map,
        folder=output_folder,
        filename="rocking_curve_dynamics.xlsx",
    )

    print(f"Done. Results saved to: {output_folder.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())