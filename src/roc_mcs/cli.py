from __future__ import annotations

import argparse
import yaml
from pathlib import Path

from roc_mcs.io.config import load_config
from roc_mcs.pipeline.build import run_experiment
from roc_mcs.reporting import print_run_report
from roc_mcs.diagnostics.registry import DIAGNOSTICS


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="roc-mcs",
        description="Build ROC map from MCS files."
    )
    # === Добавляем аргументы CLI ===
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", type=Path, 
                       help="Path to YAML config file")
    group.add_argument("--input-folder", type=Path,
                        help="Folder with .mcs files (default: current directory)")
    parser.add_argument("--amplitude", type=float,
                        help="Current piezo amplitude, mVpp")
    parser.add_argument("--reference-amplitude", type=float, default=400.0,
                        help="Reference amplitude used for calibration, mVpp")
    parser.add_argument("--reference-angle", type=float, default=180.0,
                        help="Reference angle used for calibration, arcsec",)
    parser.add_argument("--output-folder", type=Path, default=None,
                        help="Folder for output files")
    parser.add_argument("--diagnostics", type=str, default="",
                        help="Comma-separated list of diagnostics, e.g. phase,branch")
    parser.add_argument("--fit-models", type=str, default="", 
                        help="Comma-separated list of fitting models, e.g. gauss,pvoigt")
    parser.add_argument("--trajectory-models", type=str, default="",
                        help="Comma-separated list of trajectory models, e.g. gauss,pvoigt")    
    parser.add_argument("--save-artifact", action="store_true",
                        help="Save ExperimentArtifact (*.pkl)")
    args = parser.parse_args()

    # === Ветка --config ===
    if args.config is not None:
        cfg = load_config(args.config)
        input_folder = Path(cfg["input_folder"])
        amplitude = cfg["amplitude"]
        reference_amplitude = cfg.get("reference_amplitude", 400.0)
        reference_angle = cfg.get("reference_angle", 180.0)
        output_folder = cfg.get("output_folder")
        #diag_names = cfg.get("diagnostics", [])
        #diagnostics = [DIAGNOSTICS[name] for name in diag_names]        
        diagnostics = cfg.get("diagnostics", [])
        fit_models = cfg.get("fit_models", [])
        trajectory_models = cfg.get("trajectory_models", [])
        save_artifact = cfg.get("save_artifact", False) or args.save_artifact
    # === Ветка аргументов командной строки ===
    else:
        if args.amplitude is None:
            parser.error("--amplitude is required when --config is not used")        
        input_folder = args.input_folder or Path.cwd()
        amplitude = args.amplitude
        reference_amplitude = args.reference_amplitude
        reference_angle = args.reference_angle
        output_folder = args.output_folder
        #diag_names = [name.strip() for name in args.diagnostics.split(",") if name.strip()]
        #diagnostics = [DIAGNOSTICS[name] for name in diag_names]
        diagnostics = [name.strip() for name in args.diagnostics.split(",") if name.strip()]
        fit_models = [name.strip() for name in args.fit_models.split(",") if name.strip()]
        trajectory_models = [name.strip() for name in args.trajectory_models.split(",") if name.strip()]
        save_artifact = args.save_artifact
    output_folder = output_folder or (input_folder / "results")


    artifact = run_experiment(
        input_folder=input_folder,
        amplitude=amplitude,
        reference_amplitude=reference_amplitude,
        reference_angle=reference_angle,
        output_folder=output_folder,
        save_excel_flag=True,
        save_figure_flag=True,
        save_artifact=save_artifact,
        diagnostics=diagnostics,
        fit_models=fit_models,
        trajectory_models=trajectory_models,
    )
    print_run_report(output_folder, artifact.output_files)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())