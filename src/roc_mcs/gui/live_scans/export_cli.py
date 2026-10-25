"""Внутренний CLI: экспорт без GUI в отдельном процессе."""
from __future__ import annotations
import json, os, sys, traceback
from pathlib import Path

def main() -> int:
    if len(sys.argv) != 2: print("Ожидается путь к JSON-заданию", file=sys.stderr); return 2
    root = Path(__file__).resolve().parents[4]
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path: sys.path.insert(0, str(src))
    os.environ["MPLBACKEND"] = "Agg"
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        from roc_mcs.pipeline.build import run_experiment
        from roc_mcs.diagnostics.registry import DIAGNOSTICS
        diagnostics = tuple(DIAGNOSTICS.keys()) if request.get("save_diagnostics") else ()
        artifact = run_experiment(input_folder=request["input_folder"], amplitude=float(request["amplitude"]),
            reference_amplitude=float(request["reference_amplitude"]), reference_angle=float(request["reference_angle"]),
            output_folder=request["output_folder"], save_excel_flag=bool(request.get("save_excel_flag", True)),
            save_figure_flag=bool(request.get("save_figure_flag", True)), save_artifact=bool(request.get("save_artifact", False)),
            diagnostics=diagnostics, compute_moments=bool(request.get("compute_moments", True)),
            fit_models=(), trajectory_models=(), branch=str(request.get("branch", "up")))
        output = Path(request["output_folder"])
        files = sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())
        print(json.dumps({"output_folder": str(output), "files": files}, ensure_ascii=False))
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
