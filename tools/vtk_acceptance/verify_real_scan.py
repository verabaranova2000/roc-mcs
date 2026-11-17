from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _paths(h5: h5py.File, scan: str) -> tuple[str, str]:
    for data_path, theta_path in ((f"{scan}/measurement/Frame", f"{scan}/measurement/th"),
                                  (f"{scan}/raw_data_2D", f"{scan}/raw_data_th")):
        if data_path in h5 and theta_path in h5:
            return data_path, theta_path
    raise ValueError(f"Не найдены поддерживаемые data/theta datasets для {scan!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Низкопамятная сверка VTK H5 loader с точечным чтением исходного HDF5")
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--probes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=472)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    h5_path = args.h5.expanduser().resolve()
    before = h5_path.stat()
    from roc_mcs.gui.roi_explorer.vtk_backend.h5_volume_source import H5VolumeLoader
    loader = H5VolumeLoader(h5_path)
    loader.set_bad_pixels([])
    payload = loader.load(args.scan, mode="full", remove_spikes=False, smoothing_sigma=0.0, theta_range=None)
    with h5py.File(h5_path, "r") as h5:
        data_path, theta_path = _paths(h5, args.scan)
        ds = h5[data_path]
        theta_source = np.asarray(h5[theta_path]).squeeze()
        angle_axes = [i for i in (0, 2) if ds.shape[i] == theta_source.size]
        if len(angle_axes) != 1:
            raise ValueError(f"Ось θ неоднозначна: data={ds.shape}, theta={theta_source.shape}.")
        angle_axis = angle_axes[0]
        if angle_axis == 0:
            ny, nx = int(ds.shape[1]), int(ds.shape[2])
        else:
            ny, nx = int(ds.shape[0]), int(ds.shape[1])
        if payload.shape[:2] != (nx, ny) or payload.shape[2] != theta_source.size:
            raise AssertionError(f"Размерность payload {payload.shape} не соответствует HDF5 ({nx}, {ny}, {theta_source.size}).")
        theta_expected = theta_source if theta_source[0] < theta_source[-1] else theta_source[::-1]
        if not np.array_equal(payload.theta, theta_expected):
            raise AssertionError("Ось θ payload не совпадает с исходной осью, отсортированной по возрастанию.")
        positions = [(0, 0), (0, ny - 1), (nx - 1, 0), (nx - 1, ny - 1), (nx // 2, ny // 2)]
        rng = np.random.default_rng(args.seed)
        positions.extend((int(rng.integers(0, nx)), int(rng.integers(0, ny))) for _ in range(max(0, args.probes - len(positions))))
        unique = list(dict.fromkeys(positions))
        results = []
        for xi, yi in unique:
            expected = np.asarray(ds[:, yi, xi] if angle_axis == 0 else ds[yi, xi, :])
            if theta_source[0] > theta_source[-1]:
                expected = expected[::-1]
            expected = expected.astype(payload.volume.dtype, copy=False)
            actual = np.asarray(payload.volume[xi, yi, :])
            if not np.array_equal(actual, expected, equal_nan=True):
                raise AssertionError(f"Несовпадение volume[x={xi}, y={yi}, theta] с прямым чтением HDF5.")
            finite = np.isfinite(expected)
            expected_sum = float(np.sum(expected[finite], dtype=np.float64)) if finite.any() else float("nan")
            actual_sum = float(payload.projection[xi, yi])
            if not (np.isnan(expected_sum) and np.isnan(actual_sum)) and not np.isclose(actual_sum, expected_sum, rtol=0.0, atol=1e-12):
                raise AssertionError(f"Несовпадение XY projection в пикселе x={xi}, y={yi}: {actual_sum} != {expected_sum}.")
            results.append({"x_index": xi, "y_index": yi, "theta_samples": int(expected.size),
                            "finite_theta": int(finite.sum()), "projection_sum": expected_sum, "status": "PASS"})
    after = h5_path.stat()
    unchanged = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    if not unchanged:
        raise AssertionError("Размер или время изменения исходного HDF5 изменились во время сверки.")
    report = {"status": "PASS", "time": datetime.now().astimezone().isoformat(), "h5": str(h5_path),
              "scan": args.scan, "dataset": data_path, "theta_dataset": theta_path, "theta_axis_in_source": angle_axis,
              "source_shape": tuple(int(v) for v in ds.shape), "payload_shape": payload.shape,
              "payload_dtype": payload.scalar_dtype, "coord_system": payload.coord_system,
              "checks": ["full-volume shape", "ascending theta axis", "selected voxel vectors equal direct HDF5 reads",
                         "selected XY sums equal independent finite-only sum", "source HDF5 stat unchanged"],
              "probe_count": len(results), "probes": results, "source_unchanged": unchanged,
              "limitations": ["Проверяется только full mode без spikes, smoothing, ROI и global bad-pixel mask.",
                              "Это численная сверка адаптера с исходными отсчётами, а не независимая физическая калибровка."]}
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
