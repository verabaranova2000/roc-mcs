"""Bounded diagnostic benchmark for the ROI Explorer 3D rendering pipeline.

Run from the installed project environment:
    python -m roc_mcs.gui.roi_explorer.volume_rendering_benchmark --points 25000 --webgl

The benchmark uses synthetic data only. It never opens an experimental HDF5 file,
changes application settings, or touches the live ROI Explorer widget.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from roc_mcs.gui.roi_explorer.volume_rendering import (
    MAX_RENDER_POINTS,
    build_volume_figure,
    prepare_volume,
)
from roc_mcs.roi.roi_types import ROIVolume


DEFAULT_POINTS = 25_000
MAX_SAFE_POINTS = min(int(MAX_RENDER_POINTS), 95_000)
MAX_SYNTHETIC_VOXELS = 6_000_000


def _timed(call: Callable[[], Any], repeats: int) -> tuple[Any, list[float]]:
    result = None
    samples: list[float] = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return result, samples


def _summary(samples: list[float]) -> dict[str, float]:
    values = [float(v) for v in samples]
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
    }


def _memory_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        vm = psutil.virtual_memory()
        result.update(
            process_rss_mib=round(process.memory_info().rss / (1024 ** 2), 1),
            system_ram_total_gib=round(vm.total / (1024 ** 3), 2),
            system_ram_available_gib=round(vm.available / (1024 ** 3), 2),
            system_ram_percent=round(float(vm.percent), 1),
        )
    except Exception:
        result["psutil"] = "not installed; process/system RAM metrics unavailable"
    return result


def _make_synthetic_roi(target_points: int) -> tuple[ROIVolume, dict[str, Any]]:
    """Create a deterministic compact test volume close to target sampled points."""
    target = max(1_000, min(int(target_points), MAX_SAFE_POINTS))
    n_theta = 32
    n_xy = max(32, 2 * int(math.floor(math.sqrt(target / (n_theta / 2)))))
    while n_xy * n_xy * n_theta > MAX_SYNTHETIC_VOXELS:
        n_xy -= 2
    if n_xy < 8:
        raise RuntimeError("Не удалось сформировать безопасный тестовый объём.")

    x = np.linspace(-1.0, 1.0, n_xy, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, n_xy, dtype=np.float64)
    theta = np.linspace(-1.0, 1.0, n_theta, dtype=np.float64)

    # Broadcasting avoids building several full 3D coordinate grids.
    xx = x.astype(np.float32)[:, None]
    yy = y.astype(np.float32)[None, :]
    tt = theta.astype(np.float32)
    lobe_a = np.exp(-(((xx - 0.22) ** 2 + (yy + 0.12) ** 2) / 0.19)).astype(np.float32)
    lobe_b = 0.72 * np.exp(-(((xx + 0.32) ** 2 + (yy - 0.28) ** 2) / 0.085)).astype(np.float32)
    angular_a = np.exp(-(((tt - 0.15) ** 2) / 0.22)).astype(np.float32)
    angular_b = 0.62 * np.exp(-(((tt + 0.42) ** 2) / 0.10)).astype(np.float32)
    plane = (lobe_a + lobe_b).astype(np.float32)
    volume = (plane[:, :, None] * angular_a[None, None, :] + 0.35 * plane[:, :, None] * angular_b[None, None, :]).astype(np.float32)

    # Constructor order matches the production ROI extraction code:
    # ROIVolume(x, y, theta, volume, coord_system, roi).
    rv = ROIVolume(x, y, theta, volume, "xy", None)
    info = {
        "synthetic_shape": list(volume.shape),
        "synthetic_voxels": int(volume.size),
        "synthetic_array_mib": round(volume.nbytes / (1024 ** 2), 2),
        "target_render_points": target,
    }
    return rv, info


def _build_cpu_report(points: int, repeats: int) -> tuple[dict[str, Any], str]:
    memory_before = _memory_snapshot()
    rv, data_info = _make_synthetic_roi(points)
    memory_after_data = _memory_snapshot()

    sampled_result, prepare_samples = _timed(
        lambda: prepare_volume(
            rv,
            smooth_sigma=0.5,
            stride=2,
            max_points=points,
        ),
        repeats,
    )
    sampled, projection, _x, _y, _z, values = sampled_result

    settings = {
        "font_family": "Times New Roman",
        "tick_size": 14,
        "label_size": 16,
        "axis_tick_length": 3,
        "show_grid": True,
        "show_projection": True,
        "show_colorbar": False,
        "show_legend": False,
        "volume_colormap": "magma",
        "projection_colorscale": "Greys",
        "volume_opacity": 0.10,
        "projection_opacity": 0.75,
        "isomin_fraction": 0.16,
        "isomax_fraction": 0.95,
        "surface_count": 25,
        "smooth_sigma": 0.5,
        "volume_stride": 2,
        "max_render_points": points,
        "dragmode": "orbit",
        "projection_type": "perspective",
        "rotation_speed": 14.0,
    }

    figure, figure_samples = _timed(
        lambda: build_volume_figure(rv, settings, "light"),
        repeats,
    )

    import plotly
    import plotly.io as pio

    payload, serialization_samples = _timed(
        lambda: pio.to_json(figure, validate=False, pretty=False, remove_uids=False),
        repeats,
    )
    payload_bytes = len(payload.encode("utf-8"))
    memory_after = _memory_snapshot()

    volume_trace = next((trace for trace in figure.data if getattr(trace, "type", "") == "volume"), None)
    surface_trace = next((trace for trace in figure.data if getattr(trace, "type", "") == "surface"), None)
    version_text = str(getattr(plotly, "__version__", "0.0"))
    try:
        version_pair = tuple(int(part) for part in version_text.split(".")[:2])
    except (TypeError, ValueError):
        version_pair = (0, 0)
    plotly_supported = (7, 1) <= version_pair < (8, 0)
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count_logical": os.cpu_count(),
        "plotly_version": version_text,
        "compatibility": {
            "plotly_required_by_widget": ">=7.1,<8.0",
            "plotly_supported_by_roi_widget": plotly_supported,
        },
        "data": data_info,
        "pipeline": {
            "prepare_volume": _summary(prepare_samples),
            "build_volume_figure_including_prepare": _summary(figure_samples),
            "json_serialization": _summary(serialization_samples),
            "json_payload_mib": round(payload_bytes / (1024 ** 2), 2),
            "volume_trace_points": int(np.asarray(volume_trace.value).size) if volume_trace is not None else 0,
            "projection_grid_points": int(np.asarray(surface_trace.surfacecolor).size) if surface_trace is not None else 0,
            "sampled_volume_shape": list(sampled.shape),
            "projection_shape": list(projection.shape),
            "finite_volume_points": int(np.asarray(values).size),
        },
        "memory": {
            "before": memory_before,
            "after_synthetic_data": memory_after_data,
            "after_figure_and_json": memory_after,
        },
        "webgl": {"status": "not_run", "reason": "Добавьте флаг --webgl для проверки Qt WebEngine/WebGL."},
    }
    return report, payload


def _run_webgl_probe(payload: str, timeout_s: int = 120) -> dict[str, Any]:
    """Render the synthetic production Plotly figure in a real Qt WebEngine page."""
    from roc_mcs.gui.qt_webengine_bootstrap import prepare_qt_webengine

    bootstrap_error = prepare_qt_webengine("ROI 3D performance benchmark")
    if bootstrap_error:
        return {"status": "failed", "error": f"Qt WebEngine bootstrap: {bootstrap_error}"}

    try:
        from importlib.metadata import PackageNotFoundError, version as package_version
        from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer, QUrl
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        import plotly
    except Exception as exc:
        return {"status": "failed", "error": f"Не удалось импортировать Qt WebEngine: {type(exc).__name__}: {exc}"}

    try:
        webengine_version = package_version("PyQt6-WebEngine")
    except PackageNotFoundError:
        return {"status": "failed", "error": "Не установлен пакет PyQt6-WebEngine."}
    match = re.match(r"^(\d+)\.(\d+)", str(webengine_version))
    webengine_pair = tuple(map(int, match.groups())) if match else (0, 0)
    if not (6, 11) <= webengine_pair < (7, 0):
        return {
            "status": "failed",
            "error": f"PyQt6-WebEngine {webengine_version} несовместим с GUI; требуется >=6.11,<7.0.",
        }

    js_file = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
    if not js_file.is_file() or js_file.stat().st_size < 100_000:
        return {"status": "failed", "error": f"Не найден локальный Plotly.js: {js_file}"}

    app = QApplication.instance() or QApplication([sys.argv[0]])
    view = QWebEngineView()
    view.setWindowTitle("ROI 3D diagnostic benchmark — synthetic data")
    view.resize(900, 650)
    web_settings = view.settings()
    for attr_name in ("JavascriptEnabled", "LocalContentCanAccessFileUrls", "WebGLEnabled"):
        attr = getattr(QWebEngineSettings.WebAttribute, attr_name, None)
        if attr is not None:
            web_settings.setAttribute(attr, True)
    remote = getattr(QWebEngineSettings.WebAttribute, "LocalContentCanAccessRemoteUrls", None)
    if remote is not None:
        web_settings.setAttribute(remote, False)

    # Validate JSON on the Python side, then stream it in small sequential IPC chunks.
    # setHtml() has a ~2 MiB percent-encoded limit, and a single very large
    # runJavaScript source string would also create unnecessary IPC pressure.
    json.loads(payload)
    transfer_chunk_chars = 64 * 1024
    transfer_chunks = [payload[i:i + transfer_chunk_chars] for i in range(0, len(payload), transfer_chunk_chars)]
    html = r'''<!doctype html><html><head><meta charset="utf-8">
<style>html,body,#plot{margin:0;width:100%;height:100%;overflow:hidden}#plot{position:absolute;inset:0}</style>
<script src="plotly.min.js"></script></head><body><div id="plot"></div><script>
window.__roi3dBenchResult = null;
window.__roi3dBenchError = null;
let __roi3dTransfer = null;
window.beginRoi3dBenchTransfer = function(count){
  if(!Number.isInteger(count)||count<1||count>2048)throw new Error('Недопустимое число частей Figure');
  __roi3dTransfer={chunks:new Array(count),received:0};
};
window.receiveRoi3dBenchChunk = function(index,count,chunk){
  try{
    if(!__roi3dTransfer||count!==__roi3dTransfer.chunks.length||!Number.isInteger(index)||index<0||index>=count||typeof chunk!=='string')throw new Error('Нарушена последовательность передачи Figure');
    if(__roi3dTransfer.chunks[index]===undefined){__roi3dTransfer.chunks[index]=chunk;__roi3dTransfer.received++;}
    if(__roi3dTransfer.received===count){
      const text=__roi3dTransfer.chunks.join('');__roi3dTransfer=null;
      window.runRoi3dBench(JSON.parse(text));
    }
  }catch(e){window.__roi3dBenchError=String(e&&e.stack||e);window.__roi3dBenchResult={status:'failed',error:window.__roi3dBenchError};}
};
const __frameIntervals=[];let __frameLast=performance.now(),__frameMonitor=true;
function __monitorFrames(now){__frameIntervals.push(now-__frameLast);__frameLast=now;if(__frameMonitor)requestAnimationFrame(__monitorFrames);}
requestAnimationFrame(__monitorFrames);
window.addEventListener('error', e => { window.__roi3dBenchError = String(e.message || e.error || 'JavaScript error'); });
window.addEventListener('unhandledrejection', e => { window.__roi3dBenchError = String(e.reason && e.reason.message || e.reason || 'Promise rejection'); });
function median(a){const b=[...a].sort((x,y)=>x-y);return b[Math.floor(b.length/2)]||0;}
function percentile(a,p){const b=[...a].sort((x,y)=>x-y);return b.length?b[Math.min(b.length-1,Math.max(0,Math.ceil((b.length-1)*p)))]:0;}
function rendererInfo(g){
  const canvases=[...g.querySelectorAll('canvas')];
  for(const canvas of canvases){
    let gl=null, kind='';
    try{gl=canvas.getContext('webgl2');if(gl)kind='WebGL 2';}catch(e){}
    if(!gl){try{gl=canvas.getContext('webgl')||canvas.getContext('experimental-webgl');if(gl)kind='WebGL 1';}catch(e){}}
    if(!gl)continue;
    let renderer='unknown', vendor='unknown', totalGpuMiB=null;
    try{
      const ext=gl.getExtension('WEBGL_debug_renderer_info');
      renderer=String(ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER));
      vendor=String(ext?gl.getParameter(ext.UNMASKED_VENDOR_WEBGL):gl.getParameter(gl.VENDOR));
    }catch(e){}
    try{
      const ext=gl.getExtension('GL_NVX_gpu_memory_info');
      if(ext){const value=gl.getParameter(ext.GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX);if(Number.isFinite(value)&&value>0)totalGpuMiB=Math.round(value/1024);}
    }catch(e){}
    return {available:true,version:kind,renderer,vendor,total_gpu_memory_mib:totalGpuMiB,canvas_count:canvases.length};
  }
  return {available:false,version:'none',renderer:'none',vendor:'none',canvas_count:canvases.length};
}
window.runRoi3dBench = async function(figure){
  try{
    if(!figure || !Array.isArray(figure.data) || !figure.layout) throw new Error('Переданная Plotly Figure повреждена');
    const graph=document.getElementById('plot');
    const started=performance.now();
    await Plotly.newPlot(graph,figure.data,figure.layout,{responsive:false,displayModeBar:false,scrollZoom:false});
    const initial=(performance.now()-started);
    const vi=figure.data.findIndex(t=>t.type==='volume');
    const pi=figure.data.findIndex(t=>t.type==='surface'&&t.name==='Интегральная проекция');
    if(vi<0)throw new Error('Volume trace не найден в диагностической фигуре');
    const thresholdBase=Number(figure.layout&&figure.layout.meta&&figure.layout.meta.roi3d&&figure.layout.meta.roi3d.threshold_base)||1;
    const contrastTimes=[];
    for(let i=0;i<3;i++){
      const t=performance.now();
      await Plotly.restyle(graph,{isomin:[thresholdBase*(0.14+i*0.01)],isomax:[thresholdBase*0.95]},[vi]);
      contrastTimes.push(performance.now()-t);
    }
    let projectionPaletteBefore=null, projectionPaletteAfter=null;
    if(pi>=0)projectionPaletteBefore=JSON.stringify(graph.data[pi].colorscale);
    const paletteTimes=[];
    for(let i=0;i<3;i++){
      const t=performance.now();
      await Plotly.restyle(graph,{colorscale:[i%2?'Magma':'Viridis']},[vi]);
      paletteTimes.push(performance.now()-t);
    }
    if(pi>=0)projectionPaletteAfter=JSON.stringify(graph.data[pi].colorscale);
    const widthBefore=graph.style.width;
    graph.style.width='97%';
    const resizeStart=performance.now();
    await Plotly.Plots.resize(graph);
    const resizeMs=performance.now()-resizeStart;
    graph.style.width=widthBefore||'100%';
    await Plotly.Plots.resize(graph);
    const baseCamera=(graph.layout&&graph.layout.scene&&graph.layout.scene.camera)||{eye:{x:1.6,y:-1.9,z:1.25},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1}};
    const cameraTimes=[];
    for(let i=0;i<3;i++){
      const shift=(i%2?1:-1)*0.035;
      const moved={eye:{x:baseCamera.eye.x+shift,y:baseCamera.eye.y,z:baseCamera.eye.z},center:{x:baseCamera.center.x+shift,y:baseCamera.center.y,z:baseCamera.center.z},up:baseCamera.up};
      const t=performance.now();await Plotly.relayout(graph,{'scene.camera':moved});cameraTimes.push(performance.now()-t);
    }
    await Plotly.relayout(graph,{'scene.camera':baseCamera});
    await new Promise(resolve=>setTimeout(resolve,250));
    __frameMonitor=false;
    const frameIntervals=__frameIntervals.slice();
    const gl=rendererInfo(graph);
    window.__roi3dBenchResult={
      status:'ok',initial_render_ms:initial,
      contrast_restyle_ms:{median:median(contrastTimes),samples:contrastTimes},
      volume_palette_restyle_ms:{median:median(paletteTimes),samples:paletteTimes},
      resize_ms:resizeMs,
      camera_relayout_ms:{median:median(cameraTimes),samples:cameraTimes},
      animation_frame_ms:{median:median(frameIntervals),p95:percentile(frameIntervals,0.95),max:Math.max(...frameIntervals),over_50ms:frameIntervals.filter(v=>v>50).length,samples:frameIntervals.length},
      webgl:gl,
      projection_palette_preserved:projectionPaletteBefore===projectionPaletteAfter,
      user_agent:navigator.userAgent
    };
  }catch(e){window.__roi3dBenchError=String(e&&e.stack||e);window.__roi3dBenchResult={status:'failed',error:window.__roi3dBenchError};}
};
</script></body></html>'''

    loop = QEventLoop()
    poll_timer = QTimer()
    poll_timer.setInterval(150)
    deadline = QTimer()
    deadline.setSingleShot(True)
    state: dict[str, Any] = {"result": None, "load_error": None, "started": False}
    app = QApplication.instance()
    quit_on_last_window = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)

    def finish_with_error(message: str) -> None:
        if state["result"] is None:
            state["result"] = {"status": "failed", "error": message}
        loop.quit()

    def read_result(raw: Any) -> None:
        if not raw:
            return
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return
        if isinstance(parsed, dict) and parsed.get("status") in {"ok", "failed"}:
            state["result"] = parsed
            loop.quit()

    def poll() -> None:
        view.page().runJavaScript(
            "window.__roi3dBenchResult ? JSON.stringify(window.__roi3dBenchResult) : "
            "(window.__roi3dBenchError ? JSON.stringify({status:'failed',error:window.__roi3dBenchError}) : '')",
            read_result,
        )

    def loaded(ok: bool) -> None:
        if not ok:
            finish_with_error("Qt WebEngine не смог загрузить локальную диагностическую страницу.")
            return
        if state["started"]:
            return
        state["started"] = True
        transfer_state = {"index": 0}

        def send_next_chunk(_result: Any = None) -> None:
            if state["result"] is not None:
                return
            index = transfer_state["index"]
            if index >= len(transfer_chunks):
                poll_timer.start()
                return
            chunk_literal = json.dumps(transfer_chunks[index], ensure_ascii=False, separators=(",", ":"))
            script = f"window.receiveRoi3dBenchChunk({index},{len(transfer_chunks)},{chunk_literal})"
            transfer_state["index"] += 1
            view.page().runJavaScript(script, send_next_chunk)

        def transfer_started(_result: Any = None) -> None:
            send_next_chunk()

        view.page().runJavaScript(
            f"window.beginRoi3dBenchTransfer({len(transfer_chunks)})",
            transfer_started,
        )

    poll_timer.timeout.connect(poll)
    deadline.timeout.connect(lambda: finish_with_error(f"Превышен лимит ожидания {timeout_s} с."))
    view.loadFinished.connect(loaded)
    base_url = QUrl.fromLocalFile(str(js_file.parent / "index.html"))
    view.setHtml(html, base_url)
    view.show()
    deadline.start(int(timeout_s * 1000))
    try:
        loop.exec()
    finally:
        poll_timer.stop()
        deadline.stop()
        try:
            page = view.page()
            page.triggerAction(page.WebAction.Stop)
            view.close()
            view.deleteLater()
            # Explicitly flush DeferredDelete before QApplication tears down its
            # default QWebEngineProfile; this avoids the profile/page lifetime warning.
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
        except Exception:
            pass
        app.setQuitOnLastWindowClosed(quit_on_last_window)
    return state["result"] or {"status": "failed", "error": "Нет результата от диагностической страницы."}


def _classify(report: dict[str, Any]) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not report.get("compatibility", {}).get("plotly_supported_by_roi_widget", True):
        return "FAIL", [
            "Установленная версия Plotly не соответствует ограничению самого ROI 3D widget (требуется >=7.1,<8.0).",
            "Проверьте активный Python interpreter и зависимости именно этого окружения.",
        ]
    webgl = report.get("webgl", {})
    if webgl.get("status") == "not_run":
        return "НЕ ПРОВЕРЕНО: CPU-only", ["GPU/WebGL и фактическая отрисовка Qt WebEngine не проверялись."]
    if webgl.get("status") != "ok":
        return "FAIL", [str(webgl.get("error", "WebGL benchmark failed"))]
    info = webgl.get("webgl", {})
    if not info.get("available"):
        return "FAIL", ["WebGL-контекст не создан; 3D Plotly в этом Qt WebEngine не подтверждён."]
    renderer = str(info.get("renderer", "")).lower()
    software_tokens = ("swiftshader", "llvmpipe", "softpipe", "swrast", "software rasterizer", "warp")
    if any(token in renderer for token in software_tokens):
        notes.append("Обнаружен программный рендерер, а не аппаратный GPU backend: " + str(info.get("renderer")))
    elif renderer.strip() in {"", "unknown", "none"}:
        notes.append("WebGL доступен, но модель GPU/рендерера не удалось достоверно определить.")
    if webgl.get("projection_palette_preserved") is False:
        notes.append("Проверка выявила изменение палитры 2D-проекции после обновления палитры 3D-объёма.")
        return "FAIL", notes

    contrast = float(webgl.get("contrast_restyle_ms", {}).get("median", math.inf))
    palette = float(webgl.get("volume_palette_restyle_ms", {}).get("median", math.inf))
    initial = float(webgl.get("initial_render_ms", math.inf))
    camera_ms = float(webgl.get("camera_relayout_ms", {}).get("median", math.inf))
    frame_p95 = float(webgl.get("animation_frame_ms", {}).get("p95", math.inf))
    frame_max = float(webgl.get("animation_frame_ms", {}).get("max", math.inf))
    if notes:
        return "WARN", notes
    if initial > 5_000 or contrast > 500 or palette > 500 or camera_ms > 500 or frame_max > 1_000:
        notes.append("Есть измеримые задержки/длинные блокировки главного потока; сравните тайминги с целевым лабораторным компьютером.")
        return "WARN", notes
    if initial > 2_000 or contrast > 150 or palette > 150 or camera_ms > 100 or frame_p95 > 50 or frame_max > 250:
        notes.append("Рендеринг работоспособен, но плавность не соответствует строгим интерактивным целям (контраст/палитра <150 мс, камера <100 мс, p95 кадра <50 мс).")
        return "WARN", notes
    notes.append("Синтетический тест прошёл заданные пороги. Это не гарантия отсутствия сбоев на реальном эксперименте.")
    return "PASS", notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Измерение CPU/Plotly pipeline и, опционально, реального Qt WebEngine/WebGL ROI 3D render."
    )
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help=f"Целевое число точек Volume trace (1000..{MAX_SAFE_POINTS}, default {DEFAULT_POINTS}).")
    parser.add_argument("--repeats", type=int, default=2, help="Число повторов CPU-измерений (1..5).")
    parser.add_argument("--webgl", action="store_true", help="Открыть диагностическое окно и измерить реальный Qt WebEngine/Plotly/WebGL.")
    parser.add_argument("--json-out", type=Path, default=None, help="Дополнительно сохранить полный отчёт в JSON.")
    args = parser.parse_args(argv)

    if not 1_000 <= args.points <= MAX_SAFE_POINTS:
        parser.error(f"--points должен быть от 1000 до {MAX_SAFE_POINTS}; не запускайте большие неконтролируемые тесты.")
    if not 1 <= args.repeats <= 5:
        parser.error("--repeats должен быть от 1 до 5.")

    memory = _memory_snapshot()
    available_gib = memory.get("system_ram_available_gib")
    if available_gib is not None and available_gib < 0.75:
        print("ОТМЕНА: доступно менее 0.75 GiB RAM. Диагностика не будет нагружать систему.", file=sys.stderr)
        return 2

    print("ROI 3D BENCHMARK — synthetic data only", flush=True)
    print(f"Точки: {args.points:,}; CPU-повторы: {args.repeats}; WebGL: {'да' if args.webgl else 'нет'}", flush=True)
    print("Тест ограничен лимитом приложения; реальные HDF5 и экспериментальные устройства не открываются.", flush=True)

    started = time.perf_counter()
    try:
        report, payload = _build_cpu_report(args.points, args.repeats)
    except Exception as exc:
        print(f"CPU PIPELINE: FAIL — {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    report["total_cpu_benchmark_seconds"] = round(time.perf_counter() - started, 3)

    pipeline = report["pipeline"]
    print("\nCPU / PLOTLY PIPELINE")
    for name, label in (
        ("prepare_volume", "prepare_volume"),
        ("build_volume_figure_including_prepare", "build_volume_figure (с prepare_volume)"),
        ("json_serialization", "JSON serialization"),
    ):
        item = pipeline[name]
        print(f"  {label}: median {item['median_ms']:.1f} ms  (min {item['min_ms']:.1f}; max {item['max_ms']:.1f})")
    print(f"  WebEngine JSON payload: {pipeline['json_payload_mib']:.2f} MiB")
    print(f"  Volume points: {pipeline['volume_trace_points']:,}; XY projection grid: {pipeline['projection_grid_points']:,}")
    compatibility = report.get("compatibility", {})
    if not compatibility.get("plotly_supported_by_roi_widget", False):
        print(f"  WARNING: Plotly {report['plotly_version']} incompatible; ROI 3D widget requires {compatibility.get('plotly_required_by_widget')}.")
    print(f"  Synthetic shape: {report['data']['synthetic_shape']}; memory for raw array: {report['data']['synthetic_array_mib']:.2f} MiB")

    if args.webgl:
        if not compatibility.get("plotly_supported_by_roi_widget", False):
            webgl_result = {
                "status": "skipped",
                "error": "Qt/WebGL-прогон пропущен: версия Plotly не соответствует требованию самого 3D GUI.",
            }
        else:
            print("\nЗапуск Qt WebEngine; откроется небольшое диагностическое окно...", flush=True)
            webgl_result = _run_webgl_probe(payload)
        report["webgl"] = webgl_result
        report.setdefault("memory", {})["after_webgl"] = _memory_snapshot()
        if webgl_result.get("status") == "ok":
            print("QT WEBENGINE / WEBGL")
            print(f"  Initial render: {webgl_result['initial_render_ms']:.1f} ms")
            print(f"  Contrast restyle median: {webgl_result['contrast_restyle_ms']['median']:.1f} ms")
            print(f"  Volume palette restyle median: {webgl_result['volume_palette_restyle_ms']['median']:.1f} ms")
            print(f"  Resize: {webgl_result['resize_ms']:.1f} ms")
            camera_result = webgl_result.get("camera_relayout_ms", {})
            if camera_result:
                print(f"  Camera relayout median: {camera_result.get('median', float('nan')):.1f} ms")
            print(f"  Frame interval median/p95/max: {webgl_result['animation_frame_ms']['median']:.1f}/{webgl_result['animation_frame_ms'].get('p95', float('nan')):.1f}/{webgl_result['animation_frame_ms']['max']:.1f} ms; >50 ms: {webgl_result['animation_frame_ms'].get('over_50ms', 0)}")
            info = webgl_result.get("webgl", {})
            print(f"  WebGL: {info.get('version')}; renderer: {info.get('renderer')}; vendor: {info.get('vendor')}")
            print(f"  2D projection palette preserved: {webgl_result.get('projection_palette_preserved')}")
        memory_after_webgl = report.get("memory", {}).get("after_webgl", {})
        process_rss = memory_after_webgl.get("process_rss_mib")
        available_ram = memory_after_webgl.get("system_ram_available_gib")
        if process_rss is not None:
            print(f"  Python process RSS after WebGL: {process_rss:.1f} MiB")
        if available_ram is not None:
            print(f"  System RAM available after WebGL: {available_ram:.2f} GiB")
        else:
            print(f"QT WEBENGINE / WEBGL: FAIL — {webgl_result.get('error', 'unknown error')}")

    verdict, notes = _classify(report)
    report["verdict"] = verdict
    report["verdict_notes"] = notes
    print(f"\nИТОГ: {verdict}")
    for note in notes:
        print(f"  - {note}")
    print("Важно: benchmark проверяет только ограниченный синтетический сценарий; он не гарантирует безопасность всех драйверов, VRAM, температур или параллельных лабораторных процессов.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Полный JSON-отчёт: {args.json_out.resolve()}")

    return 0 if verdict == "PASS" else (2 if verdict.startswith("НЕ ПРОВЕРЕНО") else 1)


if __name__ == "__main__":
    raise SystemExit(main())
