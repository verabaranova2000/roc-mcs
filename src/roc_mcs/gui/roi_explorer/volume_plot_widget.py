"""Интерактивный Plotly/WebEngine 3D-экран ROI Explorer."""
from __future__ import annotations
import json, math, logging, re, sys
import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QTimer, QElapsedTimer, QEvent, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel,
    QMessageBox, QVBoxLayout, QWidget)
from roc_mcs.roi.extraction import extract_roi_volume
from roc_mcs.roi.roi_types import ROI
from roc_mcs.gui.roi_explorer.theme import apply_popup_theme, apply_widget_palette, resolve_theme, theme_colors
from roc_mcs.gui.roi_explorer.volume_rendering import build_volume_figure, effective_volume_stride, effective_extraction_stride, crop_raw_to_roi, MAX_CACHE_VOXELS, MAX_RENDER_POINTS
from roc_mcs.gui.roi_explorer.volume_tools_widget import VolumeToolsWidget

try:
    from PyQt6 import QtCore
    _qt_app_before_webengine_import = QtCore.QCoreApplication.instance()
    _shared_contexts_before_webengine_import = bool(QtCore.QCoreApplication.testAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
    ))
    # Jupyter импортирует ExplorerWindow до QApplication — это допустимый путь.
    # TRS Apps обязана предварительно загрузить WebEngine через общий bootstrap.
    # Если приложение уже создано без shared contexts, повторный поздний импорт
    # здесь лишь повторил бы заведомо известную ошибку Qt и замаскировал её.
    if (
        _qt_app_before_webengine_import is not None
        and not _shared_contexts_before_webengine_import
        and "PyQt6.QtWebEngineWidgets" not in sys.modules
    ):
        raise RuntimeError(
            "Поздний импорт QtWebEngineWidgets заблокирован: QCoreApplication "
            "уже создан, AA_ShareOpenGLContexts выключен, а TRS Apps не выполнил "
            "ранний bootstrap в точке входа."
        )
    from PyQt6.QtWebChannel import QWebChannel
    from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_ERROR = None
except Exception as exc:
    QWebChannel = QWebEnginePage = QWebEngineSettings = QWebEngineView = None
    _WEBENGINE_ERROR = exc

_HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body,#plot{margin:0;width:100%;height:100%;min-height:1px;overflow:hidden;background:__BG__}html,body{padding:0;user-select:none;overscroll-behavior:none}#plot{position:absolute;inset:0;contain:strict}</style>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script><script src="plotly.min.js"></script></head>
<body><div id="plot"></div><script>
let bridge=null, webglOk=false, graph=document.getElementById('plot'), camera={eye:{x:1.6,y:-1.9,z:1.25},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1}}, rotating=false, raf=0, lastT=0, speed=14, internalCamera=false, middle=null, pendingErrors=[], plotEventsBound=false, pendingFigure=null, renderBusy=false, renderId=0, resizeTimer=0, resizeBusy=false, resizePending=false, forceResize=false, cameraBusy=false, lastRenderedFigure=null, boundCanvases=new WeakSet(), activeTransferId=0, transferChunks=null, transferCount=0;
function reportError(message){const text=String(message&&message.message||message||'Неизвестная ошибка');if(bridge){try{bridge.renderError(text);}catch(e){}}else pendingErrors.push(text);}
window.addEventListener('error',e=>reportError((e.message||'JavaScript error')+(e.filename?' · '+e.filename.split('/').pop():'')+(e.lineno?':'+e.lineno:'')));
window.addEventListener('unhandledrejection',e=>reportError('Ошибка отрисовки Plotly: '+(e.reason&&e.reason.message||e.reason||'отклонён Promise')));
function webglAvailable(){try{const c=document.createElement('canvas'),gl=c.getContext('webgl2')||c.getContext('webgl')||c.getContext('experimental-webgl');if(!gl)return false;const ok=Number(gl.getParameter(gl.MAX_TEXTURE_SIZE))>0,lose=gl.getExtension('WEBGL_lose_context');if(lose)lose.loseContext();return ok;}catch(e){return false;}}
function norm(v){const n=Math.hypot(v.x,v.y,v.z)||1;return {x:v.x/n,y:v.y/n,z:v.z/n};}
function cross(a,b){return {x:a.y*b.z-a.z*b.y,y:a.z*b.x-a.x*b.z,z:a.x*b.y-a.y*b.x};}
function emitCamera(){if(bridge)bridge.cameraChanged(JSON.stringify(camera));}
function bindPlotEvents(){if(plotEventsBound)return true;if(typeof graph.on!=='function')return false;graph.on('plotly_relayout',function(e){const c=e['scene.camera']||e.scene?.camera;if(c){camera=c;if(!internalCamera&&bridge){setRotating(false);bridge.autoRotateStopped();emitCamera();}}});plotEventsBound=true;return true;}
function bindCanvasEvents(){graph.querySelectorAll('canvas').forEach(c=>{if(boundCanvases.has(c))return;boundCanvases.add(c);c.addEventListener('webglcontextlost',e=>{e.preventDefault();reportError('Потерян контекст WebGL; выполняется восстановление 3D-сцены.');});c.addEventListener('webglcontextrestored',()=>{if(lastRenderedFigure){pendingFigure=JSON.stringify(lastRenderedFigure);requestAnimationFrame(flushRender);}else scheduleResize(120);});});}
function recoverAfterShow(){if(graph._fullLayout){forceResize=true;scheduleResize(120);}else if(pendingFigure!==null)requestAnimationFrame(flushRender);}
function abortTransfer(){activeTransferId=-1;transferChunks=null;transferCount=0;pendingFigure=null;renderId++;}
function purgePlot(){abortTransfer();if(window.Plotly)Plotly.purge(graph);plotEventsBound=false;lastPlotSize='';lastRenderedFigure=null;}
function setCamera(c,emit=true){if(!c||!window.Plotly||!graph._fullLayout)return;camera=Object.assign({},camera,c);internalCamera=true;Plotly.relayout(graph,{'scene.camera':camera}).then(()=>{internalCamera=false;if(emit)emitCamera();}).catch(e=>{internalCamera=false;reportError(e);});}
function fitView(){if(!graph._fullLayout)return;let d={x:camera.eye.x-camera.center.x,y:camera.eye.y-camera.center.y,z:camera.eye.z-camera.center.z};d=norm(d);const r=Math.hypot(1.6,1.9,1.25);setCamera({eye:{x:camera.center.x+d.x*r,y:camera.center.y+d.y*r,z:camera.center.z+d.z*r}});}
function tick(t){if(!rotating)return;if(!lastT)lastT=t;if(t-lastT<40){raf=requestAnimationFrame(tick);return;}const a=(t-lastT)*speed*Math.PI/180000;lastT=t;const dx=camera.eye.x-camera.center.x,dy=camera.eye.y-camera.center.y,c=Math.cos(a),s=Math.sin(a);camera.eye={x:camera.center.x+dx*c-dy*s,y:camera.center.y+dx*s+dy*c,z:camera.eye.z};if(!cameraBusy&&graph._fullLayout){cameraBusy=true;internalCamera=true;Plotly.relayout(graph,{'scene.camera':camera}).catch(reportError).finally(()=>{cameraBusy=false;internalCamera=false;});}raf=requestAnimationFrame(tick);}
function setRotating(on){rotating=!!on;lastT=0;cancelAnimationFrame(raf);if(rotating)raf=requestAnimationFrame(tick);}
function alignView(axis){if(!graph._fullLayout)return;const ctr=camera.center||{x:0,y:0,z:0},dist=Math.max(0.2,Math.hypot(camera.eye.x-ctr.x,camera.eye.y-ctr.y,camera.eye.z-ctr.z)||2.1);let eye,up={x:0,y:0,z:1};if(axis==='x')eye={x:ctr.x+dist,y:ctr.y,z:ctr.z};else if(axis==='y')eye={x:ctr.x,y:ctr.y+dist,z:ctr.z};else if(axis==='z'){eye={x:ctr.x,y:ctr.y,z:ctr.z+dist};up={x:0,y:1,z:0};}else{eye={x:ctr.x+dist*0.625,y:ctr.y+dist*0.78,z:ctr.z+dist*0.08};up={x:0,y:0,z:1};}setCamera({eye:eye,center:ctr,up:up});}
let lastPlotSize='';async function flushResize(){if(!window.Plotly||!graph._fullLayout||renderBusy)return;const r=graph.getBoundingClientRect(),key=Math.round(r.width)+'x'+Math.round(r.height),forced=forceResize;forceResize=false;if(key===lastPlotSize&&!forced)return;if(resizeBusy){resizePending=true;return;}resizeBusy=true;try{await Plotly.Plots.resize(graph);lastPlotSize=key;}catch(e){reportError(e);}finally{resizeBusy=false;if(resizePending){resizePending=false;scheduleResize(45);}if(pendingFigure!==null)requestAnimationFrame(flushRender);}}function scheduleResize(delay=45){clearTimeout(resizeTimer);resizeTimer=setTimeout(flushResize,delay);}
async function flushRender(){if(renderBusy||resizeBusy||pendingFigure===null)return;const payload=pendingFigure;pendingFigure=null;renderBusy=true;const id=++renderId;try{const f=JSON.parse(payload);if(!webglOk){reportError('WebGL недоступен в Qt WebEngine. Проверьте драйвер видеокарты и аппаратное ускорение.');return;}if(!window.Plotly||typeof Plotly.react!=='function'){reportError('Локальный Plotly.js не загрузился.');return;}const bg=(f.layout&&f.layout.paper_bgcolor)||'#fff',rect=graph.getBoundingClientRect(),renderSize=Math.round(rect.width)+'x'+Math.round(rect.height);document.documentElement.style.background=bg;document.body.style.background=bg;graph.style.background=bg;await Plotly.react(graph,f.data,f.layout,{responsive:false,displayModeBar:false,scrollZoom:true,doubleClick:false,showTips:false});if(id!==renderId){Plotly.purge(graph);return;}lastPlotSize=renderSize;lastRenderedFigure=f;bindCanvasEvents();if(!bindPlotEvents())throw new Error('Plotly не предоставил API событий после создания графика');camera=(f.layout.scene&&f.layout.scene.camera)||camera;if(pendingFigure===null&&id===renderId){emitCamera();if(bridge)bridge.rendered();}}catch(e){reportError('Plotly не смог отрисовать сцену: '+(e&&e.message||e));}finally{renderBusy=false;if(pendingFigure!==null)requestAnimationFrame(flushRender);else scheduleResize(45);}}
function beginFigure(id,count){activeTransferId=id;transferCount=count;transferChunks=new Array(count);if(!count)reportError('Пустая передача 3D-сцены.');}
function receiveFigureChunk(id,index,chunk){if(id!==activeTransferId||!transferChunks||index<0||index>=transferCount)return;transferChunks[index]=chunk;if(bridge)bridge.chunkReceived(id,index);if(index===transferCount-1&&transferChunks.every(v=>typeof v==='string')){const payload=transferChunks.join('');transferChunks=null;transferCount=0;renderFigure(payload);}}
function renderFigure(s){pendingFigure=s;if(!renderBusy)requestAnimationFrame(flushRender);}
function applyCommand(s){let c={};try{c=JSON.parse(s);}catch(e){reportError('Не удалось разобрать команду 3D: '+e);return;}try{if(c.type==='fit')fitView();else if(c.type==='align')alignView(c.axis||'xy');else if(c.type==='camera'){if(c.projection&&graph._fullLayout)Plotly.relayout(graph,{'scene.camera.projection.type':c.projection});setCamera(c.camera);}else if(c.type==='rotate')setRotating(c.enabled);else if(c.type==='speed')speed=Number(c.value)||14;else if(c.type==='rotationLock'&&graph._fullLayout)Plotly.relayout(graph,{'scene.dragmode':c.enabled?'turntable':'orbit'}).catch(reportError);else if(c.type==='theme'){const bg=c.theme==='dark'?'#111316':'#fff';document.documentElement.style.background=bg;document.body.style.background=bg;graph.style.background=bg;}}catch(e){reportError(e);}}
graph.addEventListener('mousedown',function(e){if(e.button!==1)return;e.preventDefault();setRotating(false);if(bridge)bridge.autoRotateStopped();middle={x:e.clientX,y:e.clientY,camera:JSON.parse(JSON.stringify(camera))};},{capture:true});
window.addEventListener('mousemove',function(e){if(!middle)return;e.preventDefault();const c=middle.camera,dx=e.clientX-middle.x,dy=e.clientY-middle.y,view=norm({x:c.center.x-c.eye.x,y:c.center.y-c.eye.y,z:c.center.z-c.eye.z}),right=norm(cross(view,c.up)),screenUp=norm(cross(right,view)),dist=Math.hypot(c.eye.x-c.center.x,c.eye.y-c.center.y,c.eye.z-c.center.z),scale=dist*0.0018,shift={x:(-dx*right.x+dy*screenUp.x)*scale,y:(-dx*right.y+dy*screenUp.y)*scale,z:(-dx*right.z+dy*screenUp.z)*scale};setCamera({eye:{x:c.eye.x+shift.x,y:c.eye.y+shift.y,z:c.eye.z+shift.z},center:{x:c.center.x+shift.x,y:c.center.y+shift.y,z:c.center.z+shift.z}},false);},{capture:true});
window.addEventListener('mouseup',function(e){if(e.button===1&&middle){middle=null;emitCamera();}},{capture:true});
document.addEventListener('visibilitychange',()=>{if(!document.hidden)recoverAfterShow();});
let resizeObserver=null, resizeRaf=0, resizeLastKey='';
function installResizeObserver(){
    if(resizeObserver||typeof ResizeObserver!=='function')return;
    resizeObserver=new ResizeObserver(()=>{
        if(resizeRaf)return;
        resizeRaf=requestAnimationFrame(()=>{
            resizeRaf=0;
            const r=graph.getBoundingClientRect(),key=Math.round(r.width)+'x'+Math.round(r.height);
            if(key===resizeLastKey)return;
            resizeLastKey=key;
            scheduleResize(55);
        });
    });
    resizeObserver.observe(graph);
}
installResizeObserver();
if(typeof QWebChannel!=='function'||typeof qt==='undefined'||!qt.webChannelTransport){reportError('Qt WebChannel недоступен; проверьте загрузку qwebchannel.js.');}else new QWebChannel(qt.webChannelTransport,function(ch){bridge=ch.objects.bridge;bridge.figureStarted.connect(beginFigure);bridge.figureChunk.connect(receiveFigureChunk);bridge.commandChanged.connect(applyCommand);webglOk=webglAvailable();window.__roi3dWebglOk=webglOk;window.__roi3dBridgeReady=true;const bootstrapErrors=[];if(!window.Plotly)bootstrapErrors.push('Plotly.js не загружен: проверьте установку пакета Plotly.');if(!webglOk)bootstrapErrors.push('WebGL недоступен в Qt WebEngine. Проверьте драйвер видеокарты и аппаратное ускорение.');bridge.ready();pendingErrors.forEach(e=>bridge.renderError(e));pendingErrors=[];bootstrapErrors.forEach(e=>setTimeout(()=>reportError(e),0));});
</script></body></html>'''

if QWebEnginePage is not None:
    class _DiagnosticPage(QWebEnginePage):
        consoleError = pyqtSignal(str)
        def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
            level_name = getattr(level, "name", str(level))
            if "Error" in level_name or "error" in str(message).lower(): self.consoleError.emit(f"{message} (строка {lineNumber})")
            super().javaScriptConsoleMessage(level, message, lineNumber, sourceID)
else:
    _DiagnosticPage = None

class _Bridge(QObject):
    figureStarted = pyqtSignal(int,int); figureChunk = pyqtSignal(int,int,str); commandChanged = pyqtSignal(str)
    readyState = pyqtSignal(); cameraState = pyqtSignal(dict); autoRotateStoppedState = pyqtSignal()
    renderErrorState = pyqtSignal(str); renderedState = pyqtSignal(); chunkAckState = pyqtSignal(int,int)
    @pyqtSlot()
    def ready(self): self.readyState.emit()
    @pyqtSlot(int,int)
    def chunkReceived(self, transfer_id, index): self.chunkAckState.emit(int(transfer_id),int(index))
    @pyqtSlot(str)
    def cameraChanged(self, value):
        try: data = json.loads(value)
        except Exception: return
        if isinstance(data, dict): self.cameraState.emit(data)
    @pyqtSlot()
    def autoRotateStopped(self): self.autoRotateStoppedState.emit()
    @pyqtSlot(str)
    def renderError(self, message): self.renderErrorState.emit(str(message))
    @pyqtSlot()
    def rendered(self): self.renderedState.emit()

class _WorkerSignals(QObject):
    done = pyqtSignal(int, str, object, object, object, object); failed = pyqtSignal(int, str)

class _RenderWorker(QRunnable):
    def __init__(self, token, raw, th, axis, x, y, roi, settings, theme, cache_key, cached_volume=None, cached_input_stride=1, overlay_full_detector=False, cached_projection=None):
        super().__init__(); self.token=token; self.raw=raw; self.th=th; self.axis=axis; self.x=x; self.y=y; self.roi=roi; self.settings=settings; self.theme=theme; self.cache_key=cache_key; self.cached_volume=cached_volume; self.cached_input_stride=max(1,int(cached_input_stride)); self.overlay_full_detector=bool(overlay_full_detector); self.cached_projection=cached_projection; self.signals=_WorkerSignals()
    def run(self):
        try:
            input_stride=self.cached_input_stride
            if self.cached_volume is not None: rv=self.cached_volume
            else:
                cropped,crop_x,crop_y,_=crop_raw_to_roi(self.raw,self.axis,self.x,self.y,self.roi)
                input_stride,_,_=effective_extraction_stride(self.roi,crop_x,crop_y,len(self.th),requested=1)
                # Входной шаг обычно 1: уменьшаем массив только если оценка ROI превышает лимит памяти.
                raw_view=cropped[::input_stride,::input_stride,::input_stride]; th_view=np.asarray(self.th)[::input_stride]; x_view=crop_x[::input_stride]; y_view=crop_y[::input_stride]
                rv=extract_roi_volume(raw_view,th_view,self.roi,axis=self.axis,x_coords=x_view,y_coords=y_view,order=1)
            volume=np.asarray(rv.volume); settings=dict(self.settings); requested=max(2,int(settings.get("volume_stride",2))); quality=str(settings.get("quality","auto")); point_cap={"fast":25000,"balanced":50000,"auto":MAX_RENDER_POINTS,"full":MAX_RENDER_POINTS}.get(quality,MAX_RENDER_POINTS); stride=effective_volume_stride(volume.shape,requested,point_cap); settings["volume_stride"]=stride
            settings["dragmode"]="turntable" if settings.get("dragmode")=="turntable" else "orbit"
            full_projection=None
            if self.overlay_full_detector and self.cache_key[1] != "full":
                full_projection=self.cached_projection
                if full_projection is None:
                    full_projection=np.nansum(np.asarray(self.raw),axis=self.axis,dtype=np.float64).T.astype(np.float32)
            fig=build_volume_figure(rv,settings,self.theme,full_projection=full_projection,
                detector_x=self.x if full_projection is not None else None,detector_y=self.y if full_projection is not None else None,
                overlay_roi=self.roi if full_projection is not None else None)
            import plotly.io as pio
            payload=pio.to_json(fig,validate=False,pretty=False,remove_uids=False)
            volume_trace=next((tr for tr in fig.data if getattr(tr,"type","")=="volume"),None)
            meta={"shape":list(volume.shape),"coord_system":rv.coord_system,"roi":self.roi.label or "Full detector","input_stride":input_stride,"stride":stride,"points":int(np.asarray(volume_trace.value).size) if volume_trace is not None else 0,"payload_bytes":len(payload.encode("utf-8")),"overlay_full_detector":bool(full_projection is not None)}
            cached=rv if volume.size<=MAX_CACHE_VOXELS else None
            self.signals.done.emit(self.token,payload,meta,cached,self.cache_key,full_projection)
        except Exception as exc:self.signals.failed.emit(self.token,str(exc))

class _CameraDialog(QDialog):
    def __init__(self, camera, projection_type, parent=None):
        super().__init__(parent); self.setWindowTitle("Параметры камеры 3D"); self.setMinimumWidth(300); self._camera = camera or {}
        eye=self._camera.get("eye",{}); center=self._camera.get("center",{}); dx=float(eye.get("x",1.2))-float(center.get("x",0)); dy=float(eye.get("y",1.5))-float(center.get("y",0)); dz=float(eye.get("z",0.8))-float(center.get("z",0)); dist=max(1e-6,math.sqrt(dx*dx+dy*dy+dz*dz)); az=math.degrees(math.atan2(dy,dx)); el=math.degrees(math.asin(max(-1,min(1,dz/dist)))); roll=self._current_roll()
        form=QFormLayout(self); self.az=QDoubleSpinBox(); self.az.setRange(-360,360); self.az.setDecimals(2); self.az.setValue(az); self.az.setSuffix("°"); self.el=QDoubleSpinBox(); self.el.setRange(-89.9,89.9); self.el.setDecimals(2); self.el.setValue(el); self.el.setSuffix("°"); self.roll=QDoubleSpinBox(); self.roll.setRange(-180,180); self.roll.setDecimals(2); self.roll.setValue(roll); self.roll.setSuffix("°"); self.distance=QDoubleSpinBox(); self.distance.setRange(0.05,1000); self.distance.setDecimals(3); self.distance.setValue(dist); self.projection=QComboBox(); self.projection.addItem("Перспективная", "perspective"); self.projection.addItem("Ортографическая", "orthographic"); self.projection.setCurrentIndex(max(0, self.projection.findData(projection_type if projection_type in ("perspective","orthographic") else "perspective")))
        for w in (self.az,self.el,self.roll,self.distance): w.setKeyboardTracking(False)
        form.addRow("Азимут",self.az); form.addRow("Высота",self.el); form.addRow("Поворот",self.roll); form.addRow("Расстояние",self.distance); form.addRow("Проекция",self.projection); buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel); buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Применить"); buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена"); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); form.addRow(buttons); self.setStyleSheet("QAbstractSpinBox::up-button,QAbstractSpinBox::down-button{width:0px;height:0px;border:0px;}")
    @staticmethod
    def _norm(v):
        n=math.sqrt(sum(float(x)*float(x) for x in v)) or 1.0; return tuple(float(x)/n for x in v)
    @staticmethod
    def _cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    @staticmethod
    def _dot(a,b):return sum(a[i]*b[i] for i in range(3))
    def _basis(self,view):
        world=(0.0,0.0,1.0) if abs(view[2])<0.98 else (0.0,1.0,0.0); right=self._norm(self._cross(view,world)); up0=self._norm(self._cross(right,view)); return right,up0
    def _current_roll(self):
        eye=self._camera.get("eye",{});center=self._camera.get("center",{});up=self._camera.get("up",{"x":0,"y":0,"z":1});view=self._norm((float(center.get("x",0))-float(eye.get("x",1.2)),float(center.get("y",0))-float(eye.get("y",1.5)),float(center.get("z",0))-float(eye.get("z",0.8))));right,up0=self._basis(view);u=self._norm((float(up.get("x",0)),float(up.get("y",0)),float(up.get("z",1))));return math.degrees(math.atan2(self._dot(u,right),self._dot(u,up0)))
    def result_camera(self):
        c=self._camera; center=c.get("center",{"x":0,"y":0,"z":0}); a=math.radians(self.az.value()); e=math.radians(self.el.value()); d=self.distance.value(); ce=math.cos(e); eye={"x":float(center.get("x",0))+d*ce*math.cos(a),"y":float(center.get("y",0))+d*ce*math.sin(a),"z":float(center.get("z",0))+d*math.sin(e)};view=self._norm((float(center.get("x",0))-eye["x"],float(center.get("y",0))-eye["y"],float(center.get("z",0))-eye["z"]));right,up0=self._basis(view);r=math.radians(self.roll.value());up={"x":up0[0]*math.cos(r)+right[0]*math.sin(r),"y":up0[1]*math.cos(r)+right[1]*math.sin(r),"z":up0[2]*math.cos(r)+right[2]*math.sin(r)}
        return {"eye":eye,"center":center,"up":up}

class VolumePlotWidget(QWidget):
    settings_requested = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VolumePlotWidget")
        self.setMinimumSize(300, 260)
        self._theme = resolve_theme(self)
        self._settings = {}
        self._raw = self._th = self._x = self._y = None
        self._axis = 0
        self._scan_label = ""
        self._view_label = ""
        self._rois = []
        self._token = 0
        self._last_key = None
        self._last_payload = None
        self._last_meta = None
        self._camera = {}
        self._rotation_lock = False

        # Независимые состояния: расчёт массива и запуск браузерного 3D-движка.
        self._page_ready = False
        self._page_error = ""
        self._render_error = ""
        self._scene_rendered = False
        self._data_status = "Данные: ожидается загрузка HDF5"
        self._data_tooltip = "После загрузки HDF5 будет рассчитан 3D-объём."
        self._engine_detail = "ожидает первого показа панели"
        self._engine_tooltip = "Локальная HTML-страница будет запущена при открытии ROI Explorer в TRS Apps."

        self._worker_active = False
        self._pending_render = None
        self._dataset_generation = 0
        self._volume_cache_key = None
        self._volume_cache = None
        self._volume_cache_input_stride = 1
        self._projection_cache_key = None
        self._projection_cache = None

        self._transfer_id = 0
        self._transfer_payload = None
        self._transfer_chunk_size = 128 * 1024
        self._transfer_chunk_index = 0
        self._transfer_chunk_count = 0
        self._awaiting_webgl_render = False
        self._transfer_timer = QTimer(self)
        self._transfer_timer.setSingleShot(True)
        self._transfer_timer.setInterval(45000)
        self._transfer_timer.timeout.connect(self._on_transfer_timeout)

        self._render_pool = QThreadPool(self)
        self._render_pool.setMaxThreadCount(1)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(90)
        self._render_timer.timeout.connect(self._start_latest_render)
        self._render_elapsed = QElapsedTimer()
        self._render_status_timer = QTimer(self)
        self._render_status_timer.setInterval(1000)
        self._render_status_timer.timeout.connect(self._update_render_elapsed_status)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)
        self.toolbar = VolumeToolsWidget(self)
        root.addWidget(self.toolbar)
        self.status = QLabel(self._data_status, self)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(22)
        root.addWidget(self.status)

        self.web = None
        self.bridge = None
        self._page_load_attempts = 0
        self._html_base_url = None
        self._html_text = ""
        self._page_started = False
        self._bootstrap_retry_pending = False
        self._last_bootstrap_state = {}
        self._bootstrap_timer = QTimer(self)
        self._bootstrap_timer.setSingleShot(True)
        self._bootstrap_timer.setInterval(6500)
        self._bootstrap_timer.timeout.connect(self._bootstrap_watchdog)

        app = QApplication.instance()
        startup_error = app.property("rocMcsWebEngineBootstrapError") if app is not None else ""
        if startup_error:
            # Не пытаемся повторить импорт после QApplication: это не лечит
            # порядок инициализации и даёт вторичную, менее точную ошибку.
            self._show_web_error(
                "Ранний bootstrap Qt WebEngine в точке входа завершился ошибкой: "
                + str(startup_error)
            )
        elif QWebEngineView is not None:
            try:
                self._init_webengine(root)
            except Exception as exc:
                self._show_web_error(exc)
        else:
            self._show_web_error(_WEBENGINE_ERROR or RuntimeError(
                "Qt WebEngine отсутствует в текущем процессе; проверьте startup log."
            ))

        self.toolbar.roi_source_changed.connect(lambda *_: self.refresh(force=True))
        self.toolbar.roi_overlay_changed.connect(self._on_roi_overlay_changed)
        self.toolbar.reset_view_requested.connect(lambda: self._command("fit"))
        self.toolbar.camera_requested.connect(self._open_camera_dialog)
        self.toolbar.autorotate_toggled.connect(lambda enabled: self._command("rotate", enabled=enabled))
        self.toolbar.align_xy_requested.connect(lambda: self._command("align", axis="xy"))
        self.toolbar.view_axis_requested.connect(lambda axis: self._command("align", axis=axis))
        self.toolbar.rotation_lock_toggled.connect(self._on_rotation_lock)
        self.toolbar.colormap_changed.connect(self._on_colormap_changed)
        self.toolbar.contrast_changed.connect(self._on_contrast_changed)
        self.toolbar.contrast_auto_requested.connect(self._on_contrast_auto)
        self.toolbar.settings_requested.connect(self.settings_requested.emit)
        self.toolbar.info_requested.connect(self._show_info)
        self._update_status()

    def _init_webengine(self, root):
        import plotly
        from pathlib import Path
        from importlib.metadata import PackageNotFoundError, version as package_version
        match = re.match(r"^(\d+)\.(\d+)", str(plotly.__version__))
        version = tuple(map(int, match.groups())) if match else (0, 0)
        if version < (7, 1) or version >= (8, 0): raise RuntimeError(f"Требуется Plotly >= 7.1, < 8 (по pyproject.toml), установлена версия {plotly.__version__}. Обновите зависимости именно выбранного Python: python -m pip install -e .")
        try:
            webengine_text = package_version("PyQt6-WebEngine")
        except PackageNotFoundError as exc:
            raise RuntimeError("Не найден пакет PyQt6-WebEngine. Установите зависимости проекта: python -m pip install -e .") from exc
        webengine_match = re.match(r"^(\d+)\.(\d+)", webengine_text)
        webengine_version = tuple(map(int, webengine_match.groups())) if webengine_match else (0, 0)
        if webengine_version < (6, 11) or webengine_version >= (7, 0): raise RuntimeError(f"Требуется PyQt6-WebEngine >= 6.11, < 7 (по pyproject.toml), установлена версия {webengine_text}. Обновите зависимости: python -m pip install -e .")
        js=Path(plotly.__file__).resolve().parent/"package_data"/"plotly.min.js"
        if not js.is_file() or js.stat().st_size < 100_000: raise RuntimeError(f"Не найден или повреждён локальный plotly.min.js: {js}")
        from PyQt6.QtCore import QUrl
        self.web=QWebEngineView(self); self.web.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu); self.web.setPage(_DiagnosticPage(self.web)); self.web.page().setBackgroundColor(QColor("#111316" if self._theme == "dark" else "#ffffff")); self.web.setStyleSheet(f"background:{'#111316' if self._theme == 'dark' else '#ffffff'};"); settings=self.web.settings()
        for name in ("JavascriptEnabled", "LocalContentCanAccessFileUrls", "WebGLEnabled"):
            attr=getattr(QWebEngineSettings.WebAttribute,name,None)
            if attr is not None: settings.setAttribute(attr,True)
        remote=getattr(QWebEngineSettings.WebAttribute,"LocalContentCanAccessRemoteUrls",None)
        if remote is not None: settings.setAttribute(remote,False)
        self.bridge=_Bridge(self); channel=QWebChannel(self.web.page()); channel.registerObject("bridge",self.bridge); self.web.page().setWebChannel(channel); self._channel=channel
        self.bridge.readyState.connect(self._on_page_ready); self.bridge.chunkAckState.connect(self._on_transfer_chunk_ack); self.bridge.cameraState.connect(self._on_camera_state); self.bridge.autoRotateStoppedState.connect(self._sync_autorotate_off); self.bridge.renderErrorState.connect(self._on_js_error); self.bridge.renderedState.connect(self._on_figure_rendered)
        self.web.page().consoleError.connect(self._on_js_error); self.web.loadFinished.connect(self._on_page_loaded)
        terminated=getattr(self.web.page(),"renderProcessTerminated",None)
        if terminated is not None:terminated.connect(lambda status,code:self._on_js_error(f"Процесс Qt WebEngine завершился: {status}, код {code}."))
        self._html_base_url=QUrl.fromLocalFile(str(js.parent)+"/"); self._html_text=_HTML.replace("__BG__", "#111316" if self._theme == "dark" else "#ffffff")
        root.insertWidget(1,self.web,1); root.setStretch(1,1); root.setStretch(2,0); self.status.setMinimumHeight(22); self.web.show()
        # Не загружаем Chromium в конструкторе скрытого MDI-подокна.
        # Страница запускается после первого фактического показа панели.

    def event(self, event):
        result = super().event(event)
        if event.type() == QEvent.Type.ShowToParent and self.web is not None and not self._page_started:
            QTimer.singleShot(0, self._load_web_page)
        return result

    def showEvent(self, event):
        super().showEvent(event)
        if self.web is None:
            return
        if not self._page_started:
            QTimer.singleShot(0, self._load_web_page)
        QTimer.singleShot(220, self._recover_after_show)

    def _load_web_page(self):
        if self.web is None or self._page_ready:
            return
        if self._page_started and not self._bootstrap_retry_pending:
            return
        self._page_started = True
        self._page_load_attempts += 1
        self._bootstrap_retry_pending = False
        self._page_error = ""
        self._last_bootstrap_state = {}
        self._engine_detail = f"загрузка HTML/JavaScript (попытка {self._page_load_attempts})"
        self._engine_tooltip = "Загружаются локальный Plotly.js и qwebchannel.js; затем проверяется Qt WebChannel handshake."
        self._update_status()
        self.web.setHtml(self._html_text, self._html_base_url)
        self._bootstrap_timer.start()

    def _recover_after_show(self):
        if self.web is None or not self.isVisible() or not self.web.isVisible():return
        if not self._page_started:self._load_web_page();return
        if not self._page_ready:
            self._check_web_bootstrap()
            if not self._bootstrap_timer.isActive():self._bootstrap_timer.start()
            return
        self.web.update();self.web.page().runJavaScript("if(typeof recoverAfterShow==='function')recoverAfterShow();")

    def _check_web_bootstrap(self):
        if self.web is None or self._page_ready:
            return
        probe = "JSON.stringify({qt:typeof qt,transport:typeof qt!=='undefined'&&!!qt.webChannelTransport,channel:typeof QWebChannel,plotly:typeof Plotly,webgl:window.__roi3dWebglOk===undefined?null:!!window.__roi3dWebglOk,ready:!!window.__roi3dBridgeReady})"
        self.web.page().runJavaScript(probe, self._on_bootstrap_probe)

    def _bootstrap_watchdog(self):
        if self._page_ready or self.web is None:
            return
        # Скрытая MDI-страница не считается зависшей: проверка продолжится после показа.
        if not self.isVisible() or not self.web.isVisible():
            return
        self._check_web_bootstrap()
        if self._page_load_attempts < 2 and not self._bootstrap_retry_pending:
            self._bootstrap_retry_pending = True
            self._engine_detail = "handshake не подтверждён; выполняется одна повторная загрузка"
            self._engine_tooltip = "За 6,5 с не завершился JavaScript/Qt handshake. Проверяется состояние страницы; выполняется одна повторная попытка."
            self._update_status()
            logging.getLogger(__name__).warning(
                "Qt WebEngine не подтвердил handshake; повторная загрузка страницы (попытка %s)",
                self._page_load_attempts,
            )
            QTimer.singleShot(350, self._load_web_page)
        elif self._page_load_attempts >= 2:
            QTimer.singleShot(1400, self._finalize_bootstrap_failure)

    def _on_bootstrap_probe(self, result):
        if self._page_ready:
            return
        try:
            state = json.loads(result) if isinstance(result, str) else (result if isinstance(result, dict) else {})
        except Exception:
            state = {"probe": str(result)}
        if state:
            self._last_bootstrap_state = state
            logging.getLogger(__name__).debug("Qt WebEngine bootstrap probe: %s", state)
        if state.get("ready") and not self._page_ready:
            logging.getLogger(__name__).debug("JS bridge готов; ожидается Qt ready-сигнал.")

    def _finalize_bootstrap_failure(self):
        if self._page_ready or self._page_error:
            return
        state = self._last_bootstrap_state or {"probe": "JavaScript не вернул ответ"}
        if state.get("probe"):
            cause = "JavaScript WebEngine не отвечает на диагностический запрос; проверьте QtWebEngineProcess и журнал Qt."
        elif state.get("qt") != "object":
            cause = "Qt WebChannel bootstrap не создал объект qt; qwebchannel.js не был подключён к странице."
        elif not state.get("transport"):
            cause = "Нет qt.webChannelTransport; WebEngine не поднял транспорт Qt WebChannel."
        elif state.get("channel") != "function":
            cause = "Не загрузился qrc:///qtwebchannel/qwebchannel.js (QWebChannel отсутствует)."
        elif state.get("plotly") not in ("object", "function"):
            cause = "Не загрузился локальный plotly.min.js."
        elif not state.get("ready"):
            cause = "JS доступен, но Python↔JavaScript bridge handshake не завершился."
        elif not state.get("webgl", True):
            cause = "Qt WebChannel готов, но WebGL-контекст не создан; вероятна проблема графического драйвера или аппаратного ускорения."
        else:
            cause = "JavaScript сообщает готовность, но Python не получил ready-сигнал Qt WebChannel."
        self._on_js_error(cause + " Диагностика: " + json.dumps(state, ensure_ascii=False))

    def _on_page_loaded(self, ok):
        if not ok:
            message = f"Qt WebEngine loadFinished(False), попытка {self._page_load_attempts}."
            logging.getLogger(__name__).error(message)
            if self._page_load_attempts >= 2:
                self._on_js_error(
                    message + " HTML/JavaScript-страница не загрузилась после одной повторной попытки. "
                    "Проверьте QtWebEngineProcess.exe, qtwebengine_resources.pak, icudtl.dat и локальный plotly.min.js."
                )
            elif not self._bootstrap_retry_pending:
                self._bootstrap_retry_pending = True
                self._engine_detail = "первая загрузка HTML завершилась ошибкой; повторная попытка"
                self._engine_tooltip = message + " Повтор через 350 мс."
                self._update_status()
                QTimer.singleShot(350, self._load_web_page)
            return
        if self._page_ready:
            return
        self._engine_detail = "HTML загружен; проверяются Qt WebChannel, Plotly.js и WebGL"
        self._engine_tooltip = "loadFinished(True). Следующий этап — JavaScript probe и ready-handshake Python↔JavaScript."
        self._update_status()
        QTimer.singleShot(350, self._check_web_bootstrap)

    def _update_status(self, data_status=None, data_tooltip=None, engine_status=None, engine_tooltip=None):
        """Показывает отдельно расчёт данных и отдельный конвейер Qt WebEngine/WebGL."""
        if data_status is not None:
            self._data_status = " ".join(str(data_status).split())
        if data_tooltip is not None:
            self._data_tooltip = str(data_tooltip)
        if engine_status is not None:
            self._engine_detail = " ".join(str(engine_status).split())
        if engine_tooltip is not None:
            self._engine_tooltip = str(engine_tooltip)

        if self._page_error:
            engine_line = "3D-движок: ОШИБКА — " + self._page_error[:180]
        elif self._page_ready:
            engine_line = "3D-движок: Qt WebChannel готов"
            if self._scene_rendered:
                engine_line += "; WebGL подтвердил отрисовку"
        elif self.web is None:
            engine_line = "3D-движок: недоступен"
        else:
            engine_line = "3D-движок: " + self._engine_detail

        self.status.setText(f"{self._data_status}  ·  {engine_line}")
        details = []
        if self._data_tooltip:
            details.append("Данные/расчёт:\n" + self._data_tooltip)
        if self._engine_tooltip:
            details.append("Qt WebEngine/WebGL:\n" + self._engine_tooltip)
        if self._page_error:
            details.append("Точная причина сбоя 3D-движка:\n" + self._page_error)
        if self._render_error:
            details.append("Точная причина сбоя расчёта объёма:\n" + self._render_error)
        if self._last_bootstrap_state:
            details.append("Диагностика JavaScript handshake:\n" + json.dumps(self._last_bootstrap_state, ensure_ascii=False))
        if self._page_ready and not self._scene_rendered:
            details.append("Qt WebChannel подтверждён. Далее ожидаются расчёт, передача данных и callback Plotly/WebGL.")
        self.status.setToolTip("\n\n".join(details))
        self.status.show()

    def _update_render_elapsed_status(self):
        if not self._worker_active or not self._render_elapsed.isValid():
            self._render_status_timer.stop()
            return
        elapsed = int(self._render_elapsed.elapsed() / 1000)
        if elapsed >= 5:
            self._update_status(
                f"Расчёт 3D-объёма выполняется · {elapsed} с",
                "Рабочий поток активен. Алгоритм не предоставляет процент прогресса; "
                "поэтому отображается фактическое время без выдуманного процента.",
            )

    def _on_js_error(self, message):
        text = " ".join(str(message).split())
        if not text:
            return
        self._transfer_timer.stop()
        self._bootstrap_timer.stop()
        self._transfer_payload = None
        self._awaiting_webgl_render = False
        if self.web and self._page_ready:
            self.web.page().runJavaScript("if(typeof abortTransfer==='function')abortTransfer();")
        self._page_error = text[:1500]
        self._engine_detail = "ошибка на этапе запуска/передачи/отрисовки"
        self._engine_tooltip = "Смотрите полное сообщение причины сбоя в подсказке статуса."
        self._scene_rendered = False
        logging.getLogger(__name__).error("Ошибка 3D/WebEngine: %s", self._page_error)
        self._update_status()

    def _on_figure_rendered(self):
        self._transfer_timer.stop()
        self._awaiting_webgl_render = False
        self._scene_rendered = True
        self._page_error = ""
        self._render_error = ""
        self._engine_detail = "Plotly/WebGL успешно отрисовал 3D-сцену"
        self._engine_tooltip = "Получен callback rendered() от JavaScript после Plotly.react()."
        if self._last_meta:
            meta = self._last_meta
            title = f"3D-объём отрисован · {meta.get('roi', 'ROI')} · {tuple(meta.get('shape', ())) }"
            details = (
                f"Система координат: {meta.get('coord_system')}; сетка={tuple(meta.get('shape', ())) }; "
                f"точек={meta.get('points', 0):,}; JSON={meta.get('payload_bytes', 0) / 1048576:.2f} MiB."
            )
        else:
            title = "3D-объём отрисован"
            details = "Plotly/WebGL подтвердил завершение отрисовки."
        self._update_status(title, details + " Колесо — масштаб, ЛКМ — вращение, СКМ — панорамирование.")

    def _show_web_error(self, exc):
        message = " ".join(str(exc or "неизвестная ошибка Qt WebEngine").split())
        self._page_error = message[:1500]
        self._engine_detail = "ошибка инициализации PyQt6-WebEngine"
        self._engine_tooltip = "Инициализация WebEngine завершилась исключением: " + self._page_error
        logging.getLogger(__name__).error("Не удалось инициализировать Qt WebEngine: %s", self._page_error)
        self._update_status()

    def _on_page_ready(self):
        self._page_ready = True
        self._page_error = ""
        self._bootstrap_timer.stop()
        self._engine_detail = "Qt WebChannel готов; Plotly.js и JavaScript bridge подключены"
        self._engine_tooltip = "Python↔JavaScript handshake подтверждён. Далее возможна передача JSON-сцены в Plotly/WebGL."
        self._command("theme", theme=self._theme)
        self._command("speed", value=float(self._settings.get("rotation_speed", 14.0)))
        self._on_rotation_lock(self._rotation_lock)
        if self._last_payload:
            self._send_figure(self._last_payload)
        elif self._raw is None:
            self._update_status("Данные: ожидается загрузка HDF5")
        else:
            self._update_status("Данные: ожидается запуск расчёта объёма")

    def _on_camera_state(self,camera): self._camera=camera
    def _sync_autorotate_off(self):
        self.toolbar.rotate_btn.blockSignals(True); self.toolbar.rotate_btn.setChecked(False); self.toolbar.rotate_btn.blockSignals(False)
    def _command(self, kind, **payload):
        if not self.bridge or not self._page_ready:return
        self.bridge.commandChanged.emit(json.dumps({"type":kind,**payload},separators=(",",":")))

    def set_dataset(self, raw, th, axis, x_coords, y_coords, saved_rois, *, scan_label="", view_label=""):
        self._dataset_generation+=1; self._volume_cache_key=None; self._volume_cache=None; self._volume_cache_input_stride=1; self._projection_cache_key=None; self._projection_cache=None; self._raw=np.asarray(raw); self._th=np.asarray(th,dtype=float); self._axis=int(axis); self._x=np.asarray(x_coords,dtype=float); self._y=np.asarray(y_coords,dtype=float); self._scan_label=str(scan_label); self._view_label=str(view_label); self._rois=[r for r in (saved_rois or []) if str(r.roi.type).lower() in ("rect","rectangle","rotated_rectangle") and r.roi.normalized().is_valid]
        choices=[(int(r.id),r.roi.label or f"ROI {r.id}",r.roi.color) for r in self._rois]; self.toolbar.set_roi_choices(choices,self.toolbar.current_source()); self.refresh()
    def clear_dataset(self, message="Откройте HDF5, чтобы построить 3D-объём"):
        self._transfer_timer.stop()
        self._transfer_payload = None
        self._awaiting_webgl_render = False
        self._render_status_timer.stop()
        self._token += 1
        self._dataset_generation += 1
        self._volume_cache_key = None
        self._volume_cache = None
        self._volume_cache_input_stride = 1
        self._projection_cache_key = None
        self._projection_cache = None
        self._render_timer.stop()
        self._pending_render = None
        self._raw = self._th = self._x = self._y = None
        self._rois = []
        self._last_key = None
        self._last_payload = None
        self._last_meta = None
        self._render_error = ""
        self._scene_rendered = False
        # Ошибка/готовность WebEngine относится к самому движку, а не к HDF5;
        # очистка набора данных не должна стирать эту диагностику.
        self.toolbar.set_roi_choices([], "full")
        if self.web and self._page_ready:
            self.web.page().runJavaScript("if(typeof purgePlot==='function')purgePlot();")
        self._update_status(message, "Набор данных очищен. Статус 3D-движка отображается независимо.")

    def _source_roi(self):
        key=self.toolbar.current_source()
        if key != "full":
            saved=next((r for r in self._rois if int(r.id)==int(key)),None)
            if saved is not None:return saved.roi.normalized()
            self.toolbar.set_roi_choices([(int(r.id),r.roi.label or f"ROI {r.id}",r.roi.color) for r in self._rois],"full")
        dx=abs(float(np.median(np.diff(self._x)))) if len(self._x)>1 else 1.0; dy=abs(float(np.median(np.diff(self._y)))) if len(self._y)>1 else 1.0
        return ROI(cx=float((self._x[0]+self._x[-1])/2),cy=float((self._y[0]+self._y[-1])/2),width=float(abs(self._x[-1]-self._x[0])+dx),height=float(abs(self._y[-1]-self._y[0])+dy),angle=0,label="Весь детектор",type="rect")

    def refresh(self, force=False):
        if self._raw is None or self._th is None or self._x is None or self._y is None:
            return
        roi = self._source_roi()
        settings = dict(self._settings)
        settings["volume_stride"] = max(2, int(settings.get("volume_stride", 2)))
        settings["dragmode"] = "turntable" if self._rotation_lock else "orbit"
        settings["quality"] = str(settings.get("quality", "auto"))
        if self._camera:
            settings["camera"] = self._camera
        overlay = bool(self.toolbar.show_roi_on_full_detector()) and self.toolbar.current_source() != "full"
        key = (
            self._scan_label, self._view_label, self.toolbar.current_source(), overlay,
            round(roi.cx, 8), round(roi.cy, 8), round(roi.width, 8),
            round(roi.height, 8), round(roi.angle, 8), self._theme,
            json.dumps(settings, sort_keys=True, default=str),
        )
        if not force and key == self._last_key:
            return
        self._last_key = key
        self._token += 1
        token = self._token
        self._scene_rendered = False
        self._render_error = ""
        cache_key = (
            self._dataset_generation, self.toolbar.current_source(), round(roi.cx, 8),
            round(roi.cy, 8), round(roi.width, 8), round(roi.height, 8), round(roi.angle, 8),
        )
        self._pending_render = (
            token, self._raw, self._th, self._axis, self._x, self._y,
            roi, settings, self._theme, cache_key, overlay,
        )
        self._update_status(
            "Расчёт 3D-объёма: задача поставлена в очередь",
            "Задача будет передана однопоточному QThreadPool после debounce 90 мс. "
            "Если worker активен, промежуточные изменения объединяются.",
        )
        self._render_timer.start()

    def _start_latest_render(self):
        if self._worker_active or self._pending_render is None:
            return
        token, raw, th, axis, x, y, roi, settings, theme, cache_key, overlay = self._pending_render
        self._pending_render = None
        self._worker_active = True
        self._render_elapsed.start()
        self._render_status_timer.start()
        self._update_status(
            "Расчёт 3D-объёма выполняется в фоновом потоке",
            "Worker выполняет извлечение ROI-объёма, построение Plotly figure и JSON-сериализацию.",
        )
        cached = self._volume_cache if self._volume_cache_key == cache_key else None
        projection = self._projection_cache if self._projection_cache_key == self._dataset_generation else None
        worker = _RenderWorker(
            token, raw, th, axis, x, y, roi, settings, theme, cache_key,
            cached, self._volume_cache_input_stride, overlay, projection,
        )
        worker.signals.done.connect(self._render_ready)
        worker.signals.failed.connect(self._render_failed)
        self._worker = worker
        self._render_pool.start(worker)

    def _worker_finished(self):
        self._worker_active = False
        self._worker = None
        self._render_status_timer.stop()
        if self._pending_render is not None:
            QTimer.singleShot(0, self._start_latest_render)

    def _render_ready(self, token, payload, meta, rv, cache_key, full_projection):
        if rv is not None and isinstance(cache_key, tuple) and cache_key and cache_key[0] == self._dataset_generation:
            self._volume_cache_key = cache_key
            self._volume_cache = rv
            self._volume_cache_input_stride = int(meta.get("input_stride", 1))
        if full_projection is not None and isinstance(cache_key, tuple) and cache_key and cache_key[0] == self._dataset_generation:
            self._projection_cache_key = self._dataset_generation
            self._projection_cache = full_projection
        self._worker_finished()
        if token != self._token:
            return
        self._last_payload = payload
        self._last_meta = meta
        self._render_error = ""
        meta_text = (
            f"ROI={meta.get('roi')}; сетка={tuple(meta['shape'])}; входной шаг ×{meta.get('input_stride', 1)}; "
            f"шаг рендера ×{meta.get('stride', 1)}; точек={meta.get('points', 0):,}; "
            f"JSON={meta.get('payload_bytes', 0) / 1048576:.2f} MiB."
        )
        if self._page_error:
            self._update_status("Объём рассчитан; передача невозможна из-за ошибки 3D-движка", meta_text)
        elif self.bridge and self._page_ready:
            self._send_figure(payload)
        else:
            self._update_status(
                "Объём рассчитан; ожидается готовность 3D-движка",
                meta_text + " Расчёт завершён. Передача в WebGL ещё не начиналась.",
            )

    def _send_figure(self, payload):
        if not self.bridge or not self._page_ready:
            self._update_status("Объём рассчитан; WebChannel ещё не готов", self._data_tooltip)
            return
        self._transfer_timer.stop()
        self._awaiting_webgl_render = True
        self._transfer_id += 1
        self._transfer_payload = str(payload)
        self._transfer_chunk_index = 0
        self._transfer_chunk_count = max(1, math.ceil(len(self._transfer_payload) / self._transfer_chunk_size))
        self._transfer_timer.start()
        self._update_status(
            f"Передача 3D-сцены: 0/{self._transfer_chunk_count} блоков",
            f"Размер JSON={len(self._transfer_payload) / 1048576:.2f} MiB; блок={self._transfer_chunk_size // 1024} KiB; "
            "ожидаются подтверждения каждого блока от JavaScript bridge.",
        )
        self.bridge.figureStarted.emit(self._transfer_id, self._transfer_chunk_count)
        self._send_next_figure_chunk()

    def _send_next_figure_chunk(self):
        if self._transfer_payload is None or not self.bridge or not self._page_ready:
            return
        index = self._transfer_chunk_index
        start = index * self._transfer_chunk_size
        chunk = self._transfer_payload[start:start + self._transfer_chunk_size]
        self._update_status(
            f"Передача 3D-сцены: блок {index + 1}/{self._transfer_chunk_count}",
            f"Подтверждено блоков: {index}/{self._transfer_chunk_count}; размер текущего блока {len(chunk) / 1024:.1f} KiB.",
        )
        self.bridge.figureChunk.emit(self._transfer_id, index, chunk)

    def _on_transfer_chunk_ack(self, transfer_id, index):
        if self._transfer_payload is None:
            return
        if int(transfer_id) != self._transfer_id or int(index) != self._transfer_chunk_index:
            return
        self._transfer_chunk_index += 1
        if self._transfer_chunk_index >= self._transfer_chunk_count:
            self._transfer_payload = None
            self._transfer_timer.start(45000)
            self._update_status(
                "Данные переданы; Plotly/WebGL выполняет отрисовку",
                f"Все {self._transfer_chunk_count} блоков подтверждены Qt WebChannel. "
                "Теперь ожидается callback rendered() от Plotly после WebGL-отрисовки (тайм-аут 45 с).",
            )
            return
        self._send_next_figure_chunk()

    def _on_transfer_timeout(self):
        if self._transfer_payload is not None:
            sent = self._transfer_chunk_index
            total = self._transfer_chunk_count
            self._transfer_payload = None
            self._on_js_error(
                f"Тайм-аут передачи Qt WebChannel: за 45 с подтверждено {sent} из {total} блоков. "
                "Проблема находится на этапе передачи данных, а не расчёта объёма."
            )
        elif self._awaiting_webgl_render:
            self._on_js_error(
                "Тайм-аут Plotly/WebGL: все блоки данных переданы, но callback завершения отрисовки "
                "не пришёл за 45 с. Проверьте WebGL, графический драйвер, видеопамять и журнал Qt WebEngine."
            )

    def _on_roi_overlay_changed(self, enabled):
        self.refresh(force=True)

    def _on_colormap_changed(self, name):
        self._settings["volume_colormap"] = str(name)
        self.refresh(force=True)

    def _on_contrast_changed(self, low, high):
        low=max(0.0,min(99.5,float(low))); high=max(low+0.5,min(100.0,float(high)))
        self._settings["isomin_fraction"] = low / 100.0
        self._settings["isomax_fraction"] = high / 100.0
        self.refresh(force=True)

    def _on_contrast_auto(self):
        # Auto intentionally reproduces the established default rendering thresholds.
        self._settings["isomin_fraction"] = 0.16
        self._settings["isomax_fraction"] = 0.95
        self.toolbar.set_contrast(16.0, 95.0)
        self.refresh(force=True)

    def _render_failed(self, token, message):
        self._worker_finished()
        if token != self._token:
            return
        self._last_key = None
        self._render_error = " ".join(str(message).split())[:1500]
        logging.getLogger(__name__).error("Не удалось построить 3D-объём: %s", self._render_error)
        self._update_status("Ошибка расчёта 3D-объёма", "Расчёт завершился исключением: " + self._render_error)

    def apply_theme(self, theme=None):
        self._theme=apply_widget_palette(self,theme); self.toolbar.apply_theme(self._theme); c=theme_colors(self._theme); self.status.setStyleSheet(f"color:{c['muted']}; background:{c['canvas']}; padding:5px 8px;")
        if self.web: self.web.page().setBackgroundColor(QColor(c["canvas"])); self.web.setStyleSheet(f"background:{c['canvas']};")
        self._command("theme",theme=self._theme); self.refresh(force=True)
    def apply_display_settings(self, settings=None):
        if settings is not None:self._settings=dict(settings)
        self._rotation_lock=str(self._settings.get("dragmode","orbit"))=="turntable"; self.toolbar.set_rotation_lock(self._rotation_lock)
        self.toolbar.set_colormap(str(self._settings.get("volume_colormap","magma")))
        self.toolbar.set_contrast(100.0*float(self._settings.get("isomin_fraction",0.16)),100.0*float(self._settings.get("isomax_fraction",0.95)))
        self._command("speed",value=float(self._settings.get("rotation_speed",14.0))); self._on_rotation_lock(self._rotation_lock); self.refresh(force=True)

    def _on_rotation_lock(self, enabled):
        self._rotation_lock=bool(enabled)
        if hasattr(self,"toolbar"):self.toolbar.set_rotation_lock(self._rotation_lock)
        self._settings["dragmode"]="turntable" if self._rotation_lock else "orbit"
        self._command("rotationLock",enabled=self._rotation_lock)

    def _open_camera_dialog(self):
        dialog=_CameraDialog(self._camera,str(self._settings.get("projection_type","perspective")),self); apply_popup_theme(dialog,resolve_theme(self))
        if dialog.exec()==QDialog.DialogCode.Accepted:
            self._settings["projection_type"]=str(dialog.projection.currentData() or "perspective"); self._camera=dialog.result_camera(); self._command("camera",camera=self._camera,projection=self._settings["projection_type"])
    def _show_info(self):
        box=QMessageBox(self); box.setWindowTitle("Управление 3D"); box.setIcon(QMessageBox.Icon.Information); box.setText("ЛКМ + перетаскивание — вращение объёма.\nСКМ + перетаскивание — панорамирование.\nКолесо мыши — приближение/отдаление.\nКнопка ориентации — выровнять плоскость XY; стрелка рядом — вид вдоль X/Y/Z и режим сохранения вертикали.\n«Вписать» возвращает весь объём в окно, сохраняя ракурс.\nАвтовращение автоматически останавливается при ручном управлении."); box.setStandardButtons(QMessageBox.StandardButton.Ok); box.button(QMessageBox.StandardButton.Ok).setText("Понятно"); apply_popup_theme(box,resolve_theme(self)); box.exec()
