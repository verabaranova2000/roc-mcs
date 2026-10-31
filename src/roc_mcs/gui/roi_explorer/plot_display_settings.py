"""Настройки отображения научных экранов ROI Explorer."""
from __future__ import annotations
from copy import deepcopy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QCheckBox,QComboBox,QDialog,QDialogButtonBox,QDoubleSpinBox,QFormLayout,QFontComboBox,QFrame,QLabel,QPushButton,QSpinBox,QTabWidget,QVBoxLayout,QWidget,QHBoxLayout,QScrollArea
from roc_mcs.gui.components.dialog_chrome import install_dialog_titlebar
from roc_mcs.gui.app_settings import load_settings,save_settings
from roc_mcs.gui.roi_explorer.volume_rendering import PROJECTION_SCALES,VOLUME_SCALES
from roc_mcs.gui.roi_explorer.colormap_catalog import NAPARI_STYLE_COLORMAPS
from roc_mcs.gui.roi_explorer.theme import apply_popup_theme,resolve_theme

DEFAULT_PLOT_DISPLAY_SETTINGS={
    "map":{"font_family":"Segoe UI","tick_size":8,"label_size":9,"colorbar_tick_size":8,"show_colorbar":True,"show_scalebar":False,"scalebar_um":2000.0,"pixel_size_um":55.0,"scalebar_font_size":9},
    "curve":{"font_family":"Segoe UI","tick_size":8,"label_size":9,"show_grid":True},
    "volume":{"font_family":"Times New Roman","tick_size":14,"label_size":16,"axis_tick_length":3,"show_grid":True,"show_projection":True,"show_colorbar":True,"show_legend":False,"volume_colormap":"magma","volume_colorscale_light":"Blues","volume_colorscale_dark":"Magma","projection_colorscale":"Greys","volume_opacity":0.10,"projection_opacity":0.75,"isomin_fraction":0.16,"isomax_fraction":0.95,"surface_count":25,"smooth_sigma":0.5,"volume_stride":2,"quality":"auto","dragmode":"orbit","projection_type":"perspective","rotation_speed":14.0},
    "workspace":{"map_geometry":None,"curve_geometry":None,"volume_geometry":None,"orientation":"focus","layout_choice":"focus","layout_ratios":{"focus_left":0.52,"focus_top":0.50,"row":[1/3,1/3,1/3],"column":[0.30,0.30,0.40],"map3d_left":0.50,"map3d_top":0.56},"viewport_size":None},
}
_SETTINGS_KEY="roi_explorer_display"

def _safe_int(value,default,low,high):
    try:return max(low,min(high,int(value)))
    except (TypeError,ValueError,OverflowError):return default

def _safe_float(value,default,low,high):
    try:
        n=float(value); return max(low,min(high,n)) if n==n and abs(n)!=float("inf") else default
    except (TypeError,ValueError,OverflowError):return default

def _safe_bool(value,default):
    if isinstance(value,bool):return value
    if isinstance(value,(int,float)) and value in (0,1):return bool(value)
    if isinstance(value,str):
        s=value.strip().lower()
        if s in ("true","1","yes","on"):return True
        if s in ("false","0","no","off"):return False
    return default

def load_plot_display_settings()->dict:
    raw=load_settings().get(_SETTINGS_KEY,{}); result=deepcopy(DEFAULT_PLOT_DISPLAY_SETTINGS)
    if isinstance(raw,dict):
        for section in result:
            values=raw.get(section,{})
            if isinstance(values,dict):result[section].update({k:v for k,v in values.items() if k in result[section]})
    m,c,v,w=result["map"],result["curve"],result["volume"],result["workspace"]
    for key,values in (("map",m),("curve",c),("volume",v)):
        defaults=DEFAULT_PLOT_DISPLAY_SETTINGS[key]
        if not isinstance(values.get("font_family"),str) or not values["font_family"].strip():values["font_family"]="Segoe UI"
        values["tick_size"]=_safe_int(values.get("tick_size"),defaults["tick_size"],6,16);values["label_size"]=_safe_int(values.get("label_size"),defaults["label_size"],6,18)
    m["colorbar_tick_size"]=m["tick_size"];m["scalebar_font_size"]=_safe_int(m.get("scalebar_font_size"),9,6,16);m["scalebar_um"]=_safe_float(m.get("scalebar_um"),2000,10,100000);m["pixel_size_um"]=_safe_float(m.get("pixel_size_um"),55,0.001,100000);m["show_colorbar"]=_safe_bool(m.get("show_colorbar"),True);m["show_scalebar"]=_safe_bool(m.get("show_scalebar"),False);c["show_grid"]=_safe_bool(c.get("show_grid"),True)
    for key,default in (("show_grid",True),("show_projection",True),("show_colorbar",False),("show_legend",False)):v[key]=_safe_bool(v.get(key),default)
    # Однократная миграция прежних значений по умолчанию; пользовательские настройки сохраняются.
    old_defaults=(abs(_safe_float(v.get("isomin_fraction"),0.12,0,0.99)-0.12)<1e-9 and abs(_safe_float(v.get("isomax_fraction"),1.0,0.01,1)-1.0)<1e-9 and abs(_safe_float(v.get("volume_opacity"),0.15,0.01,1)-0.15)<1e-9 and abs(_safe_float(v.get("projection_opacity"),0.72,0,1)-0.72)<1e-9)
    if isinstance(raw,dict) and isinstance(raw.get("volume"),dict) and old_defaults:
        v["isomin_fraction"],v["isomax_fraction"],v["volume_opacity"],v["projection_opacity"]=0.16,0.95,0.10,0.75
        v["font_family"],v["tick_size"],v["label_size"]="Times New Roman",14,16
    v["volume_opacity"]=_safe_float(v.get("volume_opacity"),0.10,0.01,1);v["projection_opacity"]=_safe_float(v.get("projection_opacity"),0.75,0,1);v["isomin_fraction"]=_safe_float(v.get("isomin_fraction"),0.16,0,0.99);v["isomax_fraction"]=_safe_float(v.get("isomax_fraction"),0.95,0.01,1)
    if v["isomax_fraction"]<=v["isomin_fraction"]:v["isomax_fraction"]=min(1,v["isomin_fraction"]+0.05)
    v["surface_count"]=_safe_int(v.get("surface_count"),25,2,25);v["smooth_sigma"]=_safe_float(v.get("smooth_sigma"),0.5,0,5);v["volume_stride"]=_safe_int(v.get("volume_stride"),2,2,8);v["rotation_speed"]=_safe_float(v.get("rotation_speed"),14,1,90);v["axis_tick_length"]=_safe_int(v.get("axis_tick_length"),3,0,10)
    if v.get("volume_colormap") not in NAPARI_STYLE_COLORMAPS:v["volume_colormap"]="magma"
    if v.get("volume_colorscale_light") not in VOLUME_SCALES:v["volume_colorscale_light"]="Blues"
    if v.get("volume_colorscale_dark") not in VOLUME_SCALES:v["volume_colorscale_dark"]="Magma"
    if v.get("projection_colorscale") not in PROJECTION_SCALES:v["projection_colorscale"]="Greys"
    if v.get("quality") not in ("auto","full","balanced","fast"):v["quality"]="auto"
    if v.get("dragmode") not in ("orbit","turntable"):v["dragmode"]="orbit"
    if v.get("projection_type") not in ("perspective","orthographic"):v["projection_type"]="perspective"
    if w.get("orientation") not in ("horizontal","vertical"):w["orientation"]="horizontal"
    if w.get("layout_choice") not in ("focus","row","column","map3d-top"):w["layout_choice"]="focus"
    if not isinstance(w.get("layout_ratios"),dict):w["layout_ratios"]=deepcopy(DEFAULT_PLOT_DISPLAY_SETTINGS["workspace"]["layout_ratios"])
    for key in ("map_geometry","curve_geometry","volume_geometry"):
        g=w.get(key)
        if not isinstance(g,(list,tuple)) or len(g)!=4:w[key]=None
        else:
            try:x,y,width,height=map(int,g);w[key]=[x,y,max(210,width),max(170,height)]
            except (TypeError,ValueError,OverflowError):w[key]=None
    s=w.get("viewport_size")
    if not isinstance(s,(list,tuple)) or len(s)!=2:w["viewport_size"]=None
    else:
        try:w["viewport_size"]=[max(1,int(s[0])),max(1,int(s[1]))]
        except (TypeError,ValueError,OverflowError):w["viewport_size"]=None
    return result

def save_plot_display_settings(values:dict)->None:
    app=load_settings();current=load_plot_display_settings()
    for section in current:
        if isinstance(values.get(section),dict):current[section].update(values[section])
    app[_SETTINGS_KEY]=current;save_settings(app)

def _spin(value,low,high,decimals=0,suffix="",step=None):
    w=QDoubleSpinBox() if decimals else QSpinBox();w.setRange(low,high);w.setValue(value);w.setSuffix(suffix);w.setKeyboardTracking(False)
    if decimals:w.setDecimals(decimals);w.setSingleStep(step if step is not None else 0.05)
    w.setFixedWidth(104 if decimals else 76);return w

def _combo(items,current):
    w=QComboBox()
    for item in items:
        if isinstance(item,tuple):w.addItem(item[0],item[1])
        else:w.addItem(str(item),str(item))
    idx=w.findData(current)
    if idx<0:idx=w.findText(str(current))
    if idx>=0:w.setCurrentIndex(idx)
    return w

class PlotDisplaySettingsDialog(QDialog):
    """Единые компактные настройки карты, кривых и 3D-сцены."""
    def __init__(self,settings:dict|None=None,parent=None):
        super().__init__(parent);self.setObjectName("plotDisplaySettings");self._defaults=deepcopy(DEFAULT_PLOT_DISPLAY_SETTINGS);self._values=deepcopy(settings or load_plot_display_settings());self._controls={"map":{},"curve":{},"volume":{}}
        self.setMinimumSize(445,430);self.resize(495,575);root=QVBoxLayout(self);root.setContentsMargins(7,6,7,7);root.setSpacing(5);install_dialog_titlebar(self,root,"Настройки экранов ROI Explorer");tabs=QTabWidget(self);tabs.setDocumentMode(True);root.addWidget(tabs,1)
        for key,title in (("map","Карта"),("curve","ROC-кривые"),("volume","3D")):
            scroll=QScrollArea(tabs);scroll.setWidgetResizable(True);scroll.setFrameShape(QFrame.Shape.NoFrame);page=QWidget();form=QFormLayout(page);form.setContentsMargins(7,6,7,5);form.setHorizontalSpacing(10);form.setVerticalSpacing(4);form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow);form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows);values=self._values[key];controls=self._controls[key]
            font=QFontComboBox(page);font.setCurrentFont(QFont(str(values["font_family"])));font.setMinimumWidth(145);controls["font_family"]=font;form.addRow("Шрифт",font)
            tick=_spin(int(values["tick_size"]),6,16);controls["tick_size"]=tick;form.addRow("Числа осей, pt",tick);label=_spin(int(values["label_size"]),6,18);controls["label_size"]=label;form.addRow("Подписи осей, pt",label)
            if key=="map":
                for name,caption,rowlabel in (("show_colorbar","Показывать цветовую шкалу","Цветовая шкала"),("show_scalebar","Показывать масштабный отрезок","Масштаб")):
                    w=QCheckBox(caption);w.setChecked(bool(values[name]));controls[name]=w;form.addRow(rowlabel,w)
                for name,caption,value,lo,hi,dec,suffix,step in (("scalebar_um","Длина масштабного отрезка",float(values["scalebar_um"]),10,100000,1," µm",5),("pixel_size_um","Размер пикселя детектора",float(values["pixel_size_um"]),0.001,100000,3," µm/пиксель",0.1),("scalebar_font_size","Подпись масштаба, pt",int(values["scalebar_font_size"]),6,16,0,"",None)):
                    w=_spin(value,lo,hi,dec,suffix,step);controls[name]=w;form.addRow(caption,w)
            elif key=="curve":
                w=QCheckBox("Ненавязчивая координатная сетка");w.setChecked(bool(values["show_grid"]));controls["show_grid"]=w;form.addRow("Сетка",w)
            else:
                for name,caption in (("show_grid","Координатная сетка"),("show_projection","2D-проекция под объёмом"),("show_colorbar","Цветовая шкала объёма"),("show_legend","Легенда")):
                    w=QCheckBox(caption);w.setChecked(bool(values[name]));controls[name]=w;form.addRow("",w)
                options=(("volume_colormap","Цветовая карта объёма",NAPARI_STYLE_COLORMAPS),("projection_colorscale","2D-проекция",PROJECTION_SCALES),("quality","Качество",(("Авто","auto"),("Полное","full"),("Сбалансированное","balanced"),("Быстрое","fast"))),("dragmode","Вращение мышью",(("Орбитальное","orbit"),("Поворот по вертикали","turntable"))),("projection_type","Проекция камеры",(("Перспективная","perspective"),("Ортографическая","orthographic"))))
                for name,caption,items in options:w=_combo(items,values[name]);controls[name]=w;form.addRow(caption,w)
                specs=(("volume_opacity","Прозрачность объёма",0.01,1,2),("projection_opacity","Прозрачность проекции",0,1,2),("isomin_fraction","Нижний порог объёма",0,0.99,2),("isomax_fraction","Верхний порог объёма",0.01,1,2),("smooth_sigma","Сглаживание σ",0,5,2),("rotation_speed","Скорость автоворота, °/с",1,90,1))
                for name,caption,lo,hi,dec in specs:w=_spin(float(values[name]),lo,hi,dec);controls[name]=w;form.addRow(caption,w)
                for name,caption,lo,hi in (("surface_count","Изоповерхностей",2,25),("volume_stride","Минимальный шаг прореживания",2,8),("axis_tick_length","Длина ticks 3D, px",0,10)):
                    w=_spin(int(values[name]),lo,hi);controls[name]=w;form.addRow(caption,w)
                note=QLabel("Авто увеличивает шаг только для больших объёмов. Геометрия ROI и координатные оси не меняются.",page);note.setWordWrap(True);note.setStyleSheet("color:palette(mid);font-size:10px;background:transparent;");form.addRow("",note)
            scroll.setWidget(page);tabs.addTab(scroll,title)
        footer=QHBoxLayout();footer.setContentsMargins(0,1,0,0);footer.setSpacing(5);reset=QPushButton("По умолчанию",self);reset.clicked.connect(self.reset_defaults);footer.addWidget(reset);footer.addStretch(1);buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Save|QDialogButtonBox.StandardButton.Cancel,parent=self);buttons.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить");buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена");buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);footer.addWidget(buttons);root.addLayout(footer)
        self.setStyleSheet(self.styleSheet()+"\nQDialog#plotDisplaySettings QSpinBox,QDialog#plotDisplaySettings QDoubleSpinBox,QDialog#plotDisplaySettings QComboBox,QDialog#plotDisplaySettings QFontComboBox{background:palette(base);color:palette(text);border:1px solid palette(mid);border-radius:3px;padding:2px 4px;min-height:20px;} QDialog#plotDisplaySettings QAbstractSpinBox::up-button,QDialog#plotDisplaySettings QAbstractSpinBox::down-button{width:0px;height:0px;border:0px;} QDialog#plotDisplaySettings QTabWidget::pane{border:1px solid palette(mid);} QDialog#plotDisplaySettings QTabBar::tab{padding:5px 11px;} QDialog#plotDisplaySettings QPushButton{padding:4px 9px;min-height:22px;}");apply_popup_theme(self,resolve_theme(parent))
    def reset_defaults(self):self._set_controls(deepcopy(self._defaults))
    def _set_controls(self,values):
        for section,controls in self._controls.items():
            for name,w in controls.items():
                value=values[section][name]
                if name=="font_family":w.setCurrentFont(QFont(str(value)))
                elif isinstance(w,QCheckBox):w.setChecked(bool(value))
                elif isinstance(w,QComboBox):
                    idx=w.findData(value);idx=idx if idx>=0 else w.findText(str(value))
                    if idx>=0:w.setCurrentIndex(idx)
                else:w.setValue(value)
    def values(self)->dict:
        result=deepcopy(self._values)
        for section,controls in self._controls.items():
            for name,w in controls.items():result[section][name]=w.currentFont().family() if name=="font_family" else w.isChecked() if isinstance(w,QCheckBox) else (w.currentData() if isinstance(w,QComboBox) else w.value())
        return result
