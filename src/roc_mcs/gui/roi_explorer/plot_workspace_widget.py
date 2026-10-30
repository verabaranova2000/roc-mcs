"""Рабочая область научных экранов: ручное изменение окон не перетасовывает соседей."""
from __future__ import annotations
import math
from PyQt6.QtCore import QEvent,QRect,QSize,Qt,QTimer,pyqtSignal,QRectF
from PyQt6.QtWidgets import QHBoxLayout,QMdiArea,QMdiSubWindow,QSizePolicy,QSplitter,QToolButton,QVBoxLayout,QWidget,QMenu,QWidgetAction
from PyQt6.QtGui import QPainter,QPen,QColor,QPainterPath
from roc_mcs.gui.roi_explorer.roi_manager_widget import RoiManagerWidget
from roc_mcs.gui.roi_explorer.map_plot_widget import MapPlotWidget
from roc_mcs.gui.roi_explorer.curve_plot_widget import CurvePlotWidget
from roc_mcs.gui.roi_explorer.volume_plot_widget import VolumePlotWidget
from roc_mcs.gui.roi_explorer.plot_display_settings import load_plot_display_settings,save_plot_display_settings
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette,resolve_theme,theme_colors

_LAYOUTS = ("focus", "row", "column", "map3d-top")
_LAYOUT_VERSION = 7
DEFAULT_VISIBLE_PANELS = ("map", "curve", "volume")  # Для старта только с картой и ROC укажи ("map", "curve").
LAYOUT_PREVIEW_STYLE = "A"  # A/B/C — варианты рисунка раскладок; см. PNG-лист.
FOCUS_LEFT_RATIO = 0.50  # Доля ширины карты в стартовой раскладке.
FOCUS_TOP_RATIO = 0.50  # Доля высоты верхнего окна ROC справа.
ROI_RAIL_COMPACT_WIDTH = 22
ROI_RAIL_EXPANDED_WIDTH = 282

class _PlotSubWindow(QMdiSubWindow):
    """Закрытие скрывает экран и сохраняет его состояние."""
    def __init__(self,area,key,title):
        super().__init__();self._area=area;self.plot_key=key;self.setWindowTitle(title);self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose,False);self.setOption(QMdiSubWindow.SubWindowOption.RubberBandMove,True);self.setOption(QMdiSubWindow.SubWindowOption.RubberBandResize,True)
    def closeEvent(self,event):event.accept();self.hide();self._area.panel_visibility_changed.emit(self.plot_key,False)

class _PlotMdiArea(QMdiArea):
    """Раскладывает окна только по явной команде; ручной resize соседей не затрагивает."""
    panel_visibility_changed=pyqtSignal(str,bool)
    def __init__(self,parent=None):
        super().__init__(parent);self.setObjectName("plotMdiArea");self.setViewMode(QMdiArea.ViewMode.SubWindowView);self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded);self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded);self.setActivationOrder(QMdiArea.WindowOrder.ActivationHistoryOrder);self.setOption(QMdiArea.AreaOption.DontMaximizeSubWindowOnActivation,True);self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding);self.setMinimumSize(340,300);self._keys={};self._layout_choice="focus";self._viewport_size=QSize();self._area_size=QSize();self._resize_pending=False;self._save_timer=QTimer(self);self._save_timer.setSingleShot(True);self._save_timer.setInterval(500);self._save_timer.timeout.connect(self.save_geometry);self._viewport_timer=QTimer(self);self._viewport_timer.setSingleShot(True);self._viewport_timer.setInterval(180);self._viewport_timer.timeout.connect(self._fit_after_viewport_resize);self.panel_visibility_changed.connect(self._schedule_save);self.panel_visibility_changed.connect(lambda *_:QTimer.singleShot(100,lambda:self._apply_layout(self._layout_choice)))
    def add_plot(self,key,title,widget):
        win=_PlotSubWindow(self,key,title);win.setWindowFlags(Qt.WindowType.SubWindow|Qt.WindowType.CustomizeWindowHint|Qt.WindowType.WindowTitleHint|Qt.WindowType.WindowCloseButtonHint|Qt.WindowType.WindowMinMaxButtonsHint);win.setWidget(widget);m=widget.minimumSize();h=widget.minimumSizeHint();win.setMinimumSize(QSize(max(250,m.width(),h.width())+2,max(190,m.height(),h.height())+24));self.addSubWindow(win);self._keys[key]=win;win.installEventFilter(self);win.show();return win
    def show_panel(self,key,visible):
        win=self._keys.get(key)
        if win is None:return
        if visible:win.show();win.raise_();self.setActiveSubWindow(win)
        else:win.hide()
        self.panel_visibility_changed.emit(key,bool(visible))
    def _bounds(self):return self.viewport().rect().adjusted(2,2,-2,-2)
    def _visible(self):return [w for w in self._keys.values() if not w.isHidden() and not w.isMinimized()]
    def _layout_initial(self):
        self._area_size=self.size();saved=load_plot_display_settings().get("workspace",{});self._layout_choice=saved.get("layout_choice","focus") if saved.get("layout_choice") in _LAYOUTS else "focus"
        # Геометрии старых версий не восстанавливаем: они были записаны при другой геометрии панелей.
        if int(saved.get("layout_version",0) or 0)<_LAYOUT_VERSION:self._apply_layout("focus");return
        bounds=self._bounds();prev=saved.get("viewport_size") or [bounds.width(),bounds.height()];sx=bounds.width()/max(1,int(prev[0]));sy=bounds.height()/max(1,int(prev[1]))
        restored=False
        for key,w in self._keys.items():
            g=saved.get(f"{key}_geometry")
            if isinstance(g,(list,tuple)) and len(g)==4:
                try:x,y,ww,hh=map(int,g);w.setGeometry(self._clamp(QRect(round(x*sx),round(y*sy),round(ww*sx),round(hh*sy)),bounds));restored=True
                except (ValueError,TypeError,OverflowError):pass
        if not restored:self._apply_layout(self._layout_choice)
        self._viewport_size=self.viewport().size();self._area_size=self.size()
    def set_layout_choice(self,choice):
        if choice not in _LAYOUTS:return
        self._layout_choice=choice;self._apply_layout(choice);self._schedule_save()
    def _apply_layout(self,choice=None):
        if choice in _LAYOUTS:self._layout_choice=choice
        wins=self._visible();bounds=self._bounds();n=len(wins)
        if not n or bounds.width()<120 or bounds.height()<100:return
        by={w.plot_key:w for w in wins};x,y,w,h=bounds.x(),bounds.y(),bounds.width(),bounds.height();gap=6
        def minimum(key,axis):
            win=by[key];return win.minimumWidth() if axis=="w" else win.minimumHeight()
        # При тесном окне безопаснее расположить окна каскадом, чем выталкивать их за рамки.
        choice=self._layout_choice
        if n==1:by[wins[0].plot_key].setGeometry(bounds)
        elif n==2:
            keys=[k for k in ("map","curve","volume") if k in by];a,b=keys
            if choice=="column" and h-gap>=minimum(a,"h")+minimum(b,"h"):
                first=max(minimum(a,"h"),min(h-gap-minimum(b,"h"),int((h-gap)*0.5)));by[a].setGeometry(QRect(x,y,w,first));by[b].setGeometry(QRect(x,y+first+gap,w,h-first-gap))
            elif w-gap>=minimum(a,"w")+minimum(b,"w"):
                first=max(minimum(a,"w"),min(w-gap-minimum(b,"w"),int((w-gap)*0.5)));by[a].setGeometry(QRect(x,y,first,h));by[b].setGeometry(QRect(x+first+gap,y,w-first-gap,h))
            else:
                first=max(1,(h-gap)//2);by[a].setGeometry(QRect(x,y,w,first));by[b].setGeometry(QRect(x,y+first+gap,w,h-first-gap))
        elif choice=="row" and w-2*gap>=sum(minimum(k,"w") for k in ("map","curve","volume")):
            sizes=self._fit_sizes(w-2*gap,[.34,.33,.33],[minimum(k,"w") for k in ("map","curve","volume")]);pos=x
            for key,size in zip(("map","curve","volume"),sizes):by[key].setGeometry(QRect(pos,y,size,h));pos+=size+gap
        elif choice=="column" and h-2*gap>=sum(minimum(k,"h") for k in ("map","curve","volume")):
            sizes=self._fit_sizes(h-2*gap,[.30,.30,.40],[minimum(k,"h") for k in ("map","curve","volume")]);pos=y
            for key,size in zip(("map","curve","volume"),sizes):by[key].setGeometry(QRect(x,pos,w,size));pos+=size+gap
        elif choice=="map3d-top" and h-gap>=max(minimum("map","h"),minimum("volume","h"))+minimum("curve","h"):
            top=max(max(minimum("map","h"),minimum("volume","h")),min(h-gap-minimum("curve","h"),int((h-gap)*.58)));left=max(minimum("map","w"),min(w-gap-minimum("volume","w"),(w-gap)//2));by["map"].setGeometry(QRect(x,y,left,top));by["volume"].setGeometry(QRect(x+left+gap,y,w-left-gap,top));by["curve"].setGeometry(QRect(x,y+top+gap,w,h-top-gap))
        else:
            # Стартовая схема: карта слева на всю высоту; ROC сверху справа, 3D снизу справа.
            left_min=max(minimum("map","w"),180);right_min=max(minimum("curve","w"),minimum("volume","w"),180)
            if w-gap>=left_min+right_min and h-gap>=minimum("curve","h")+minimum("volume","h"):
                left=max(left_min,min(w-gap-right_min,round((w-gap)*FOCUS_LEFT_RATIO)));right=w-left-gap
                top=max(minimum("curve","h"),min(h-gap-minimum("volume","h"),round((h-gap)*FOCUS_TOP_RATIO)))
                by["map"].setGeometry(QRect(x,y,left,h));by["curve"].setGeometry(QRect(x+left+gap,y,right,top));by["volume"].setGeometry(QRect(x+left+gap,y+top+gap,right,h-top-gap))
            else:
                # Если места мало, панели не накладываются: они доступны прокруткой MDI.
                heights=self._fit_sizes(max(3,h-2*gap),[.34,.33,.33],[minimum(k,"h") for k in ("map","curve","volume")]);pos=y
                for key,size in zip(("map","curve","volume"),heights):by[key].setGeometry(QRect(x,pos,w,size));pos+=size+gap
        self._viewport_size=self.viewport().size();self._area_size=self.size();self._schedule_save()
    @staticmethod
    def _fit_sizes(total,weights,minimums):
        total=max(1,int(total));minimums=[max(1,int(v)) for v in minimums]
        if sum(minimums)>total:return [max(1,int(total*max(.01,w)/sum(weights))) for w in weights]
        weights=[max(.01,float(v)) for v in weights];sw=sum(weights);sizes=[max(m,int(total*v/sw)) for m,v in zip(minimums,weights)]
        while sum(sizes)>total:
            i=max((i for i,v in enumerate(sizes) if v>minimums[i]),key=lambda i:sizes[i]-minimums[i],default=None)
            if i is None:break
            sizes[i]-=1
        while sum(sizes)<total:sizes[max(range(len(weights)),key=lambda i:weights[i])]+=1
        return sizes
    @staticmethod
    def _clamp(rect,bounds):
        ww=max(1,min(rect.width(),bounds.width()));hh=max(1,min(rect.height(),bounds.height()));return QRect(max(bounds.x(),min(rect.x(),bounds.right()-ww+1)),max(bounds.y(),min(rect.y(),bounds.bottom()-hh+1)),ww,hh)
    def resizeEvent(self,event):
        old=QSize(self._area_size);super().resizeEvent(event);new=QSize(self.size())
        # Переразмеривание дочернего QMdiSubWindow не влияет на соседей. При изменении
        # внешней области лишь удерживаем каждое окно в пределах нового viewport.
        if old.isValid() and old.width()>0 and old.height()>0 and (new.width()!=old.width() or new.height()!=old.height()):self._viewport_timer.start()
        elif not old.isValid():self._area_size=new;self._viewport_size=self.viewport().size()
    def _fit_after_viewport_resize(self):
        bounds=self._bounds()
        for win in self._visible():
            g=win.geometry();ww=min(g.width(),bounds.width());hh=min(g.height(),bounds.height())
            x=max(bounds.left(),min(g.x(),bounds.right()-ww+1));y=max(bounds.top(),min(g.y(),bounds.bottom()-hh+1))
            if (x,y,ww,hh)!=(g.x(),g.y(),g.width(),g.height()):win.setGeometry(QRect(x,y,ww,hh))
        self._viewport_size=self.viewport().size();self._area_size=self.size();self._schedule_save()
    def eventFilter(self,watched,event):
        # Только сохраняем ручную геометрию — никогда не меняем её в ответ на drag/resize.
        if watched in self._keys.values() and event.type() in (QEvent.Type.Move,QEvent.Type.Resize):self._schedule_save()
        return super().eventFilter(watched,event)
    def _schedule_save(self,*_):self._save_timer.start()
    def save_geometry(self):
        data=load_plot_display_settings();work=data.setdefault("workspace",{})
        for key,win in self._keys.items():work[f"{key}_geometry"]=list(win.geometry().getRect()) if not win.isHidden() else None
        work["orientation"]="manual";work["layout_choice"]=self._layout_choice;work["layout_version"]=_LAYOUT_VERSION;work["viewport_size"]=[max(1,self.viewport().width()),max(1,self.viewport().height())]
        try:save_plot_display_settings(data)
        except Exception:pass

class _LayoutPreviewButton(QToolButton):
    """Схематичная плитка-предпросмотр схемы окон."""
    def __init__(self,mode,label,parent=None):
        super().__init__(parent);self.mode=mode;self.label=label;self._selected=False;self.setFixedSize(158,54);self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly);self.setCursor(Qt.CursorShape.PointingHandCursor);self.setToolTip(label)
    def set_selected(self,value):self._selected=bool(value);self.update()
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing,True);pal=self.palette();bg=pal.color(pal.ColorRole.Base);fg=pal.color(pal.ColorRole.Text);edge=pal.color(pal.ColorRole.Highlight if self._selected else pal.ColorRole.Mid)
        p.setPen(QPen(edge,1.5 if self._selected else 0.8));p.setBrush(bg);p.drawRoundedRect(QRectF(1,1,self.width()-2,self.height()-2),4,4)
        left,top,ww,hh,gap=8,7,82,40,3;panel=QColor(100,145,185,75)
        def tile(rx,ry,rw,rh,kind):
            p.setPen(QPen(QColor(100,145,185,180),0.75));p.setBrush(panel);p.drawRoundedRect(QRectF(rx,ry,rw,rh),2,2)
            if kind=="ROC":
                # Только гауссиана без текста «ROC»: проще распознаётся в маленькой плитке.
                x0=rx+max(2,rw*.10);x1=rx+rw-max(2,rw*.10);baseline=ry+rh*.78;mid=(x0+x1)*.5;half=max(1,(x1-x0)*(.19 if LAYOUT_PREVIEW_STYLE=="A" else .24 if LAYOUT_PREVIEW_STYLE=="B" else .15));amp=max(2,rh*(.43 if LAYOUT_PREVIEW_STYLE!="C" else .34));path=QPainterPath();pts=[]
                for j in range(31):
                    xx=x0+(x1-x0)*j/30;yy=baseline-amp*math.exp(-.5*((xx-mid)/half)**2);pts.append((xx,yy))
                path.moveTo(*pts[0]);[path.lineTo(xx,yy) for xx,yy in pts[1:]]
                if LAYOUT_PREVIEW_STYLE=="B":p.setPen(Qt.PenStyle.NoPen);p.setBrush(QColor(90,150,195,75));fill=QPainterPath(path);fill.lineTo(x1,baseline);fill.lineTo(x0,baseline);fill.closeSubpath();p.drawPath(fill)
                p.setBrush(Qt.BrushStyle.NoBrush);p.setPen(QPen(fg,1.25 if LAYOUT_PREVIEW_STYLE!="C" else 1.0));p.drawPath(path)
                if LAYOUT_PREVIEW_STYLE=="C":p.setPen(QPen(fg,0.7));p.drawLine(QLineF(x0,baseline,x1,baseline))
            else:
                p.setPen(fg);p.drawText(QRectF(rx,ry,rw,rh),int(Qt.AlignmentFlag.AlignCenter),kind)
        if self.mode=="focus":tile(left,top,ww*.5-gap/2,hh,"2D");tile(left+ww*.5+gap/2,top,ww*.5-gap/2,hh*.49,"ROC");tile(left+ww*.5+gap/2,top+hh*.51,ww*.5-gap/2,hh*.49,"3D")
        elif self.mode=="row":
            cw=(ww-2*gap)/3
            for i,kind in enumerate(("2D","ROC","3D")):tile(left+i*(cw+gap),top,cw,hh,kind)
        elif self.mode=="column":
            ch=(hh-2*gap)/3
            for i,kind in enumerate(("2D","ROC","3D")):tile(left,top+i*(ch+gap),ww,ch,kind)
        else:tile(left,top,ww*.5-gap/2,hh*.49,"2D");tile(left+ww*.5+gap/2,top,ww*.5-gap/2,hh*.49,"3D");tile(left,top+hh*.51,ww,hh*.49,"ROC")
        p.setPen(fg);p.drawText(QRectF(96,4,56,46),int(Qt.AlignmentFlag.AlignVCenter|Qt.AlignmentFlag.AlignLeft|Qt.TextFlag.TextWordWrap),self.label);p.end()

class PlotWorkspaceWidget(QSplitter):
    """Слева менеджер ROI; справа три независимых MDI-экрана."""
    def __init__(self,parent=None):
        super().__init__(Qt.Orientation.Horizontal,parent);self.setObjectName("PlotWorkspaceWidget");self.roi_manager=RoiManagerWidget();self.roi_table=self.roi_manager.table;self.map_widget=MapPlotWidget();self.curve_widget=CurvePlotWidget();self.volume_widget=VolumePlotWidget();self._map_ratio=.53
        self.plot_area=QWidget(self);self.plot_area.setObjectName("PlotScreensArea");self.plot_area.setMinimumSize(340,300);layout=QVBoxLayout(self.plot_area);layout.setContentsMargins(0,0,0,0);layout.setSpacing(2)
        controls=QWidget(self.plot_area);controls.setObjectName("PlotScreensToolbar");row=QHBoxLayout(controls);row.setContentsMargins(2,0,2,0);row.setSpacing(4);self._buttons={}
        for key,text,tip in (("map","Карта","Показать или скрыть окно карты"),("curve","ROC-кривые","Показать или скрыть окно ROC-кривых"),("volume","3D","Показать или скрыть 3D-объём")):
            b=QToolButton(controls);b.setText(text);b.setCheckable(True);b.setChecked(key in DEFAULT_VISIBLE_PANELS);b.setToolTip(tip);b.setAutoRaise(True);b.setFixedHeight(23);row.addWidget(b);self._buttons[key]=b
        self.map_window_btn=self._buttons["map"];self.curve_window_btn=self._buttons["curve"];self.volume_window_btn=self._buttons["volume"]
        self.layout_btn=QToolButton(controls);self.layout_btn.setText("Раскладка ▦");self.layout_btn.setToolTip("Выбрать схему размещения окон");self.layout_btn.setAutoRaise(True);self.layout_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup);self.layout_menu=QMenu(self.layout_btn);self.layout_btn.setMenu(self.layout_menu);self._layout_preview_buttons={}
        self.arrange_btn=QToolButton(controls);self.arrange_btn.setText("Выровнять");self.arrange_btn.setToolTip("Применить выбранную схему к видимым окнам");self.arrange_btn.setAutoRaise(True);self.arrange_btn.setFixedHeight(23);row.addStretch(1);row.addWidget(self.layout_btn);row.addWidget(self.arrange_btn)
        self.mdi_area=_PlotMdiArea(self.plot_area);layout.addWidget(controls);layout.addWidget(self.mdi_area,1)
        self.map_subwindow=self.mdi_area.add_plot("map","2D MAP · Детектор",self.map_widget);self.curve_subwindow=self.mdi_area.add_plot("curve","ROC CURVES · Rocking curves",self.curve_widget);self.volume_subwindow=self.mdi_area.add_plot("volume","3D VOLUME · Diffraction",self.volume_widget)
        for key,b in self._buttons.items():
            b.toggled.connect(lambda visible,k=key:self.mdi_area.show_panel(k,visible))
            if key not in DEFAULT_VISIBLE_PANELS:self.mdi_area.show_panel(key,False)
        for mode,label in (("focus","Карта слева · ROC и 3D справа"),("row","Три окна в ряд"),("column","Три окна вертикально"),("map3d-top","Карта и 3D сверху · ROC снизу")):
            preview=_LayoutPreviewButton(mode,label,self.layout_menu);action=QWidgetAction(self.layout_menu);action.setDefaultWidget(preview);self.layout_menu.addAction(action);preview.clicked.connect(lambda _=False,m=mode:self._choose_layout(m));self._layout_preview_buttons[mode]=preview
        self.arrange_btn.clicked.connect(lambda:self.mdi_area.set_layout_choice(self.mdi_area._layout_choice));self.mdi_area.panel_visibility_changed.connect(self._sync_panel_buttons)
        self.addWidget(self.roi_manager);self.addWidget(self.plot_area);self.setStretchFactor(0,0);self.setStretchFactor(1,1);self.setCollapsible(0,True);self.setCollapsible(1,False);self.setHandleWidth(8);self.roi_manager.setFixedWidth(22);self.setSizes([22,900]);self.setMinimumSize(400,340);self._style_toolbar();QTimer.singleShot(0,self._force_compact_rail);QTimer.singleShot(0,self.mdi_area._layout_initial)
    def _choose_layout(self,mode):
        self.mdi_area.set_layout_choice(mode)
        for key,button in self._layout_preview_buttons.items():button.set_selected(key==mode)
        self.layout_menu.close()
    def _sync_panel_buttons(self,key,visible):
        b=self._buttons.get(key)
        if b:b.blockSignals(True);b.setChecked(bool(visible));b.blockSignals(False)
    def _style_toolbar(self):
        c=theme_colors(resolve_theme(self));self.plot_area.setStyleSheet(f"QWidget#PlotScreensArea{{background:{c['window']};}} QWidget#PlotScreensToolbar QToolButton{{color:{c['text']};background:transparent;border:1px solid transparent;border-radius:3px;padding:2px 7px;font-size:10px;}} QWidget#PlotScreensToolbar QToolButton:hover{{background:{c['hover']};border-color:{c['border']};}} QWidget#PlotScreensToolbar QToolButton:checked{{background:{c['button']};border-color:{c['border']};}}")
    def apply_theme(self,theme=None):
        apply_widget_palette(self,theme);self._style_toolbar();c=theme_colors(theme or resolve_theme(self))
        for win in (self.map_subwindow,self.curve_subwindow,self.volume_subwindow):win.setPalette(self.palette());win.setStyleSheet(f"QMdiSubWindow{{background:{c['window']};border:2px solid {c['border']};border-radius:2px;}} QMdiSubWindow:hover{{border-color:{c['highlight']};}}")
        self.mdi_area.setBackground(self.palette().window())
        for key,button in self._layout_preview_buttons.items():button.set_selected(key==self.mdi_area._layout_choice)
    def setSizes(self,sizes):
        values=[max(0,int(v)) for v in (sizes or [])]
        if not values:return
        if len(values)>=3:
            right_total=max(1,sum(values[1:]));self._map_ratio=max(.25,min(.75,values[1]/right_total))
        total=max(sum(values),self.width() if self.width()>0 else 0);rail=ROI_RAIL_EXPANDED_WIDTH if self.roi_manager._expanded else ROI_RAIL_COMPACT_WIDTH
        super().setSizes([rail,max(1,total-rail)])
    def _force_compact_rail(self):
        if not self.roi_manager._expanded:
            self.roi_manager.table.hide();self.roi_manager.setFixedWidth(ROI_RAIL_COMPACT_WIDTH);self.roi_manager.rail.setFixedWidth(ROI_RAIL_COMPACT_WIDTH-2)
            super().setSizes([ROI_RAIL_COMPACT_WIDTH,max(1,self.width()-ROI_RAIL_COMPACT_WIDTH-8)])
