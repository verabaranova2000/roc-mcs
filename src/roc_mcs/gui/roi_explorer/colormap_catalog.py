"""Shared colormap catalog for ROI Explorer 2D/3D controls.

The public names follow napari's image-layer palette vocabulary where practical,
while rendering stays independent from napari itself.  Matplotlib supplies the
actual LUTs and Plotly receives sampled RGB stops from the same LUT, so 2D and
3D use one visual source of truth.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, to_hex

NAPARI_STYLE_COLORMAPS = (
    "gray", "gray_r", "blue", "cyan", "green", "magenta", "red", "yellow",
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "twilight", "twilight_shifted", "hsv", "gist_earth", "fire", "ice",
    "bop blue", "bop orange", "bop purple", "I Blue", "I Bordeaux",
    "I Forest", "I Orange", "I Purple", "PiYG", "HiLo",
)

_FALLBACKS = {
    "blue": ("#000000", "#0000ff"), "cyan": ("#000000", "#00ffff"),
    "green": ("#000000", "#00ff00"), "magenta": ("#000000", "#ff00ff"),
    "red": ("#000000", "#ff0000"), "yellow": ("#000000", "#ffff00"),
    "fire": ("#000000", "#7f0000", "#ff6a00", "#ffffcc"),
    "ice": ("#000000", "#003b70", "#00bfff", "#eaffff"),
    "bop blue": ("#000000", "#163d70", "#2f78b7", "#d8efff"),
    "bop orange": ("#000000", "#783b0b", "#d77a1f", "#fff0cc"),
    "bop purple": ("#000000", "#4a1f67", "#8a4ba8", "#f0d8ff"),
    "I Blue": ("#ffffff", "#b9d8f3", "#3176b5", "#081d58"),
    "I Bordeaux": ("#ffffff", "#e7b8c5", "#9b3353", "#4c1024"),
    "I Forest": ("#ffffff", "#b9ddb7", "#46884b", "#10391d"),
    "I Orange": ("#ffffff", "#ffd29d", "#e67818", "#713000"),
    "I Purple": ("#ffffff", "#d7c0e7", "#75539a", "#2e174d"),
    "HiLo": ("#0000ff", "#ffffff", "#ff0000"),
}

@lru_cache(maxsize=None)
def mpl_colormap(name: str):
    name=str(name)
    # Matplotlib names are case sensitive for some registered maps.
    for candidate in (name, name.lower(), name.replace(" ", "_"), name.replace(" ", "_" ).lower()):
        try:
            return colormaps[candidate]
        except KeyError:
            pass
    colors=_FALLBACKS.get(name) or _FALLBACKS.get(name.lower())
    if colors is None:
        return colormaps["viridis"]
    return LinearSegmentedColormap.from_list("roi_"+name.replace(" ", "_"), colors, N=256)

def plotly_colorscale(name: str, samples: int = 32):
    cmap=mpl_colormap(name); xs=np.linspace(0.0,1.0,max(2,int(samples)))
    return [[float(x), to_hex(cmap(float(x)), keep_alpha=False)] for x in xs]

def colormap_icon(name: str, width: int = 72, height: int = 14):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPixmap
    pix=QPixmap(width,height); pix.fill(Qt.GlobalColor.transparent); painter=QPainter(pix)
    grad=QLinearGradient(0,0,width,0); cmap=mpl_colormap(name)
    for x in np.linspace(0.0,1.0,17):
        r,g,b,a=cmap(float(x)); grad.setColorAt(float(x),QColor.fromRgbF(float(r),float(g),float(b),float(a)))
    painter.fillRect(0,0,width,height,grad); painter.end(); return QIcon(pix)
