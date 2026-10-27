"""Тематические SVG-иконки Qt с кэшированием по цвету и размеру."""
from __future__ import annotations

import re
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication

ICON_FOREGROUND_PROPERTY = "rocMcsIconForeground"


def set_icon_theme(target: Any, foreground: QColor | str) -> None:
    """Задаёт явный цвет SVG-иконок для приложения/виджета; фон/QSS не анализируются."""
    value = QColor(foreground)
    if not value.isValid():
        raise ValueError(f"Некорректный цвет темы: {foreground!r}")
    target.setProperty(ICON_FOREGROUND_PROPERTY, value.name(QColor.NameFormat.HexRgb))


def _configured_color(widget: Any = None) -> QColor | None:
    app = QApplication.instance()
    current = widget
    while current is not None:
        value = current.property(ICON_FOREGROUND_PROPERTY)
        if value:
            candidate = QColor(str(value))
            if candidate.isValid():
                return candidate
        current = current.parentWidget() if hasattr(current, "parentWidget") else None
    if app is not None:
        value = app.property(ICON_FOREGROUND_PROPERTY)
        if value:
            candidate = QColor(str(value))
            if candidate.isValid():
                return candidate
    return None


def _color_name(color: QColor | str | None, palette: QPalette | None, widget: Any = None) -> str:
    if color is not None:
        result = QColor(color)
    else:
        result = _configured_color(widget)
        if result is None:
            app = QApplication.instance()
            if palette is not None:
                active_palette = palette
            elif widget is not None and hasattr(widget, "palette"):
                active_palette = widget.palette()
            else:
                active_palette = app.palette() if app else QPalette()
            result = active_palette.color(QPalette.ColorRole.WindowText)
    if not result.isValid():
        raise ValueError(f"Некорректный цвет иконки: {color!r}")
    return result.name(QColor.NameFormat.HexRgb)


def _recolor_svg(source: str, color: str) -> str:
    """Перекрашивает одноцветный SVG, включая фигуры без явного fill."""
    source = re.sub(r"\bcurrentColor\b", color, source, flags=re.IGNORECASE)

    def paint(value: str) -> str:
        value = value.strip(); low = value.lower()
        return value if low in {"none", "transparent"} or low.startswith("url(") else color

    def attr(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}{paint(match.group(3))}{match.group(2)}"

    source = re.sub(r"(\b(?:fill|stroke)\s*=\s*)(['\"])(.*?)\2", attr, source, flags=re.IGNORECASE | re.DOTALL)
    source = re.sub(r"(\b(?:fill|stroke)\s*:\s*)([^;{}\"']+)", lambda m: m.group(1) + paint(m.group(2)), source, flags=re.IGNORECASE)

    # В SVG по умолчанию fill чёрный. Задаём цвет на корневом <svg>,
    # чтобы перекрашивались и path без собственных fill/stroke.
    def set_root_fill(match: re.Match[str]) -> str:
        tag = match.group(0)
        if re.search(r"\bfill\s*=", tag, flags=re.IGNORECASE):
            tag = re.sub(r"(\bfill\s*=\s*)(['\"])(.*?)\2", lambda m: m.group(1) + m.group(2) + paint(m.group(3)) + m.group(2), tag, count=1, flags=re.IGNORECASE)
        else:
            tag = tag[:-1] + f' fill="{color}">' if tag.endswith('>') else tag + f' fill="{color}"'
        if not re.search(r"\bcolor\s*=", tag, flags=re.IGNORECASE):
            tag = tag[:-1] + f' color="{color}">' if tag.endswith('>') else tag + f' color="{color}"'
        return tag

    source = re.sub(r"<svg\b[^>]*>", set_root_fill, source, count=1, flags=re.IGNORECASE)
    return source


@lru_cache(maxsize=256)
def _render_icon(source: str, color: str, width: int, height: int, monochrome: bool) -> QIcon:
    source = _recolor_svg(source, color) if monochrome else re.sub(r"\bcurrentColor\b", color, source, flags=re.IGNORECASE)
    renderer = QSvgRenderer(QByteArray(source.encode("utf-8")))
    if not renderer.isValid():
        raise ValueError("Файл содержит некорректный SVG.")
    dpr = 2; pixmap = QPixmap(width * dpr, height * dpr); pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform); renderer.render(painter, QRectF(0, 0, width * dpr, height * dpr)); painter.end()
    pixmap.setDevicePixelRatio(dpr)
    return QIcon(pixmap)


def _dimensions(size: int | QSize) -> tuple[int, int]:
    width, height = (size.width(), size.height()) if isinstance(size, QSize) else (int(size), int(size))
    if width < 1 or height < 1:
        raise ValueError("Размер иконки должен быть положительным.")
    return width, height


def svg_icon(path: str | Path, *, size: int | QSize = 16, color: QColor | str | None = None,
             palette: QPalette | None = None, widget: Any = None, monochrome: bool = True) -> QIcon:
    """Создаёт QIcon из SVG-файла; цвет темы задаётся явно или берётся из свойства/палитры."""
    source_path = Path(path)
    if source_path.suffix.lower() != ".svg":
        raise ValueError(f"Ожидался SVG-файл: {source_path}")
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Не удалось прочитать SVG-иконку: {source_path}") from exc
    width, height = _dimensions(size)
    return _render_icon(source, _color_name(color, palette, widget), width, height, monochrome)


def resource_svg_icon(name: str, *, size: int | QSize = 16, color: QColor | str | None = None,
                      palette: QPalette | None = None, widget: Any = None, monochrome: bool = True) -> QIcon:
    """Загружает SVG из roc_mcs.gui.resources/icons без жёстких путей к диску."""
    if Path(name).name != name or not name.lower().endswith(".svg"):
        raise ValueError("Укажите только имя SVG-файла из каталога resources/icons.")
    resource = files("roc_mcs.gui.resources").joinpath("icons", name)
    try:
        source = resource.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError) as exc:
        raise FileNotFoundError(f"В ресурсах не найдена SVG-иконка: {name}") from exc
    width, height = _dimensions(size)
    return _render_icon(source, _color_name(color, palette, widget), width, height, monochrome)
