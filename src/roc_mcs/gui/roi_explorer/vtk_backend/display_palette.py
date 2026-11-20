"""Тематически согласованный выбор палитры 2D-проекции VTK."""
from __future__ import annotations


def resolve_projection_palette(configured: str | None, theme: str) -> str:
    """Автоматически инвертировать нейтральную серую шкалу в тёмной теме."""
    active_theme = str(theme).strip().lower()
    if active_theme not in ("dark", "light"):
        active_theme = "light"
    name = str(configured or "auto").strip()
    key = name.lower()
    if key in ("", "auto"):
        return "Greys_r" if active_theme == "dark" else "Greys"
    if key in ("slate", "greys", "grey", "gray"):
        if key in ("grey", "gray"):
            return "gray_r" if active_theme == "dark" else "gray"
        return "Greys_r" if active_theme == "dark" else "Greys"
    return name
