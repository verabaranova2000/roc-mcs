from roc_mcs.gui.roi_explorer.vtk_backend.display_palette import resolve_projection_palette


def test_auto_projection_palette_follows_theme():
    assert resolve_projection_palette(None, "dark") == "Greys_r"
    assert resolve_projection_palette("auto", "light") == "Greys"


def test_explicit_neutral_grayscale_palette_is_theme_aware():
    assert resolve_projection_palette("Greys", "dark") == "Greys_r"
    assert resolve_projection_palette("slate", "dark") == "Greys_r"
    assert resolve_projection_palette("gray", "dark") == "gray_r"
    assert resolve_projection_palette("gray", "light") == "gray"


def test_non_grayscale_palette_is_preserved():
    assert resolve_projection_palette("viridis", "dark") == "viridis"
    assert resolve_projection_palette("magma", "light") == "magma"


def test_3d_volume_range_slider_uses_the_same_theme_highlight_as_2d():
    import pytest
    QtGui = pytest.importorskip("PyQt6.QtGui")
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    QPalette = QtGui.QPalette
    QApplication = QtWidgets.QApplication

    from roc_mcs.gui.roi_explorer.theme import theme_colors
    from roc_mcs.gui.roi_explorer.vtk_backend.vtk_volume_display_widget import (
        _VolumeContrastRangeSlider,
    )

    app = QApplication.instance() or QApplication([])
    slider = _VolumeContrastRangeSlider()
    for theme in ("light", "dark"):
        slider.set_theme(theme)
        actual = slider.palette().color(QPalette.ColorRole.Highlight).name()
        assert actual == theme_colors(theme)["highlight"].lower()
    slider.deleteLater()
    app.processEvents()
