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
