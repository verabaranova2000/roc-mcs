import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

MODULE = Path(__file__).with_name("volume_backend_lab.py")
spec = importlib.util.spec_from_file_location("volume_backend_lab_test", MODULE)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)

def test_default_stride_and_projection_keep_full_resolution():
    rv = module.make_synthetic_volume()
    data = module.prepare_render_data(rv, stride=2)
    assert data.stride == 2
    assert data.volume.shape == (36, 32, 22)
    assert data.projection.shape == (72, 64)
    assert len(data.projection_x) == 72 and len(data.projection_y) == 64

def test_plotly_builds_expected_traces_and_settings():
    rv = module.make_synthetic_volume()
    fig, stats = module.build_plotly_reference(rv, theme="dark", stride=2)
    assert len(fig.data) == 2 and fig.data[0].type == "surface" and fig.data[1].type == "volume"
    assert fig.data[1].to_plotly_json()["surface"]["count"] == 25
    assert fig.data[1].opacity == 0.10
    assert fig.layout.scene.bgcolor == "#111316"
    assert fig.data[0].colorscale[0][1] != fig.data[0].colorscale[-1][1]
    assert stats["stride"] == 2 and stats["volume_nodes_sent"] <= module.MAX_PLOTLY_NODES
    assert "data" in fig.to_json()

def test_adaptive_stride_protects_large_render_mesh():
    rv = module.make_synthetic_volume((180,160,120))
    data = module.prepare_render_data(rv, stride=2, max_nodes=100_000, smooth_sigma=0)
    assert data.stride >= 2
    assert data.volume.size <= 100_000

def test_descending_axes_are_normalized_for_rendering():
    from types import SimpleNamespace
    rv0 = module.make_synthetic_volume((24,20,18))
    rv = SimpleNamespace(volume=rv0.volume[::-1,::-1,::-1], x=rv0.x[::-1], y=rv0.y[::-1], theta=rv0.theta[::-1], coord_system="xy")
    data = module.prepare_render_data(rv, stride=2)
    assert np.all(np.diff(data.x) > 0) and np.all(np.diff(data.y) > 0) and np.all(np.diff(data.theta) > 0)
    assert data.projection.shape == (24,20)
    fig, stats = module.build_plotly_reference(rv, theme="light", stride=2)
    assert fig.data[0].surfacecolor.shape == (24,20)

def test_bad_volume_shape_is_rejected():
    from types import SimpleNamespace
    bad = SimpleNamespace(volume=np.zeros((4,5,6)), x=np.arange(4), y=np.arange(5), theta=np.arange(5), coord_system="xy")
    with pytest.raises(ValueError): module.prepare_render_data(bad)
