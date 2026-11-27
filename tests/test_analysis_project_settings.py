from __future__ import annotations

from roc_mcs.gui import app_settings


def test_analysis_project_root_uses_configured_absolute_directory(tmp_path):
    configured = tmp_path / "custom analysis root"
    resolved = app_settings.resolve_analysis_projects_directory(
        {"analysis_projects_dir": str(configured)}
    )
    assert resolved == configured.resolve()


def test_analysis_project_root_falls_back_to_documents_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        app_settings, "default_analysis_projects_directory", lambda: tmp_path
    )
    assert app_settings.resolve_analysis_projects_directory({}) == tmp_path
    assert app_settings.resolve_analysis_projects_directory(
        {"analysis_projects_dir": ""}
    ) == tmp_path
