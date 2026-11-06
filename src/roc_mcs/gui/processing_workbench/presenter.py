"""Typed bridge between the workbench and the host ROI Explorer views.

The host app can connect these signals to its own MapPlotWidget, CurvePlotWidget
and VTK viewer without making the workflow engine depend on GUI implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class ArtifactPreviewRequest:
    project_directory: str
    node_id: str
    operation_id: str
    source_file: str
    scan_key: str
    grid_id: str
    input_refs: Mapping[str, Mapping[str, str]]
    output_refs: Mapping[str, str]
    preferred_view: str
    summary: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PreviewHostProtocol(Protocol):
    """Documented duck-typed contract for the main ROI Explorer window.

    Host methods:
      show_analysis_preview(request: ArtifactPreviewRequest)
      restore_analysis_preview()
    """

    def show_analysis_preview(self, request: ArtifactPreviewRequest) -> None:
        raise NotImplementedError

    def restore_analysis_preview(self) -> None:
        raise NotImplementedError
