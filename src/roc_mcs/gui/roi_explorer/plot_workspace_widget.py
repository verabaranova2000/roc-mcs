from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplitter

from roc_mcs.gui.roi_explorer.roi_manager_widget import RoiManagerWidget
from roc_mcs.gui.roi_explorer.map_plot_widget import MapPlotWidget
from roc_mcs.gui.roi_explorer.curve_plot_widget import CurvePlotWidget


class PlotWorkspaceWidget(QSplitter):
    """Единая рабочая область: менеджер ROI, карта и кривая."""
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.roi_manager = RoiManagerWidget()
        self.roi_table = self.roi_manager.table
        self.map_widget = MapPlotWidget()
        self.curve_widget = CurvePlotWidget()
        for widget in (self.roi_manager, self.map_widget, self.curve_widget):
            self.addWidget(widget)
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 1)
        self.setStretchFactor(2, 1)
        self.setSizes([26, 500, 500])
