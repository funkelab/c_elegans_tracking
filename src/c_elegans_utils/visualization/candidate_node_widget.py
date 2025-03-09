from warnings import warn

import napari
import napari.layers
import numpy as np
import pyqtgraph as pg
from napari.layers import Points
from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..dist_to_spline import dist_to_spline
from ..worm_space import WormSpace


class CandidateNodeWidget(QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        lattice_points: np.ndarray,
    ):
        super().__init__()
        self.viewer = viewer
        self.lattice_points = lattice_points
        layout = QVBoxLayout()

        button = QPushButton("Compute spline distances")
        button.clicked.connect(self.compute_spline_distances)
        self.dist_plot = self._plot_widget()
        layout.addWidget(button)
        layout.addWidget(self.dist_plot)
        self.setLayout(layout)
        # for debugging
        self.viewer.add_layer(Points(data=np.array([[4, 125, 202, 209]]), name="target"))

    def compute_spline_distances(self):
        active_layer = self.viewer.layers.selection.active
        if not isinstance(active_layer, napari.layers.Points):
            warn(
                "Please select a point in a points layer before computing spline "
                "distances",
                stacklevel=2,
            )
            return
        selected_points = active_layer.selected_data
        if len(selected_points) != 1:
            warn(
                "Please select one point in a points layer before computing spline "
                "distances",
                stacklevel=2,
            )
            return
        self.dist_plot.getPlotItem().clear()
        for point in selected_points:
            data = active_layer.data[point]
            time = int(data[0])
            loc = data[1:]
            worm_space = WormSpace(self.lattice_points[time])
            self.plot_distances(loc, worm_space)

    def plot_distances(self, loc: np.ndarray, worm_space: WormSpace):
        """Plot the distance from the point to center spline of worm space
        Args:
            point_loc (np.ndarray): 3d location
            worm_space (WormSpace): worm space at the time
        """
        ap_pos, dist, local_minima = dist_to_spline(
            [loc], worm_space.center_spline, worm_space.valid_range
        )
        dist = dist[0]
        local_minima = local_minima[0]
        self.dist_plot.getPlotItem().plot(ap_pos, dist)
        for _min in local_minima:
            self.dist_plot.getPlotItem().plot(
                [ap_pos[_min]], [dist[_min]], symbol="star", color="yellow"
            )

    def _plot_widget(self) -> pg.PlotWidget:
        """
        Returns:
            pg.PlotWidget: a widget containg an (empty) plot of the distance to the 
            center spline
        """
        gap_plot = pg.PlotWidget()
        gap_plot.setBackground((37, 41, 49))
        styles = {
            "color": "white",
        }
        gap_plot.plotItem.setLabel("left", "Distance to center spline", **styles)
        gap_plot.plotItem.setLabel("bottom", "Anterior-posterior position", **styles)
        return gap_plot

