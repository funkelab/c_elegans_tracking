from warnings import warn

import napari
import napari.layers
import numpy as np
import pyqtgraph as pg
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..dist_to_spline import dist_to_spline
from ..tracking.create_cand_graph import get_threshold
from ..worm_space import WormSpace


class CandidateNodeWidget(QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        lattice_points: np.ndarray,
        tracks_viewer=None,
    ):
        super().__init__()
        self.viewer = viewer
        self.lattice_points = lattice_points
        self.tracks_viewer: TracksViewer | None = tracks_viewer
        if self.tracks_viewer is not None:
            self.tracks_viewer.selected_nodes.list_updated.connect(
                self.compute_spline_distances
            )
        layout = QVBoxLayout()

        button = QPushButton("Compute spline distances")
        clear_button = QPushButton("Clear")
        button.clicked.connect(self.compute_spline_distances)
        clear_button.clicked.connect(self._clear_plots)
        self.dist_plot = self._dist_plot_widget()
        self.cand_plot = self._cand_plot_widget()
        layout.addWidget(button)
        layout.addWidget(self.dist_plot)
        layout.addWidget(self.cand_plot)
        layout.addWidget(clear_button)
        self.setLayout(layout)

    def compute_spline_distances(self):
        if self.tracks_viewer is not None:
            selected_points = self.tracks_viewer.selected_nodes._list
            data = self.tracks_viewer.tracks.get_positions(
                selected_points, incl_time=True
            )
        else:
            active_layer = self.viewer.layers.selection.active
            if not isinstance(active_layer, napari.layers.Points):
                warn(
                    "Please select a point in a points layer before computing spline "
                    "distances",
                    stacklevel=2,
                )
                return
            selected_points = list(active_layer.selected_data)
            data = active_layer.data[selected_points]
        if len(selected_points) != 1:
            warn(
                "Please select one point before computing spline distances",
                stacklevel=2,
            )
            return
        self.dist_plot.getPlotItem().clear()
        for location in data:
            time = int(location[0])
            loc = location[1:]
            worm_space = WormSpace(self.lattice_points[time])
            self.plot_distances(loc, worm_space)

    def plot_distances(self, loc: np.ndarray, worm_space: WormSpace):
        """Plot the distance from the point to center spline of worm space
        Args:
            point_loc (np.ndarray): 3d location
            worm_space (WormSpace): worm space at the time
        """
        threshold = get_threshold(worm_space)
        ap_pos, dist, local_minima = dist_to_spline(
            loc, worm_space.center_spline, worm_space.valid_range, threshold=threshold
        )
        self.dist_plot.getPlotItem().plot(ap_pos, dist)
        cand_locations = []
        for _min in local_minima:
            self.dist_plot.getPlotItem().plot([ap_pos[_min]], [dist[_min]], symbol="star")
            cand_locations.append(worm_space.get_worm_coords(loc, ap_pos[_min]))
        self.dist_plot.getPlotItem().plot(
            ap_pos,
            [
                threshold,
            ]
            * len(ap_pos),
        )
        color = np.random.randint(0, 155, size=(3,))
        self.plot_cand_locations(cand_locations, color=color)

    def plot_cand_locations(self, cand_locations, color):
        pen = pg.mkPen(color)
        brush = pg.mkBrush(color)
        for cand_loc in cand_locations:
            self.cand_plot.getPlotItem().plot(
                [cand_loc[1]],
                [cand_loc[2]],
                pen=pen,
                symbolBrush=brush,
                symbolPen="w",
            )

    def _clear_plots(self):
        self.cand_plot.getPlotItem().clear()
        white = (255, 255, 255)
        self.cand_plot.getPlotItem().plot(
            [0],
            [0],
            pen=pg.mkPen(white),
            symbolBrush=pg.mkBrush(white),
            symbol="+",
        )

    def _dist_plot_widget(self) -> pg.PlotWidget:
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

    def _cand_plot_widget(self) -> QWidget:
        """
        Returns:
            QWIdget: a widget containg a plot of candidate locations
        """
        gap_plot = pg.PlotWidget()
        gap_plot.setBackground((37, 41, 49))
        styles = {
            "color": "white",
        }
        gap_plot.plotItem.setLabel("left", "Dorsal ventral position", **styles)
        gap_plot.plotItem.setLabel("bottom", "Medial lateral position", **styles)

        white = (255, 255, 255)
        gap_plot.getPlotItem().plot(
            [0],
            [0],
            pen=pg.mkPen(white),
            symbolBrush=pg.mkBrush(white),
            symbol="+",
        )
        gap_plot.plotItem.setXRange(-100, 100)
        gap_plot.plotItem.setYRange(-100, 100)
        return gap_plot
