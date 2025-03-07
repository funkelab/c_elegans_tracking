from warnings import warn

import napari
import napari.layers
import numpy as np
import pyqtgraph as pg
from napari.layers import Points, Shapes
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QDoubleSlider

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
        ap_pos, dist, local_minima = dist_to_spline([loc], worm_space.center_spline)
        dist = dist[0]
        local_minima = local_minima[0]
        self.dist_plot.getPlotItem().plot(ap_pos, dist)
        for min in local_minima:
            self.dist_plot.getPlotItem().plot(
                [ap_pos[min]], [dist[min]], symbol="star", color="yellow"
            )

    def _plot_widget(self) -> pg.PlotWidget:
        """
        Returns:
            pg.PlotWidget: a widget containg an (empty) plot of the distance to the center spline
        """
        gap_plot = pg.PlotWidget()
        gap_plot.setBackground((37, 41, 49))
        styles = {
            "color": "white",
        }
        gap_plot.plotItem.setLabel("left", "Distance to center spline", **styles)
        gap_plot.plotItem.setLabel("bottom", "Anterior-posterior position", **styles)
        return gap_plot


class WormSpaceWidget(QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        lattice_points: np.ndarray,
    ):
        super().__init__()
        self.viewer = viewer
        self.lattice_points = lattice_points
        layout = QVBoxLayout()
        self.ap_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.ap_slider.setRange(-1, 11)
        # self.ap_slider.setTracking(False)
        self.ap_slider.valueChanged.connect(self.compute_bases)
        layout.addWidget(QLabel("AP Position"))
        layout.addWidget(self.ap_slider)
        self.setLayout(layout)
        self._init_splines()
        self._init_basis_layers()

    def compute_bases(self):
        time = self.viewer.dims.current_step[0]
        ap = self.ap_slider.value()
        worm_space = WormSpace(self.lattice_points[time])
        self.display_basis_vectors(worm_space, ap, time)

    def _init_splines(self):
        self.splines_layer = Shapes(ndim=4, name="splines")
        self.viewer.add_layer(self.splines_layer)
        for time in range(self.lattice_points.shape[0]):
            worm_space = WormSpace(self.lattice_points[time])
            splines = [
                worm_space.center_spline,
                worm_space.left_spline,
                worm_space.right_spline,
            ]
            colors = ["white", "blue", "red"]
            paths = []
            for spline in splines:
                points = spline.interpolate(np.linspace(-1, 11, 120))
                times = np.ones(shape=(points.shape[0], 1)) * time
                points = np.hstack((times, points))
                paths.append(points)
            self.splines_layer.add_paths(paths, edge_color=colors)

    def _init_basis_layers(self):
        self.axes = Shapes(ndim=4, name="basis vectors")
        self.intersection_points = Points(
            data=[], ndim=4, face_color="red", size=10, name="intersection points"
        )
        self.viewer.add_layer(self.axes)
        self.viewer.add_layer(self.intersection_points)

    def display_basis_vectors(self, worm_space: WormSpace, ap, time):
        center_loc = worm_space.center_spline.interpolate([ap])[0]
        right_spline_loc = worm_space.right_spline.interpolate([ap])[0]
        left_spline_loc = worm_space.left_spline.interpolate([ap])[0]
        points = np.array([center_loc, left_spline_loc, right_spline_loc])

        times = np.ones(shape=(points.shape[0], 1)) * time
        points = np.hstack((times, points))

        ml_basis, dv_basis, tan_vec = worm_space.get_basis_vectors(ap)
        ml_basis = ml_basis * 100
        dv_basis = dv_basis * 100
        tan_vec = tan_vec / np.linalg.norm(tan_vec) * 100
        xaxis = np.array([[time, *center_loc], [time, *(center_loc + ml_basis)]])
        yaxis = np.array([[time, *center_loc], [time, *(center_loc + dv_basis)]])
        zaxis = np.array([[time, *center_loc], [time, *(center_loc + tan_vec)]])

        self.intersection_points.data = []
        self.intersection_points.add(points)
        self.axes.data = []
        self.axes.add_lines([xaxis, yaxis, zaxis], edge_color=["red", "purple", "green"])
