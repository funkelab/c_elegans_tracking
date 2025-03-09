import napari
import napari.layers
import numpy as np
from napari.layers import Points, Shapes
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)
from superqt import QDoubleSlider

from ..worm_space import WormSpace


class WormSpaceWidget(QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        lattice_points: np.ndarray,
    ):
        super().__init__()
        self.viewer = viewer
        self.lattice_points = lattice_points
        self.time = self.viewer.dims.current_step[0]
        self.worm_space = WormSpace(lattice_points[self.time])
        layout = QVBoxLayout()
        self.ap_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.ap_slider.setRange(
            *self.worm_space.valid_range
        )  # self.ap_slider.setTracking(False)
        self.ap_slider.valueChanged.connect(self.compute_bases)
        self.viewer.dims.events.current_step.connect(self.change_step)
        layout.addWidget(QLabel("AP Position"))
        layout.addWidget(self.ap_slider)
        self.setLayout(layout)
        self._init_splines()
        self._init_basis_layers()

    def change_step(self):
        self.time = self.viewer.dims.current_step[0]
        self.worm_space = WormSpace(self.lattice_points[self.time])
        self.ap_slider.setRange(*self.worm_space.valid_range)

    def compute_bases(self):
        time = self.viewer.dims.current_step[0]
        ap = self.ap_slider.value()
        # worm_space = WormSpace(self.lattice_points[time])
        self.display_basis_vectors(self.worm_space, ap, time)

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
                points = spline.interpolate(np.linspace(*worm_space.valid_range, 120))
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
