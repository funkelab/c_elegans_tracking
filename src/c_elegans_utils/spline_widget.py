from warnings import warn

import napari
import napari.layers
import numpy as np
import pyqtgraph as pg
from napari.layers import Points, Shapes
from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# from .spline_widget import SplineDistanceWidget
from .worm_space import WormSpace


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
        print(selected_points)
        if len(selected_points) != 1:
            warn(
                "Please select one point in a points layer before computing spline "
                "distances",
                stacklevel=2,
            )
            return
        self.dist_plot.getPlotItem().clear()
        point = next(iter(selected_points))
        data = active_layer.data[point]
        print(f"target point {point} loc: {data}")
        time = int(data[0])
        loc = data[1:]
        worm_space = WormSpace(self.lattice_points[time])
        self.display_splines(worm_space, time)
        cand_locs = worm_space.get_candidate_locations(loc)
        print(f"candidate locations in worm space: {cand_locs}")

        # ap_pos, dist, local_minima = dist_to_spline([loc], spline)
        # dist = dist[0]
        # self.dist_plot.getPlotItem().plot(ap_pos, dist)

        # for index in local_minima:
        #     ap = ap_pos[index[0]]
        #     center_loc = spline.interpolate([ap])[0]
        #     normal_plane = spline.get_normal_plane(ap)
        #     print(f"normal plane: ", normal_plane)
        #     perp_vector = normal_plane[0:3]
        #     basis1, basis2 = self.get_basis_vectors(perp_vector)
        #     radius = 50
        #     basis1 = basis1 * radius
        #     basis2 = basis2 * radius
        #     points = np.array([
        #       center_loc - basis1,
        #       center_loc - basis2,
        #       center_loc + basis1,
        #       center_loc + basis2
        #    ])

        self.display_worm_coords(worm_space, cand_locs, time)

    def display_splines(self, worm_space: WormSpace, time: int):
        layer = Shapes(ndim=4)
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
        layer.add_paths(paths, edge_color=colors)
        self.viewer.add_layer(layer)

    def display_worm_coords(self, worm_space: WormSpace, cand_locs, time):
        shapes_layer = Shapes(ndim=4)
        points_layer = Points(data=[], ndim=4, face_color="red", size=10)
        for loc in cand_locs:
            ap, ml, dv = loc
            print(f"ap {ap}")
            center_loc = worm_space.center_spline.interpolate([ap])[0]
            right_spline_loc = worm_space.right_spline.interpolate([ap])[0]
            left_spline_loc = worm_space.left_spline.interpolate([ap])[0]
            points = np.array([center_loc, left_spline_loc, right_spline_loc])

            times = np.ones(shape=(points.shape[0], 1)) * time
            points = np.hstack((times, points))
            print(points)
            points_layer.add(points)

            ml_basis, dv_basis, tan_vec = worm_space.get_basis_vectors(ap)
            ml_basis = ml_basis * 100
            dv_basis = dv_basis * 100
            tan_vec = tan_vec / np.linalg.norm(tan_vec) * 100
            xaxis = np.array([[time, *center_loc], [time, *(center_loc + ml_basis)]])
            yaxis = np.array([[time, *center_loc], [time, *(center_loc + dv_basis)]])
            zaxis = np.array([[time, *center_loc], [time, *(center_loc + tan_vec)]])
            shapes_layer.add_lines(
                [xaxis, yaxis, zaxis], edge_color=["red", "purple", "green"]
            )
            break
        self.viewer.add_layer(shapes_layer)
        self.viewer.add_layer(points_layer)

    def _plot_widget(self) -> pg.PlotWidget:
        """
        Returns:
            pg.PlotWidget: a widget containg an (empty) plot of the solver gap
        """
        gap_plot = pg.PlotWidget()
        gap_plot.setBackground((37, 41, 49))
        styles = {
            "color": "white",
        }
        gap_plot.plotItem.setLabel("left", "Distance to center spline", **styles)
        gap_plot.plotItem.setLabel("bottom", "Anterior-posterior position", **styles)
        return gap_plot
