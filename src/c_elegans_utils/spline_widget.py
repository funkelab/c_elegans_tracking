from warnings import warn

import napari
import napari.layers
import numpy as np
import pyqtgraph as pg
from napari.layers import Shapes
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
        print(point)
        data = active_layer.data[point]
        time = int(data[0])
        loc = data[1:]
        worm_space = WormSpace(self.lattice_points[time])
        cand_locs = worm_space.get_candidate_locations(loc)

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

        self.display_worm_coords(cand_locs)

    def display_worm_coords(self, worm_space: WormSpace, cand_locs, time):
        layer = Shapes(ndim=4)
        for loc in cand_locs:
            ap, ml, dv = loc
            center_loc = worm_space.center_spline.interpolate([ap])[0]
            ml_basis, dv_basis = worm_space.get_basis_vectors(ap)
            xaxis = [[time, *center_loc], [time, *(center_loc + ml_basis)]]
            yaxis = [[time, *center_loc], [time, *(center_loc + dv_basis)]]
            layer.add_lines([xaxis, yaxis], edge_color=["red", "blue"])

        self.viewer.add_layer(layer)

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
