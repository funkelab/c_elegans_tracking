import argparse

import napari
import napari.layers
import numpy as np
from motile_tracker.application_menus import MenuWidget
from motile_tracker.data_model import SolutionTracks, Tracks
from motile_tracker.data_views.views.tree_view.tree_widget import TreeWidget
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from napari.layers import Shapes

from c_elegans_utils.compute_central_spline import (
    CubicSpline3D,
)
from c_elegans_utils.experiment import Dataset
from c_elegans_utils.visualization.candidate_node_widget import CandidateNodeWidget
from c_elegans_utils.visualization.worm_space_widget import (
    WormSpaceWidget,
)


def _test_exists(path):
    assert path.exists(), f"{path} does not exist"


def _crop_tracks(tracks: Tracks, time_range):
    nodes_to_keep = [
        node
        for node, data in tracks.graph.nodes(data=True)
        if data["time"] < time_range[1] and data["time"] >= time_range[0]
    ]
    tracks.graph = tracks.graph.subgraph(nodes_to_keep)
    if time_range[0] > 0:
        for node in nodes_to_keep:
            tracks.graph.nodes[node]["time"] = (
                tracks.graph.nodes[node]["time"] - time_range[0]
            )


def view_splines(splines: dict[int, CubicSpline3D]):
    layer = Shapes(ndim=4)
    times = sorted(splines.keys())
    for idx, time in enumerate(times):
        spline = splines[time]
        points = spline.interpolate(np.linspace(0, 10, 100))
        times = np.ones(shape=(points.shape[0], 1)) * idx
        points = np.hstack((times, points))
        layer.add_paths([points])
    return layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=[
            "post_twitching_neurons",
            "lin_26_0208_Pos4",
            "lin_26_0213_Pos3",
            "lin_26_0315_Pos4",
        ],
    )
    parser.add_argument("--time-range", nargs=2, type=int, default=None)
    parser.add_argument("-c", "--cluster", action="store_true")
    parser.add_argument("-s", "--straightened", action="store_true")

    parser.add_argument("--all", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--seam-cell-raw", action="store_true")
    parser.add_argument("--seg", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--seam-cell-tracks", action="store_true")
    parser.add_argument("--worm-space", action="store_true")
    parser.add_argument("--seg-centers", action="store_true")
    args = parser.parse_args()
    ds = Dataset(args.dataset, cluster=args.cluster, time_range=args.time_range)

    viewer = napari.Viewer()
    if args.raw or args.all:
        viewer.add_image(data=ds.raw, contrast_limits=(0, np.iinfo(np.uint16).max))
    if args.seam_cell_raw or args.all:
        viewer.add_image(
            data=ds.seam_cell_raw,
            contrast_limits=(0, np.iinfo(np.uint16).max),
            colormap="green",
            opacity=0.66,
        )
    if args.seg or args.all:
        viewer.add_labels(data=ds.seg, name="CellPose", blending="translucent_no_depth")

    tracks_viewer = None
    if args.manual or args.seam_cell_tracks or args.all:
        tracks_viewer = TracksViewer(viewer)
        tree_widget = TreeWidget(tracks_viewer)
        menu_widget = MenuWidget(viewer)
        viewer.window.add_dock_widget(tree_widget)

    if args.manual or args.all:
        solution_tracks = SolutionTracks(graph=ds.manual_tracks, ndim=4)
        tracks_viewer.tracks_list.add_tracks(solution_tracks, "manual_annotations")

    if args.seam_cell_tracks or args.all:
        solution_tracks = SolutionTracks(graph=ds.seam_cell_tracks, ndim=4)
        tracks_viewer.tracks_list.add_tracks(solution_tracks, "seam_cell_tracks")

    if args.worm_space or args.all:
        cand_loc_widget = CandidateNodeWidget(
            viewer, ds.lattice_points, tracks_viewer=tracks_viewer
        )
        viewer.window.add_dock_widget(cand_loc_widget)
        worm_space_widget = WormSpaceWidget(viewer, ds.lattice_points)
        viewer.window.add_dock_widget(worm_space_widget)

    if args.seg_centers or args.all:
        viewer.add_points(
            data=ds.seg_centers, name="cellpose_centers", size=5, face_color="pink"
        )

    napari.run()
