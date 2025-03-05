import argparse
from pathlib import Path

import funlib.persistence as fp
import napari
import napari.layers
import numpy as np
import pandas as pd
import zarr
from motile_tracker.application_menus import MainApp
from motile_tracker.data_model import SolutionTracks, Tracks
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from napari.layers import Shapes

from c_elegans_utils.compute_central_spline import (
    CubicSpline3D,
    compute_central_splines,
)
from c_elegans_utils.spline_widget import WormSpaceWidget


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
    parser.add_argument("--center-spline", action="store_true")
    parser.add_argument("--seg-centers", action="store_true")
    args = parser.parse_args()

    # constant configuration
    raw_group = "raw"
    seam_cell_group = "seam_cell_raw"
    seg_group = "CellPose"
    manual_tracks_dir = "manual_tracks"
    seam_cell_tracks_dir = "seam_cell_tracks"
    lattice_points_dir = "lattice_points"
    seg_centers_file = "CellPoseCenters.csv"

    if args.time_range is not None:
        time_range = args.time_range
    else:
        time_range = None

    mount_path = Path("/nrs") if args.cluster else Path("/Volumes")
    base_path = mount_path / "funke/data/lightsheet/shroff_c_elegans" / args.dataset
    _test_exists(base_path)
    zarr_file = base_path / ("straightened.zarr" if args.straightened else "twisted.zarr")
    _test_exists(zarr_file)

    viewer = napari.Viewer()
    if args.raw or args.all:
        store = zarr_file / raw_group
        fp_array = fp.open_ds(store)
        if time_range:
            raw = fp_array[time_range[0] : time_range[1]]
        else:
            raw = fp_array[:]
        viewer.add_image(data=raw, contrast_limits=(0, np.iinfo(np.uint16).max))
    if args.seam_cell_raw or args.all:
        store = zarr_file / seam_cell_group
        fp_array = fp.open_ds(store)
        if time_range:
            seam_cell_raw = fp_array[time_range[0] : time_range[1]]
        else:
            seam_cell_raw = fp_array[:]
        viewer.add_image(
            data=seam_cell_raw,
            contrast_limits=(0, np.iinfo(np.uint16).max),
            colormap="green",
            opacity=0.66,
        )
    if args.seg or args.all:
        store = zarr_file / seg_group
        fp_array = fp.open_ds(store)
        if time_range:
            seg = fp_array[time_range[0] : time_range[1]]
        else:
            seg = fp_array[:]
        viewer.add_labels(data=seg, name="CellPose", blending="translucent_no_depth")

    if args.manual or args.seam_cell_tracks or args.all:
        motile_widget = MainApp(viewer)
        tracks_viewer = TracksViewer.get_instance(viewer)
        viewer.window.add_dock_widget(motile_widget)

    if args.manual or args.all:
        manual_dir = zarr_file / manual_tracks_dir
        _test_exists(manual_dir)
        tracks = Tracks.load(manual_dir)
        if time_range is not None:
            _crop_tracks(tracks, time_range)
        solution_tracks = SolutionTracks.from_tracks(tracks)
        tracks_viewer.tracks_list.add_tracks(solution_tracks, "manual_annotations")

    if args.seam_cell_tracks or args.all:
        seam_cell_tracks_dir = zarr_file / seam_cell_tracks_dir
        _test_exists(seam_cell_tracks_dir)
        tracks = Tracks.load(seam_cell_tracks_dir)
        if time_range is not None:
            _crop_tracks(tracks, time_range)
        solution_tracks = SolutionTracks.from_tracks(tracks)
        tracks_viewer.tracks_list.add_tracks(solution_tracks, "seam_cell_tracks")

    if args.center_spline or args.all:
        lattice_points_dir = zarr_file / lattice_points_dir
        _test_exists(lattice_points_dir)
        lattice_points = zarr.open(lattice_points_dir, "r")
        if time_range is not None:
            lattice_points = lattice_points[time_range[0] : time_range[1]]
        else:
            lattice_points = lattice_points[:]
        splines = compute_central_splines(lattice_points_dir, time_range=time_range)
        shapes_layer = view_splines(splines)
        viewer.add_layer(shapes_layer)
        spline_dist_widget = WormSpaceWidget(viewer, lattice_points)
        viewer.window.add_dock_widget(spline_dist_widget)

    if args.seg_centers or args.all:
        seg_centers_file = zarr_file / seg_centers_file
        _test_exists(seg_centers_file)
        points_df = pd.read_csv(seg_centers_file)
        if time_range is not None:
            points_df = points_df[points_df["t"] >= time_range[0]]
            points_df = points_df[points_df["t"] < time_range[1]]
            points_df["t"] = points_df["t"] - time_range[0]
        points = points_df[["t", "z", "y", "x"]].to_numpy()
        viewer.add_points(data=points, name="cellpose_centers", size=5, face_color="pink")

    napari.run()
