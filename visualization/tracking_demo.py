import argparse
from pathlib import Path

import funlib.persistence as fp
import napari
import numpy as np
import pandas as pd
import toml
from motile_plugin.application_menus import MainApp
from motile_plugin.data_model import SolutionTracks, Tracks
from motile_plugin.data_views.views_coordinator.tracks_viewer import TracksViewer
from motile_plugin.motile.backend import MotileRun


def _test_exists(path):
    assert path.exists(), f"{path} does not exist"


def _crop_tracks(tracks: Tracks, time_range):
    nodes_to_keep = [
        node
        for node, data in tracks.graph.nodes(data=True)
        if data["time"] < time_range[1] and data["time"] >= time_range[0]
    ]
    tracks.graph = tracks.graph.subgraph(nodes_to_keep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--seg-centers", action="store_true")
    args = parser.parse_args()
    config = toml.load(args.config)["data"]
    time_range = (0, 10)

    base_path = Path(config["base_path"])
    _test_exists(base_path)
    zarr_file = base_path / config["zarr"]
    _test_exists(zarr_file)

    viewer = napari.Viewer()
    store = zarr_file / config["raw_group"]
    raw = fp.open_ds(store)[time_range[0] : time_range[1]]
    viewer.add_image(data=raw, contrast_limits=(0, np.iinfo(np.uint16).max))

    store = zarr_file / config["seg_group"]
    seg = fp.open_ds(store)[time_range[0] : time_range[1]]
    viewer.add_labels(data=seg, name="CellPose", blending="translucent_no_depth")

    motile_widget = MainApp(viewer)
    tracks_viewer = TracksViewer.get_instance(viewer)
    viewer.window.add_dock_widget(motile_widget)
    run = MotileRun.load(
        "/Users/malinmayorc/experiments/c_elegans/12172024_101839_demo"
    )

    if args.manual:
        manual_dir = zarr_file / config["manual_tracks_dir"]
        _test_exists(manual_dir)
        tracks = Tracks.load(manual_dir)
        if args.time_range is not None:
            _crop_tracks(tracks, time_range)
        solution_tracks = SolutionTracks.from_tracks(tracks)
        tracks_viewer.tracks_list.add_tracks(solution_tracks, "manual_annotations")

    if args.seg_centers:
        seg_centers_file = zarr_file / config["seg_centers_file"]
        _test_exists(seg_centers_file)
        points_df = pd.read_csv(seg_centers_file)
        if args.time_range is not None:
            points_df = points_df[points_df["t"] >= time_range[0]]
            points_df = points_df[points_df["t"] < time_range[1]]
        points = points_df[["t", "z", "y", "x"]].to_numpy()
        viewer.add_points(
            data=points, name="cellpose_centers", size=5, face_color="pink"
        )

    tracks_viewer.tracks_list.add_tracks(run, name=run.run_name)

    napari.run()
