from pathlib import Path
import numpy as np
import napari
from motile_tracker.data_model import Tracks, SolutionTracks
from motile_tracker.application_menus import MainApp
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
import argparse
import funlib.persistence as fp
import toml
import networkx as nx
import pandas as pd

def _test_exists(path):
    assert path.exists(), f"{path} does not exist"

def _crop_tracks(tracks: Tracks, time_range):
    nodes_to_keep = [node for node, data in tracks.graph.nodes(data=True)
                     if data["time"] < time_range[1] and data["time"] >= time_range[0] ]
    tracks.graph = tracks.graph.subgraph(nodes_to_keep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--time-range", nargs=2, type=int, default=None)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--seam-cell-raw", action="store_true")
    parser.add_argument("--seg", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--seam-cell-tracks", action="store_true")
    parser.add_argument("--seg-centers", action="store_true")
    args = parser.parse_args()
    config = toml.load(args.config)["data"]
    if args.time_range is not None:
        time_range = args.time_range
    else:
        time_range = config["time_range"]
    
    base_path = Path(config["base_path"])
    _test_exists(base_path)
    zarr_file = base_path / config["zarr"]
    _test_exists(zarr_file)

    viewer = napari.Viewer()
    if args.raw:
        store = zarr_file / config["raw_group"]
        raw = fp.open_ds(store)[time_range[0]:time_range[1]]
        viewer.add_image(data=raw, contrast_limits=(0, np.iinfo(np.uint16).max))
    if args.seam_cell_raw:
        store = zarr_file / config["seam_cell_group"]
        seam_cell_raw = fp.open_ds(store)[time_range[0]:time_range[1]]
        viewer.add_image(data=seam_cell_raw, contrast_limits=(0, np.iinfo(np.uint16).max), colormap="green", opacity=.66)
    if args.seg:
        store = zarr_file / config["seg_group"]
        seg = fp.open_ds(store)[time_range[0]:time_range[1]]
        viewer.add_labels(data=seg, name="CellPose", blending="translucent_no_depth")

    
    if args.manual or args.seam_cell_tracks:
        motile_widget = MainApp(viewer)
        tracks_viewer = TracksViewer.get_instance(viewer)
        viewer.window.add_dock_widget(motile_widget)
    
    if args.manual:
        manual_dir = zarr_file / config["manual_tracks_dir"]
        _test_exists(manual_dir)
        tracks = Tracks.load(manual_dir)
        if args.time_range is not None:
            _crop_tracks(tracks, time_range)
        solution_tracks = SolutionTracks.from_tracks(tracks)
        tracks_viewer.tracks_list.add_tracks(solution_tracks, "manual_annotations")
    
    if args.seam_cell_tracks:
        seam_cell_tracks_dir = zarr_file / config["seam_cell_tracks_dir"]
        _test_exists(seam_cell_tracks_dir)
        tracks = Tracks.load(seam_cell_tracks_dir)
        if args.time_range is not None:
            _crop_tracks(tracks, time_range)
        solution_tracks = SolutionTracks.from_tracks(tracks)

        tracks_viewer.tracks_list.add_tracks(solution_tracks, "seam_cell_tracks")

    if args.seg_centers:
        seg_centers_file = zarr_file / config["seg_centers_file"]
        _test_exists(seg_centers_file)
        points_df = pd.read_csv(seg_centers_file)
        if args.time_range is not None:
            points_df = points_df[points_df["t"] >= time_range[0]]
            points_df = points_df[points_df["t"] < time_range[1]]
        points = points_df[["t", "z", "y", "x"]].to_numpy()
        viewer.add_points(data=points, name="cellpose_centers", size=5, face_color="pink")

    napari.run()