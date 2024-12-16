from pathlib import Path
import numpy as np
import napari
from motile_plugin.data_model import Tracks, SolutionTracks
from motile_plugin.application_menus import MainApp
from motile_plugin.data_views.views_coordinator.tracks_viewer import TracksViewer
import argparse
import funlib.persistence as fp
import toml

def _test_exists(path):
    assert path.exists(), f"{path} does not exist"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    # parser.add_argument("--time-range", nargs=2, type=int, default=None)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--seam-cell-raw", action="store_true")
    parser.add_argument("--seg", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--seam-cell-tracks", action="store_true")
    args = parser.parse_args()
    config = toml.load(args.config)["data"]
    # if args.time_range is not None:
    #     config["time_range"] = args.time_range
    
    base_path = Path(config["base_path"])
    _test_exists(base_path)
    # time_range = config["time_range"]
    zarr_file = base_path / config["zarr"]
    _test_exists(zarr_file)


    viewer = napari.Viewer()
    if args.raw:
        store = zarr_file / config["raw_group"]
        raw = fp.open_ds(store)[:]
        viewer.add_image(data=raw, contrast_limits=(0, np.iinfo(np.uint16).max))
    if args.seam_cell_raw:
        store = zarr_file / config["seam_cell_group"]
        seam_cell_raw = fp.open_ds(store)[:]
        viewer.add_image(data=seam_cell_raw, contrast_limits=(0, np.iinfo(np.uint16).max), colormap="green", opacity=.66)
    if args.seg:
        store = zarr_file / config["seg_group"]
        seg = fp.open_ds(store)[:]
        viewer.add_labels(data=seg, name="CellPose")
    if args.manual:
        manual_dir = zarr_file / config["manual_tracks_dir"]
        _test_exists(manual_dir)
        tracks = Tracks.load(manual_dir)
        solution_tracks = SolutionTracks.from_tracks(tracks)

        motile_widget = MainApp(viewer)
        tracks_viewer = TracksViewer.get_instance(viewer)
        tracks_viewer.tracks_list.add_tracks(solution_tracks, "manual_annotations")
    
    if args.seam_cell_tracks:
        seam_cell_tracks_dir = zarr_file / config["seam_cell_tracks_dir"]
        _test_exists(seam_cell_tracks_dir)
        tracks = Tracks.load(seam_cell_tracks_dir)
        solution_tracks = SolutionTracks.from_tracks(tracks)

        motile_widget = MainApp(viewer)
        tracks_viewer = TracksViewer.get_instance(viewer)
        tracks_viewer.tracks_list.add_tracks(solution_tracks, "seam_cell_tracks")


    # raw = zarr.open(zarr_file, path="RegB")
    # if straightened:
    #     raw, seam_cell_raw, segs, manual_graph, carsen_points = load_straightened(time_range)
    # else:
    #     raw, seam_cell_raw, segs, manual_graph, carsen_points = load_twisted(time_range)
    # viewer.add_image(seam_cell_raw, name="RegA")
    # if segs is not None:
    #     viewer.add_labels(segs, "cellpose_seg")
    # viewer.add_points(data=carsen_points, name="carsen_annotations", face_color="pink", size=5)

    napari.run()