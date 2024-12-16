from pathlib import Path
import tifffile
import numpy as np
import napari
import pandas as pd
from tqdm import tqdm
import networkx as nx
from motile_plugin.data_model import SolutionTracks
from motile_plugin.application_menus import MainApp
from motile_plugin.data_views.views_coordinator.tracks_viewer import TracksViewer
import argparse
import funlib.persistence as fp
import zarr

def _test_exists(path):
    assert path.exists(), f"{path} does not exist"


if __name__ == "__main__":
    base_path = Path("/Volumes/funke/data/lightsheet/shroff_c_elegans/post_twitching_neurons/")
    _test_exists(base_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--twisted", action="store_true")
    parser.add_argument("--time-range", nargs=2, type=int, default=(11, 85))
    args = parser.parse_args()
    time_range = args.time_range
    straightened = not args.twisted
    if straightened: 
        zarr_file = base_path / "straightened.zarr"
    else:
        zarr_file = base_path / "twisted.zarr"
    _test_exists(zarr_file)
    raw = fp.open_ds(zarr_file, path="RegB")
    # raw = zarr.open(zarr_file, path="RegB")
    # if straightened:
    #     raw, seam_cell_raw, segs, manual_graph, carsen_points = load_straightened(time_range)
    # else:
    #     raw, seam_cell_raw, segs, manual_graph, carsen_points = load_twisted(time_range)
    viewer = napari.Viewer()
    viewer.add_image(data=raw[:], contrast_limits=(0, np.iinfo(np.uint16).max))
    # viewer.add_image(seam_cell_raw, name="RegA")
    # if segs is not None:
    #     viewer.add_labels(segs, "cellpose_seg")
    # viewer.add_points(data=carsen_points, name="carsen_annotations", face_color="pink", size=5)

    # motile_widget = MainApp(viewer)
    # tracks_viewer = TracksViewer.get_instance(viewer)
    # tracks_viewer.tracks_list.add_tracks(SolutionTracks(manual_graph, ndim=4), "manual_annotations")
    napari.run()