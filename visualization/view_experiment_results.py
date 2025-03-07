import argparse
from pathlib import Path

import napari
from motile_tracker.application_menus import MainApp
from motile_tracker.data_model import SolutionTracks
from motile_tracker.data_views import TracksViewer

from c_elegans_utils.experiment import Experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--time-range", nargs=2, type=int, default=None)
    parser.add_argument("-c", "--cluster", action="store_true")
    args = parser.parse_args()

    mount_path = Path("/groups/funke") if args.cluster else Path("/Volumes/funke$")
    base_path = mount_path / "malinmayorc/experiments/c_elegans_tracking"

    exp_dir = base_path / args.exp_dir
    exp = Experiment.from_dir(exp_dir, cluster=args.cluster)

    solution_tracks = SolutionTracks(graph=exp.solution_graph, ndim=4)

    viewer = napari.Viewer()

    motile_widget = MainApp(viewer)
    tracks_viewer = TracksViewer.get_instance(viewer)
    viewer.window.add_dock_widget(motile_widget)
    tracks_viewer.tracks_list.add_tracks(solution_tracks, exp.name)
    napari.run()
