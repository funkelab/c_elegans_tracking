import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import funlib.persistence as fp
import networkx as nx
import numpy as np
import pandas as pd
import toml
import zarr

from c_elegans_utils.tracking import SolverParams


def _test_exists(path):
    assert path.exists(), f"{path} does not exist"


def _crop_tracks(graph: nx.DiGraph, time_range):
    nodes_to_keep = [
        node
        for node, data in graph.nodes(data=True)
        if data["time"] < time_range[1] and data["time"] >= time_range[0]
    ]
    graph = graph.subgraph(nodes_to_keep)
    if time_range[0] > 0:
        for node in nodes_to_keep:
            graph.nodes[node]["time"] = graph.nodes[node]["time"] - time_range[0]
    return graph


class Dataset:
    raw_group = "raw"
    seam_cell_raw_group = "seam_cell_raw"
    seg_group = "CellPose"
    manual_tracks_dir = "manual_tracks"
    seam_cell_tracks_dir = "seam_cell_tracks"
    lattice_points_dir = "lattice_points"
    seg_centers_file = "CellPoseCenters.csv"

    choices = [
        "post_twitching_neurons",
        "lin_26_0208_Pos4",
        "lin_26_0213_Pos3",
        "lin_26_0315_Pos4",
    ]

    def __init__(self, name: str, cluster=False, time_range=None):
        if name not in self.choices:
            raise ValueError(f"Dataset {name} not in valid set")
        self.name = name
        self.cluster = cluster
        self.time_range = time_range
        self.mount_path = Path("/nrs/funke") if cluster else Path("/Volumes/funke")
        _test_exists(self._base_path)
        _test_exists(self._zarr_file)

    @property
    def _base_path(self):
        return self.mount_path / "data/lightsheet/shroff_c_elegans" / self.name

    @property
    def _zarr_file(self):
        return self._base_path / "twisted.zarr"

    def _load_array(self, store: Path) -> np.ndarray:
        _test_exists(store)
        fp_array = fp.open_ds(store)
        if self.time_range:
            arr = fp_array[self.time_range[0] : self.time_range[1]]
        else:
            arr = fp_array[:]
        return arr

    def _load_lattice_points(self, store: Path) -> np.ndarray:
        _test_exists(store)
        zarray = zarr.open(store)
        if self.time_range:
            arr = zarray[self.time_range[0] : self.time_range[1]]
        else:
            arr = zarray[:]
        return arr

    def _load_graph(self, json_file: Path) -> nx.DiGraph:
        _test_exists(json_file)
        with open(json_file) as f:
            json_graph = json.load(f)

        graph = nx.node_link_graph(json_graph, directed=True)
        if self.time_range is not None:
            graph = _crop_tracks(graph)
        return graph

    def _load_csv(self, csv_file: Path) -> np.ndarray:
        _test_exists(csv_file)
        points_df = pd.read_csv(csv_file)
        if self.time_range is not None:
            points_df = points_df[points_df["t"] >= self.time_range[0]]
            points_df = points_df[points_df["t"] < self.time_range[1]]
            points_df["t"] = points_df["t"] - self.time_range[0]
        return points_df[["t", "z", "y", "x"]].to_numpy()

    @property
    def raw(self) -> np.ndarray:
        return self._load_array(self._zarr_file / self.raw_group)

    @property
    def seam_cell_raw(self) -> np.ndarray:
        return self._load_array(self._zarr_file / self.seam_cell_raw_group)

    @property
    def seg(self) -> np.ndarray:
        return self._load_array(self._zarr_file / self.seg_group)

    @property
    def manual_tracks(self) -> nx.DiGraph:
        return self._load_graph(self.manual_tracks_dir / "graph.json")

    @property
    def seam_cell_tracks(self) -> nx.DiGraph:
        return self._load_graph(self.seam_cell_tracks_dir / "graph.json")

    @property
    def lattice_points(self) -> np.ndarray:
        return self._load_lattice_points(self._zarr_file / self.lattice_points_dir)

    @property
    def seg_centers(self) -> np.ndarray:
        return self._load_csv(self._zarr_file / self.seg_centers_file)


class Experiment:
    timestamp_format = "%y%m%d_%H%M%S"
    config_file = "config.toml"
    solution_graph_file = "solution_graph.json"

    def __init__(self, name, config: dict, cluster=False, new=True, timestamp=None):
        self.name = name
        self.config = config
        self.timestamp = timestamp
        self.cluster = cluster
        self.mount_path = Path("/groups/funke") if cluster else Path("/Volumes/funke$")
        if new:
            self._initialize_experiment()
        else:
            _test_exists(self.exp_base_dir)
        print(f"Experiment directory: {self.exp_base_dir}")
        self.dataset = Dataset(**self.config["dataset"], cluster=self.cluster)
        self.solver_params = SolverParams(**self.config["solver_params"])
        self._solution_graph = None

    def _initialize_experiment(self):
        _test_exists(self.base_dir)
        self.config["version"] = version("c_elegans_utils")
        self.timestamp = datetime.now()
        self.exp_base_dir.mkdir(exist_ok=True)
        with open(self.exp_base_dir / self.config_file, "w") as f:
            toml.dump(self.config, f)

    @classmethod
    def from_dir(cls, dir: Path, cluster=False) -> "Experiment":
        _test_exists(dir)
        stem = dir.stem
        components = stem.split("_")
        timestamp = datetime.strptime("_".join(components[0:2]), cls.timestamp_format)
        name = "_".join(components[2:])

        print(stem, components, timestamp, name)
        config_file = dir / cls.config_file
        _test_exists(config_file)
        with open(config_file) as f:
            config = toml.load(f)
        return cls(
            name=name, config=config, cluster=cluster, new=False, timestamp=timestamp
        )

    def _load_graph(self, json_file: Path) -> nx.DiGraph:
        _test_exists(json_file)
        with open(json_file) as f:
            json_graph = json.load(f)

        return nx.node_link_graph(json_graph, directed=True)

    def _save_graph(self, json_file: Path, graph: nx.DiGraph):
        graph_data = nx.node_link_data(graph)

        def convert_np_types(data):
            """Recursively convert numpy types to native Python types."""

            if isinstance(data, dict):
                return {key: convert_np_types(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [convert_np_types(item) for item in data]
            elif isinstance(data, np.ndarray):
                return data.tolist()  # Convert numpy arrays to Python lists
            elif isinstance(data, np.integer):
                return int(data)  # Convert numpy integers to Python int
            elif isinstance(data, np.floating):
                return float(data)  # Convert numpy floats to Python float
            else:
                return data  # Return the data as-is if it's already a native Python type

        graph_data = convert_np_types(graph_data)
        with open(json_file, "w") as f:
            json.dump(graph_data, f)

    @property
    def base_dir(self):
        return self.mount_path / "malinmayorc/experiments/c_elegans_tracking"

    @property
    def exp_base_dir(self):
        return self.base_dir / f"{self.uid}_{self.name}"

    @property
    def uid(self) -> str:
        return self.timestamp.strftime(self.timestamp_format)

    @property
    def solution_graph(self) -> nx.DiGraph:
        if self._solution_graph is not None:
            return self._solution_graph
        path = self.exp_base_dir / self.solution_graph_file
        try:
            return self._load_graph(path)
        except AssertionError:
            return None

    @solution_graph.setter
    def solution_graph(self, solution_graph: nx.DiGraph):
        self._save_graph(self.exp_base_dir / self.solution_graph_file, solution_graph)
        self._solution_graph = solution_graph
