import json
from pathlib import Path

import funlib.persistence as fp
import networkx as nx
import numpy as np
import pandas as pd
import zarr

from .graph_attrs import NodeAttr
from .utils import _test_exists


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

    choices = (
        "post_twitching_neurons",
        "lin_26_0208_Pos4",
        "lin_26_0213_Pos3",
        "lin_26_0315_Pos4",
    )

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

    def _save_array(self, store: Path, arr: np.ndarray):
        fp_array = fp.open_ds(store, mode="a")
        if self.time_range:
            fp_array[self.time_range[0] : self.time_range[1]] = arr
        else:
            fp_array[:] = arr

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
            graph = _crop_tracks(graph, self.time_range)
        return graph

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

    def _load_csv(self, csv_file: Path) -> np.ndarray:
        _test_exists(csv_file)
        df = pd.read_csv(csv_file)
        if self.time_range is not None and NodeAttr.time in df.columns:
            df = df[df[NodeAttr.time] >= self.time_range[0]]
            df = df[df[NodeAttr.time] < self.time_range[1]]
            df[NodeAttr.time] = df[NodeAttr.time] - self.time_range[0]
        return df

    def _save_csv(self, csv_file: Path, df: pd.DataFrame):
        df.to_csv(csv_file, index=False)

    @property
    def raw(self) -> np.ndarray:
        return self._load_array(self._zarr_file / self.raw_group)

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        # only include the spatial dimensions (z, y, x)
        store = self._zarr_file / self.raw_group
        _test_exists(store)
        fp_array = fp.open_ds(store)
        vs = fp_array.voxel_size
        if len(vs) == 4:
            vs = vs[1:]
        assert len(vs) == 3
        print("voxel size", vs)
        return tuple(vs)

    @property
    def seam_cell_raw(self) -> np.ndarray:
        return self._load_array(self._zarr_file / self.seam_cell_raw_group)

    @property
    def seg(self) -> np.ndarray:
        return self._load_array(self._zarr_file / self.seg_group)

    @seg.setter
    def seg(self, arr: np.ndarray):
        self._save_array(self._zarr_file / self.seg_group, arr)

    @property
    def manual_tracks(self) -> nx.DiGraph:
        return self._load_graph(self._zarr_file / self.manual_tracks_dir / "graph.json")

    @manual_tracks.setter
    def manual_tracks(self, graph: nx.DiGraph):
        self._save_graph(self._zarr_file / self.manual_tracks_dir / "graph.json", graph)

    @property
    def seam_cell_tracks(self) -> nx.DiGraph:
        return self._load_graph(
            self._zarr_file / self.seam_cell_tracks_dir / "graph.json"
        )

    @seam_cell_tracks.setter
    def seam_cell_tracks(self, graph: nx.DiGraph):
        self._save_graph(
            self._zarr_file / self.seam_cell_tracks_dir / "graph.json", graph
        )

    @property
    def lattice_points(self) -> np.ndarray:
        return self._load_lattice_points(self._zarr_file / self.lattice_points_dir)

    @property
    def seg_centers(self) -> pd.DataFrame:
        df = self._load_csv(self._zarr_file / self.seg_centers_file)
        return df

    @seg_centers.setter
    def seg_centers(self, df: pd.DataFrame):
        self._save_csv(self._zarr_file / self.seg_centers_file, df)
