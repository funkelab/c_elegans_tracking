import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import networkx as nx
import numpy as np
import toml

from c_elegans_utils.tracking import SolverParams

from .dataset import Dataset
from .utils import _test_exists


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
