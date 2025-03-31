import json
import os
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import toml

from c_elegans_utils.tracking import SolverParams

from .dataset import Dataset
from .utils import _get_mount, _test_exists


def get_git_commit_id():
    git_hash = os.popen("git rev-parse HEAD").read().strip()  # noqa S605 S607
    return git_hash


class Experiment:
    timestamp_format = "%y%m%d_%H%M%S"
    config_file = "config.toml"
    solution_graph_file = "solution_graph.json"
    candidate_graph_file = "candidate_graph.json"
    results_file = "results.json"

    def __init__(self, name, config: dict, cluster=False, new=True, timestamp=None):
        self.name = name
        self.config = config
        self.timestamp = timestamp
        self.cluster = cluster
        self.mount_path = _get_mount("groups", cluster)
        if new:
            self._initialize_experiment()
        else:
            _test_exists(self.exp_base_dir)
        self.solver_params = SolverParams(**self.config["solver_params"])
        self._dataset = None
        self._solution_graph = None
        self._candidate_graph = None

    def _initialize_experiment(self):
        _test_exists(self.base_dir)
        self.config["version"] = get_git_commit_id()
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
    def dataset(self):
        if self._dataset is None:
            self._dataset = Dataset(**self.config["dataset"], cluster=self.cluster)
        return self._dataset

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
    def solution_graph(self) -> nx.DiGraph | None:
        if self._solution_graph is not None:
            return self._solution_graph
        path = self.exp_base_dir / self.solution_graph_file
        try:
            return self._load_graph(path)
        except AssertionError:
            return None

    @solution_graph.setter
    def solution_graph(self, graph: nx.DiGraph):
        self._save_graph(self.exp_base_dir / self.solution_graph_file, graph)
        self._solution_graph = graph

    @property
    def results(self) -> dict | None:
        json_file = self.exp_base_dir / self.results_file
        if json_file.is_file():
            with open(json_file) as f:
                results = json.load(f)
            return results
        else:
            return None

    @results.setter
    def results(self, result_dict: dict):
        json_file = self.exp_base_dir / self.results_file
        with open(json_file, "w") as f:
            json.dump(result_dict, f)

    @property
    def candidate_graph(self) -> nx.DiGraph:
        if self._candidate_graph is not None:
            return self._candidate_graph
        path = self.exp_base_dir / self.candidate_graph_file
        try:
            return self._load_graph(path)
        except AssertionError:
            return None

    @candidate_graph.setter
    def candidate_graph(self, graph: nx.DiGraph):
        self._save_graph(self.exp_base_dir / self.candidate_graph_file, graph)
        self._candidate_graph = graph

    def delete(self):
        print(f"deleting {self.uid} {self.name}")
        to_remove = [
            self.solution_graph_file,
            self.candidate_graph_file,
            self.config_file,
        ]
        for name in to_remove:
            print(f"removing {name}")
            (self.exp_base_dir / name).unlink(missing_ok=True)
        self.exp_base_dir.rmdir()
