from functools import partial
from warnings import warn

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..experiment import Experiment
from ..utils import _get_mount
from .view_results_napari import view_experiment


class Experiments:
    def __init__(self, cluster=False):
        self.cluster = cluster
        self.mount_path = _get_mount("groups", cluster)

    @property
    def base_dir(self):
        return self.mount_path / "malinmayorc/experiments/c_elegans_tracking"

    @property
    def list(self) -> list[Experiment]:
        exps = []
        for _dir in self.base_dir.iterdir():
            try:
                exps.append(Experiment.from_dir(_dir, cluster=self.cluster))
            except Exception as e:
                print(f"dir {_dir} likely doesn't contain an experiment, {e}")
        return exps

    def delete(self, uid: str):
        for experiment in self.list:
            if experiment.uid == uid:
                experiment.delete()


class ResultsViewer(QWidget):
    def __init__(self, experiment: Experiment):
        super().__init__()
        print("creating results viewer")
        results = experiment.results
        if results is None:
            warn("no results for this experiment")
            return
        self.basic_results = results["basic"]["results"]
        self.basic_metrics = [
            "Total GT {}s",
            "Total Pred {}s",
            "True Positive {}s",
            "False Negative {}s",
            "False Positive {}s",
        ]
        layout = QVBoxLayout()
        node_edge_layout = QHBoxLayout()
        for element in ["Node", "Edge"]:
            node_edge_layout.addWidget(self.metric_set(element))
        node_edge_widget = QWidget()
        node_edge_widget.setLayout(node_edge_layout)
        layout.addWidget(node_edge_widget)
        self.setLayout(layout)
        self.setWindowTitle(f"Results for {experiment.name} ({experiment.dataset.name})")

    def metric_set(self, element: str) -> QWidget:
        widget = QWidget()
        layout = QGridLayout()
        for row, metric in enumerate(self.basic_metrics):
            name = metric.format(element)
            value = self.basic_results[name]
            layout.addWidget(QLabel(name), row, 0)
            layout.addWidget(QLabel(f"{value}"), row, 1)
        # widget.resizeColumnsToContents()
        widget.setLayout(layout)
        return widget


class ExperimentsViewer(QMainWindow):
    def __init__(self, cluster=False):
        super().__init__()
        self.experiments = Experiments(cluster)
        self.list_widget = self._list_experiments_widget()
        self.refresh_list()
        self.setCentralWidget(self.list_widget)
        self.setWindowState(Qt.WindowMaximized)

    def refresh_list(self):
        self.list_widget.clearContents()
        for row, exp in enumerate(self.experiments.list):
            self.list_widget.insertRow(row)
            uid = QTableWidgetItem(exp.uid)
            self.list_widget.setItem(row, 0, uid)

            name = QTableWidgetItem(exp.name)
            self.list_widget.setItem(row, 1, name)

            dataset = QTableWidgetItem(exp.dataset.name)
            self.list_widget.setItem(row, 2, dataset)

            time_range = QTableWidgetItem(f"{exp.dataset.time_range}")
            self.list_widget.setItem(row, 3, time_range)

            if exp.solution_graph is None:
                summary = "No solution"
            else:
                summary = (
                    f"{exp.solution_graph.number_of_nodes()} nodes, "
                    f"{exp.solution_graph.number_of_nodes()} edges"
                )
            self.list_widget.setItem(row, 4, QTableWidgetItem(summary))

            results = exp.results
            if results is None:
                value = "N/A"
            else:
                score = results["basic"]["results"]["Edge F1"]
                value = f"{score:0.3f}"
            edge_f1 = QTableWidgetItem(value)
            self.list_widget.setItem(row, 5, edge_f1)

            results_button = QPushButton("results")
            results_button.clicked.connect(partial(self.view_results, exp))
            if results is None:
                results_button.setEnabled(False)
            self.list_widget.setCellWidget(row, 6, results_button)

            delete_button = QPushButton("delete")
            delete_button.clicked.connect(partial(self.delete_experiment, exp))
            self.list_widget.setCellWidget(row, 7, delete_button)

            view_button = QPushButton("view twisted")
            view_button.clicked.connect(partial(view_experiment, exp, True))
            self.list_widget.setCellWidget(row, 8, view_button)

            view_button = QPushButton("view straightened")
            view_button.clicked.connect(partial(view_experiment, exp))
            self.list_widget.setCellWidget(row, 9, view_button)

        self.list_widget.resizeColumnsToContents()

    def _list_experiments_widget(self) -> QWidget:
        widget = QTableWidget()
        columns = [
            "timestamp",
            "name",
            "dataset",
            "time range",
            "summary",
            "edge F1",
            "view results",
            "delete",
            "view twisted",
            "view straightened",
        ]
        widget.setColumnCount(len(columns))
        widget.setHorizontalHeaderLabels(columns)

        return widget

    def delete_experiment(self, experiment: Experiment):
        try:
            experiment.delete()
            self.refresh_list()
        except ValueError as e:
            print(e)

    def view_results(self, experiment: Experiment):
        self.results_widget = ResultsViewer(experiment)
        self.results_widget.show()
