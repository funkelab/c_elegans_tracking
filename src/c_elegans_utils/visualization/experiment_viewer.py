from warnings import warn

import napari
import napari.layers
import numpy as np
import pyqtgraph as pg
from napari.layers import Points, Shapes
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QMainWindow,
    QWidget,
    QHBoxLayout,
)
from superqt import QDoubleSlider
from pathlib import Path
from ..dist_to_spline import dist_to_spline
from ..worm_space import WormSpace
from ..utils import _get_mount
from ..experiment import Experiment
from functools import partial
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
        for dir in self.base_dir.iterdir():
            try:
                exps.append(Experiment.from_dir(dir, cluster=self.cluster))
            except Exception as e:
                print(f"dir {dir} likely doesn't contain an experiment, {e}")
        return exps
    
    def delete(self, uid: str):
        for experiment in self.list:
            if experiment.uid == uid:
                experiment.delete()
    

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
            time_range = QTableWidgetItem(f"{exp.dataset.time_range}")
            self.list_widget.setItem(row, 2, time_range)
            if exp.solution_graph is None:
                summary = "No solution"
            else:
                summary = (
                    f"{exp.solution_graph.number_of_nodes()} nodes, "
                    f"{exp.solution_graph.number_of_nodes()} edges"
                )
            self.list_widget.setItem(row, 3, QTableWidgetItem(summary))
            delete_button = QPushButton("delete")
            delete_button.clicked.connect(partial(self.delete_experiment, exp))
            self.list_widget.setCellWidget(row, 4, delete_button)
            view_button = QPushButton("view")
            view_button.clicked.connect(partial(view_experiment, exp))
            self.list_widget.setCellWidget(row, 5, view_button)
        self.list_widget.resizeColumnsToContents()

    def _list_experiments_widget(self) -> QWidget:
        widget = QTableWidget()
        columns = ["timestamp", "name", "time range", "summary", "delete", "view"]
        widget.setColumnCount(len(columns))
        widget.setHorizontalHeaderLabels(columns)

       
        return widget

    def delete_experiment(self, experiment: Experiment):
        try:
            experiment.delete()
            self.refresh_list()
        except ValueError as e:
            print(e)