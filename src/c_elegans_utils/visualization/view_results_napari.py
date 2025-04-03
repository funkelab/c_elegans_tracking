import napari
import networkx as nx
from motile_tracker.data_model import SolutionTracks
from motile_tracker.data_views import TracksViewer, TreeWidget
from napari.layers import Image, Points

from c_elegans_utils import Experiment, NodeAttr

from .convert_gt_track_to_worm_space import convert_gt_track_to_worm_space


class CandidatePointsLayer(Points):
    def __init__(
        self, node_ids: list[int], nodes_by_detection: dict[int, list[int]], **kwargs
    ):
        super().__init__(**kwargs)
        self.node_ids = node_ids
        self.node_id_to_points_layer_idx = {
            node: idx for idx, node in enumerate(self.node_ids)
        }
        self.nodes_by_detection = nodes_by_detection

        self.selected_data.events.items_changed.connect(self.add_detection_candidates)

    def add_detection_candidates(self):
        selected_indices = set(self.selected_data)
        print(len(selected_indices), " selected nodes")
        selected_nodes = [self.node_ids[ix] for ix in selected_indices]
        selected_detections = {
            self.features[NodeAttr.detection_id][node] for node in selected_nodes
        }
        print(len(selected_detections), " selected detections")
        new_node_indices = []
        for det_id in selected_detections:
            det_nodes = self.nodes_by_detection[det_id]
            print("detection ", det_id, "nodes", det_nodes)
            for node in det_nodes:
                if self.node_id_to_points_layer_idx[node] not in selected_nodes:
                    new_node_indices.append(self.node_id_to_points_layer_idx[node])
        new_node_indices = set(new_node_indices)
        if len(new_node_indices) > 0:
            self.selected_data.events.items_changed.disconnect(
                self.add_detection_candidates
            )
            self.selected_data.update(new_node_indices)
            self.selected_data.events.items_changed.connect(self.add_detection_candidates)


def view_experiment(exp: Experiment, twisted=False):
    print("viewing experiment")
    viewer = napari.Viewer()

    soln_tracks_viewer = TracksViewer(viewer)
    soln_tree = TreeWidget(soln_tracks_viewer)
    viewer.window.add_dock_widget(soln_tree)
    gt_tracks_viewer = TracksViewer(viewer)
    gt_tree = TreeWidget(gt_tracks_viewer)
    viewer.window.add_dock_widget(gt_tree)

    # raw data
    if twisted:
        raw = exp.dataset.raw
        viewer.add_layer(Image(data=raw, name="raw"))

    # gt tracks
    gt_graph = exp.dataset.manual_tracks
    for nodes in nx.weakly_connected_components(gt_graph):
        convert_gt_track_to_worm_space(gt_graph, nodes, exp.dataset)
    pos_attr = NodeAttr.pixel_loc if twisted else NodeAttr.worm_space_loc

    gt_tracks = SolutionTracks(graph=gt_graph, pos_attr=pos_attr, ndim=4)
    gt_tracks_viewer.tracks_list.add_tracks(gt_tracks, "manual annotations")

    # solution tracks
    seg = exp.dataset.seg if twisted else None
    soln_graph = exp.solution_graph
    if soln_graph is not None:
        if twisted:
            # need to relabel nodes to match the seg ids
            id_mapping = {
                node: soln_graph.nodes[node][NodeAttr.detection_id]
                for node in soln_graph.nodes()
            }
            soln_graph = nx.relabel_nodes(soln_graph, id_mapping)
        solution_tracks = SolutionTracks(
            graph=soln_graph, segmentation=seg, pos_attr=pos_attr, ndim=4
        )
        soln_tracks_viewer.tracks_list.add_tracks(solution_tracks, exp.name)

    # candidate detections
    def get_location(graph, node):
        return [
            graph.nodes[node][NodeAttr.time],
            *graph.nodes[node][pos_attr],
        ]

    cand_graph = exp.candidate_graph
    node_ids = list(cand_graph.nodes())
    points = [get_location(cand_graph, node) for node in node_ids]
    features = {
        NodeAttr.detection_id: [
            cand_graph.nodes[node][NodeAttr.detection_id] for node in node_ids
        ]
    }

    nodes_by_detection: dict[int, list[int]] = {}
    for node, data in cand_graph.nodes(data=True):
        detection_id = data["detection_id"]
        if detection_id not in nodes_by_detection:
            nodes_by_detection[detection_id] = []
        nodes_by_detection[detection_id].append(node)
    cand_layer = CandidatePointsLayer(
        node_ids, nodes_by_detection, data=points, features=features, name="candidates"
    )

    viewer.add_layer(cand_layer)
    napari.run(max_loop_level=2)
