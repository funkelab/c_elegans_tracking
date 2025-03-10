import napari
import networkx as nx
from motile_toolbox.candidate_graph import NodeAttr
from motile_tracker.data_model import SolutionTracks
from motile_tracker.data_views import TracksViewer, TreeWidget

from ..experiment import Experiment
from .convert_gt_track_to_worm_space import convert_gt_track_to_worm_space


def view_experiment(exp: Experiment, twisted=False):
    print("viewing experiment")
    viewer = napari.Viewer()

    soln_tracks_viewer = TracksViewer(viewer)
    soln_tree = TreeWidget(soln_tracks_viewer)
    viewer.window.add_dock_widget(soln_tree)
    gt_tracks_viewer = TracksViewer(viewer)
    gt_tree = TreeWidget(gt_tracks_viewer)
    viewer.window.add_dock_widget(gt_tree)

    # gt tracks
    print("loading gt tracks")
    gt_graph = exp.dataset.manual_tracks
    print("converting gt to worm coords")
    for nodes in nx.weakly_connected_components(gt_graph):
        convert_gt_track_to_worm_space(gt_graph, nodes, exp.dataset)
    print("adding gt tracks")
    pos_attr = NodeAttr.POS.value if twisted else "worm_pos"
    gt_tracks = SolutionTracks(graph=gt_graph, pos_attr=pos_attr, ndim=4)
    gt_tracks_viewer.tracks_list.add_tracks(gt_tracks, "manual annotations")

    # solution tracks
    pos_attr = "pixel_loc" if twisted else NodeAttr.POS.value
    seg = exp.dataset.seg if twisted else None
    solution_tracks = SolutionTracks(
        graph=exp.solution_graph, segmentation=seg, pos_attr=pos_attr, ndim=4
    )
    soln_tracks_viewer.tracks_list.add_tracks(solution_tracks, exp.name)

    # candidate detections
    def get_location(graph, node):
        return [
            graph.nodes[node][NodeAttr.TIME.value],
            *graph.nodes[node][NodeAttr.POS.value],
        ]

    cand_graph = exp.candidate_graph
    node_ids = list(cand_graph.nodes())
    points = [get_location(cand_graph, node) for node in node_ids]
    node_id_to_points_layer_idx = {node: idx for idx, node in enumerate(node_ids)}
    features = {
        "detection_id": [cand_graph.nodes[node]["detection_id"] for node in node_ids]
    }
    cand_layer = napari.layers.Points(data=points, features=features, name="candidates")
    nodes_by_detection: dict[int, list[int]] = {}
    for node, data in cand_graph.nodes(data=True):
        detection_id = data["detection_id"]
        if detection_id not in nodes_by_detection:
            nodes_by_detection[detection_id] = []
        nodes_by_detection[detection_id].append(node)

    def add_detection_candidates(data):
        selected_indices = set(cand_layer.selected_data)
        print(len(selected_indices), " selected nodes")
        selected_nodes = [node_ids[ix] for ix in selected_indices]
        selected_detections = {
            cand_layer.features["detection_id"][node] for node in selected_nodes
        }
        print(len(selected_detections), " selected detections")
        new_node_indices = []
        for det_id in selected_detections:
            det_nodes = nodes_by_detection[det_id]
            print("detection ", det_id, "nodes", det_nodes)
            for node in det_nodes:
                if node_id_to_points_layer_idx[node] not in selected_nodes:
                    new_node_indices.append(node_id_to_points_layer_idx[node])
        new_node_indices = set(new_node_indices)
        if len(new_node_indices) > 0:
            cand_layer.selected_data.events.items_changed.disconnect(
                add_detection_candidates
            )
            cand_layer.selected_data.update(new_node_indices)
            cand_layer.selected_data.events.items_changed.connect(
                add_detection_candidates
            )

    cand_layer.selected_data.events.items_changed.connect(add_detection_candidates)

    viewer.add_layer(cand_layer)
    napari.run(max_loop_level=2)
