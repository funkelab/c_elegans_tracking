import networkx as nx
import numpy as np
from motile import TrackGraph
from motile_toolbox.candidate_graph import NodeAttr
from motile_toolbox.candidate_graph.utils import add_cand_edges

from ..worm_space import WormSpace


def create_cand_graph(
    detections: np.ndarray, lattice_points: np.ndarray, max_edge_distance=20
) -> tuple[nx.DiGraph, list[list[int]]]:
    num_times = lattice_points.shape[0]
    worm_spaces = {time: WormSpace(lattice_points[time]) for time in range(num_times)}
    locations_by_time = {time: [] for time in range(num_times)}
    for detection in detections:
        time = detection[0]
        locations_by_time[time].append(detection[1:])

    cand_graph = nx.DiGraph()
    conflict_sets = []
    node_frame_dict = {time: [] for time in range(num_times)}
    node_id = 1

    for time, locations in locations_by_time.items():
        worm_space = worm_spaces[time]
        for location in locations:
            cand_list = worm_space.get_candidate_locations(np.array([location]))[0]
            conflicting = []
            for cand in cand_list:
                attrs = {NodeAttr.POS.value: np.array(cand), NodeAttr.TIME.value: time}
                cand_graph.add_node(
                    node_id,
                    **attrs,
                )
                conflicting.append(node_id)
                node_frame_dict[time].append(node_id)
                node_id += 1
            if len(conflicting) > 1:
                conflict_sets.append(conflicting)
    add_cand_edges(
        cand_graph, max_edge_distance=max_edge_distance, node_frame_dict=node_frame_dict
    )
    track_graph = TrackGraph(cand_graph, frame_attribute=NodeAttr.TIME.value)
    return track_graph, conflict_sets
