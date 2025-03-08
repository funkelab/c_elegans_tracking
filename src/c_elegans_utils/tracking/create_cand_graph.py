import networkx as nx
import numpy as np
from motile_toolbox.candidate_graph import NodeAttr
from motile_toolbox.candidate_graph.utils import add_cand_edges

from ..worm_space import WormSpace


def create_cand_graph(
    detections: np.ndarray, lattice_points: np.ndarray, max_edge_distance: int,
) -> tuple[nx.DiGraph, list[list[int]]]:
    num_times = lattice_points.shape[0]
    locations_by_time = {time: [] for time in range(num_times)}
    for detection_id, detection in enumerate(detections):
        time = detection[0]
        locations_by_time[time].append((detection_id, detection[1:]))

    cand_graph = nx.DiGraph()
    conflict_sets = []
    node_frame_dict = {time: [] for time in range(num_times)}
    node_id = 1

    for time, locations in locations_by_time.items():
        worm_space = WormSpace(lattice_points[time])
        max_dist = worm_space.get_max_side_spline_distance()
        print(f"max spline distance in time {time}: {max_dist}")
        for detection_id, location in locations:
            cand_list = worm_space.get_candidate_locations(
                np.array([location]), threshold=max_dist * 1.5
            )[0]
            # print("Candidate list", cand_list)
            conflicting = []
            for cand in cand_list:
                attrs = {
                    NodeAttr.POS.value: np.array(cand),
                    NodeAttr.TIME.value: time,
                    "detection_id": detection_id,
                }
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
    return cand_graph, conflict_sets
