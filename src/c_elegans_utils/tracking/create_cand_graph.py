from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from ..graph_attrs import NodeAttr
from ..worm_space import WormSpace


def get_threshold(worm_space: WormSpace):
    max_dist = worm_space.get_max_side_spline_distance()
    return max_dist * 1.25


def create_cand_graph(
    detections: pd.DataFrame,
    lattice_points: np.ndarray,
    max_edge_distance: int,
    area_threshold: int | None = None,
) -> tuple[nx.DiGraph, list[list[int]]]:
    num_times = lattice_points.shape[0]

    cand_graph = nx.DiGraph()
    conflict_sets = []
    node_frame_dict: dict[int, list[int]] = {time: [] for time in range(num_times)}
    node_id = 1

    for time in detections[NodeAttr.time].unique():
        filtered_df = detections[detections[NodeAttr.time] == time]
        worm_space = WormSpace(lattice_points[time])
        for _, row in filtered_df.iterrows():
            row_dict = row.to_dict()
            threshold = get_threshold(worm_space)
            location = [row_dict["z"], row_dict["y"], row_dict["x"]]
            cand_list = worm_space.get_candidate_locations(
                np.array(location), threshold=threshold
            )
            # print("Candidate list", cand_list)
            conflicting = []
            for cand in cand_list:
                area = row_dict[NodeAttr.area]
                if area_threshold is None or area >= area_threshold:
                    attrs = {
                        NodeAttr.worm_space_loc: np.array(cand),
                        NodeAttr.time: time,
                        NodeAttr.detection_id: row_dict["label"],
                        NodeAttr.pixel_loc: location,
                        NodeAttr.area: area,
                        NodeAttr.mean_intensity: row_dict[NodeAttr.mean_intensity],
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


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    node_frame_dict: dict[int, list[Any]],
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        for prev_node_id, next_node_indices in zip(
            prev_node_ids, matched_indices, strict=False
        ):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree


def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> KDTree:
    """Create a kdtree with the given nodes from the candidate graph.
    Will fail if provided node ids are not in the candidate graph.

    Args:
        cand_graph (nx.DiGraph): A candidate graph
        node_ids (Iterable[Any]): The nodes within the candidate graph to
            include in the KDTree. Useful for limiting to one time frame.

    Returns:
        KDTree: A KDTree containing the positions of the given nodes.
    """
    positions = [cand_graph.nodes[node][NodeAttr.worm_space_loc] for node in node_ids]
    return KDTree(positions)
