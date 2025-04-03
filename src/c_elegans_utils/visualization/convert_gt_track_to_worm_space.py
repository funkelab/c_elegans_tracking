import itertools
from typing import Iterable

import ilpy
import networkx as nx
import numpy as np

from c_elegans_utils import Dataset, NodeAttr
from c_elegans_utils.worm_space.worm_space import WormSpace


def get_optimal_cand_locations(
    node_id_to_cands: dict[int, list[tuple]],
) -> dict[int, tuple]:
    """Choose the optimal coordinates for each node from a set of candidates, by
    minimizing the pairwise distance of all the nodes.

    Args:
        node_id_to_cands (dict[int, list[tuple]]): A dictionary from node id to candidate
            locations

    Returns:
        dict[int, tuple]: A dictionary from node id to single locations, selected from
            the candidates
    """
    selection_variables = []
    all_candidates = []
    candidate_nodes = []
    constraints = ilpy.Constraints()
    variable_idx = 0
    for node, candidate_locs in node_id_to_cands.items():
        cand_vars = []
        constraint = ilpy.Constraint()
        for cand_loc in candidate_locs:
            variable = ilpy.Variable(f"selection_{node}", index=variable_idx)
            cand_vars.append(variable)
            constraint.set_coefficient(variable, 1)
            variable_idx += 1

            candidate_nodes.append(node)
            selection_variables.append(variable)
            all_candidates.append(np.array(cand_loc))
        constraint.set_relation(ilpy.Equal)
        constraint.set_value(1)
        constraints.add(constraint)

    objective = ilpy.expressions.Constant(0)
    for idx1, idx2 in itertools.combinations(list(range(len(all_candidates))), 2):
        var1 = selection_variables[idx1]
        var2 = selection_variables[idx2]
        cost = np.linalg.norm(all_candidates[idx1] - all_candidates[idx2])
        objective += var1 * var2 * cost
    solver = ilpy.Solver(
        num_variables=len(selection_variables), default_variable_type=ilpy.Binary
    )
    # print(objective, constraint)
    solver.set_objective(objective)
    solver.set_constraints(constraints)
    solution = solver.solve()

    soln = {}
    for idx, value in enumerate(solution):
        node_id = candidate_nodes[idx]
        candidate_loc = all_candidates[idx]
        if value > 0.5:
            assert node_id not in soln, f"node {node_id} has two cand locations selected"
            soln[node_id] = candidate_loc
    return soln


def convert_gt_track_to_worm_space(
    gt_graph: nx.DiGraph,
    nodes: Iterable[int],
    dataset: Dataset,
):
    """Find the optimal worm space coordinates for a given connected track by minimizing
    the pairwise distance in worm space. Worm coordinates will be saved in the "worm_pos"
    attribute on the graph nodes. Nodes without any candidate locations will not have
    this attribute.

    Args:
        gt_graph (nx.DiGraph): The ground truth networkx graph to be annotated
        nodes (Iterable[int]): The set of nodes contained in the track
        dataset (Dataset): The dataset, used for computin the worm spaces for each time
            point to get candidate locations
    """
    node_id_to_cands: dict[int, list] = {}
    for node in nodes:
        time = gt_graph.nodes[node][NodeAttr.time]
        pixel_loc = gt_graph.nodes[node][NodeAttr.pixel_loc]
        worm_space = WormSpace(dataset.lattice_points[time])
        candidate_locs = worm_space.get_candidate_locations(pixel_loc)
        if len(candidate_locs) > 0:
            node_id_to_cands[node] = candidate_locs

    if len(node_id_to_cands) == 0:
        return

    solution: dict[int, tuple]
    if len(node_id_to_cands) == 1:
        node = next(iter(node_id_to_cands.keys()))
        time = gt_graph.nodes[node][NodeAttr.time]
        pixel_loc = gt_graph.nodes[node][NodeAttr.pixel_loc]
        worm_space = WormSpace(dataset.lattice_points[time])
        best_loc = worm_space.get_best_candidate(pixel_loc)
        if best_loc is None:
            raise ValueError(
                "Expected gt node to have a candidate location, but found None"
            )
        else:
            solution = {node: best_loc}
    else:
        solution = get_optimal_cand_locations(node_id_to_cands)

    nx.set_node_attributes(gt_graph, values=solution, name=NodeAttr.worm_space_loc)
