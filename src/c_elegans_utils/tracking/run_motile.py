from __future__ import annotations

import logging

from motile import Solver, TrackGraph
from motile.constraints import ExclusiveNodes, MaxChildren, MaxParents
from motile.costs import Appear, EdgeDistance, Split
from motile_toolbox.candidate_graph import (
    NodeAttr,
    graph_to_nx,
)
from motile_toolbox.candidate_graph.graph_to_nx import graph_to_nx

from .solver_params import SolverParams

logger = logging.getLogger(__name__)


def run_motile(
    cand_graph: TrackGraph, conflict_sets: list[list[int]], solver_params: SolverParams
):
    """Construct a motile solver with the parameters specified in the solver
    params object, and then solve.

    Args:
        cand_graph (nx.DiGraph): The candidate graph to use in the solver
        solver_params (SolverParams): The costs and constraints to use in
            the solver

    Returns:
        Solver: A motile solver with the specified graph, costs, and
            constraints.
    """
    solver = Solver(TrackGraph(cand_graph, frame_attribute=NodeAttr.TIME.value))
    solver.add_constraint(MaxChildren(2))
    solver.add_constraint(MaxParents(1))
    solver.add_constraint(ExclusiveNodes(conflict_sets))

    # Using EdgeDistance instead of EdgeSelection for the constant cost because
    # the attribute is not optional for EdgeSelection (yet)
    if solver_params.edge_selection_cost is not None:
        solver.add_cost(
            EdgeDistance(
                weight=0,
                position_attribute=NodeAttr.POS.value,
                constant=solver_params.edge_selection_cost,
            ),
            name="edge_const",
        )
    if solver_params.appear_cost is not None:
        solver.add_cost(Appear(solver_params.appear_cost))
    if solver_params.division_cost is not None:
        solver.add_cost(Split(constant=solver_params.division_cost))

    if solver_params.distance_cost is not None:
        solver.add_cost(
            EdgeDistance(
                position_attribute=NodeAttr.POS.value,
                weight=solver_params.distance_cost,
            ),
            name="distance",
        )

    solver.solve(verbose=False, timeout=90)
    soln_graph = graph_to_nx(solver.get_selected_subgraph())
    return solver, soln_graph
