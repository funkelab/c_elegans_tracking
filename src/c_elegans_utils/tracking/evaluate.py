from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from traccuracy import TrackingGraph
from traccuracy.matchers import PointMatcher
from traccuracy.metrics import BasicMetrics, TrackOverlapMetrics

from ..graph_attrs import NodeAttr

if TYPE_CHECKING:
    from ..experiment import Experiment


def convert_pos(graph, pos_key=NodeAttr.pixel_loc, location_keys=("z", "y", "x")):
    for node, data in graph.nodes(data=True):
        pixel_loc = data[pos_key]
        location_keys = ("z", "y", "x")
        for dim in range(len(location_keys)):
            graph.nodes[node][location_keys[dim]] = pixel_loc[dim]


def evaluate(exp: Experiment, threshold=80) -> dict | None:
    if exp.solution_graph is None:
        warn(f"Experiment {exp.uid} {exp.name} has no solution. Skipping evaluation.")
        return None
    soln_graph = exp.solution_graph
    convert_pos(soln_graph, pos_key="pixel_loc")
    pred_graph = TrackingGraph(
        soln_graph, frame_key=NodeAttr.time, location_keys=("z", "y", "x")
    )

    manual_graph = exp.dataset.manual_tracks
    convert_pos(manual_graph, pos_key=NodeAttr.pixel_loc)
    gt_graph = TrackingGraph(
        manual_graph, frame_key=NodeAttr.time, location_keys=("z", "y", "x")
    )

    matcher = PointMatcher(threshold=threshold)
    matched = matcher.compute_mapping(gt_graph, pred_graph)

    basic_metrics = BasicMetrics()
    basic_results = basic_metrics.compute(matched)

    overlap_metric = TrackOverlapMetrics(include_division_edges=True)
    overlap_results = overlap_metric.compute(matched)
    results = {"basic": basic_results.to_dict(), "overlap": overlap_results.to_dict()}
    return results
