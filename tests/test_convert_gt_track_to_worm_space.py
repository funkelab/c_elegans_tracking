import networkx as nx

from c_elegans_utils.dataset import Dataset
from c_elegans_utils.graph_attrs import NodeAttr
from c_elegans_utils.visualization.convert_gt_track_to_worm_space import (
    convert_gt_track_to_worm_space,
)
from c_elegans_utils.worm_space.worm_space import WormSpace


def test_convert_gt_track_to_worm_space():
    dataset = Dataset("lin_26_0208_Pos4", time_range=(0, 10))
    gt_graph = dataset.manual_tracks
    for track in nx.weakly_connected_components(gt_graph):
        convert_gt_track_to_worm_space(gt_graph, track, dataset)
        for node in track:
            time = gt_graph.nodes[node][NodeAttr.time]
            pixel_loc = gt_graph.nodes[node][NodeAttr.pixel_loc]
            worm_space = WormSpace(dataset.lattice_points[time])
            candidate_locs = worm_space.get_candidate_locations(pixel_loc)
            if len(candidate_locs) > 0:
                assert (
                    NodeAttr.worm_space_loc in gt_graph.nodes[node]
                ), f"node {node} has no worm position selected"
