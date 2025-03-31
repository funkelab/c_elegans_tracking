from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from motile_toolbox.utils.relabel_segmentation import ensure_unique_labels

from c_elegans_utils.dataset import Dataset
from c_elegans_utils.graph_attrs import NodeAttr

if TYPE_CHECKING:
    import networkx as nx


def relabel_pos(graph: nx.DiGraph):
    for node in graph.nodes():
        graph.nodes[node][NodeAttr.pixel_loc] = graph.nodes[node].pop("pos")
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=[
            "post_twitching_neurons",
            "lin_26_0208_Pos4",
            "lin_26_0213_Pos3",
            "lin_26_0315_Pos4",
        ],
    )
    parser.add_argument("-c", "--cluster", action="store_true")
    parser.add_argument(
        "function", choices=["relabel_pos", "compute_seg_centers", "relabel_seg"]
    )
    args = parser.parse_args()

    ds = Dataset(args.dataset, cluster=args.cluster)
    if args.function == "relabel_pos":
        ds.manual_tracks = relabel_pos(ds.manual_tracks)
        ds.seam_cell_tracks = relabel_pos(ds.seam_cell_tracks)
    elif args.function == "compute_seg_centers":
        ds.compute_seg_centers()
    elif args.function == "relabel_seg":
        ds.seg = ensure_unique_labels(ds.seg)
        ds.compute_seg_centers()
