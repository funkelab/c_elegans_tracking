from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from tqdm import tqdm

from ..graph_attrs import NodeAttr

logger = logging.getLogger(__name__)


def nodes_from_segmentation(
    segmentation: np.ndarray,
    intensity: np.ndarray,
    scale: list[float] | None = None,
) -> pd.DataFrame:
    """Extract candidate nodes from a segmentation. Returns a data frame
    with only nodes, and also a dictionary from frames to node_ids for
    efficient edge adding.

    The df will will have the following attribute columns:
        - time
        - label
        - z, y, x
        - area
        - intensity_mean

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, [z], y, x). Labels must be unique across time, and the label
            will be used as the detection id.
        intensity (np.ndarray): A numpy array with intensity values to be passed into
            regionprops
        scale (list[float] | None, optional): The scale of the segmentation data in all
            dimensions (including time, which should have a dummy 1 value).
            Will be used to rescale the point locations and attribute computations.
            Defaults to None, which implies the data is isotropic.

    Returns:
        pd.DataFrame: A data frame with nodes and relevant attributes
    """
    logger.debug("Extracting nodes from segmentation")

    if scale is None:
        scale = [1] * (segmentation.ndim - 1)
    else:
        assert (
            len(scale) == segmentation.ndim - 1
        ), f"Scale {scale} should have {segmentation.ndim - 1} dims"

    node_dict = defaultdict(list)

    for t in tqdm(range(len(segmentation))):
        segs = segmentation[t]
        props = regionprops_table(
            segs,
            intensity_image=intensity[t],
            spacing=tuple(scale),
            properties=("area", "intensity_mean", "label", "centroid"),
        )
        num_nodes = len(props["label"])
        for column, values in props.items():
            node_dict[column].extend(values)
        if num_nodes > 0:
            node_dict[NodeAttr.time].extend([t] * num_nodes)

    if len(node_dict["label"]) == 0:
        empty_df = pd.DataFrame(
            columns=[
                NodeAttr.time,
                "z",
                "y",
                "x",
                NodeAttr.mean_intensity,
                NodeAttr.area,
                "label",
            ]
        )
        return empty_df

    # print(node_dict)
    df = pd.DataFrame(node_dict)
    df = df.rename(
        columns={
            "centroid-0": "z",
            "centroid-1": "y",
            "centroid-2": "x",
        }
    )
    return df
