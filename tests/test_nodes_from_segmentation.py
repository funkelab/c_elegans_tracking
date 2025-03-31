import numpy as np

from c_elegans_utils.graph_attrs import NodeAttr
from c_elegans_utils.tracking import nodes_from_segmentation


def test_extract_detections():
    shape = (10, 10, 10, 10)
    seg = np.zeros(shape=shape, dtype=np.uint16)
    intensity = np.random.random_sample(size=shape)
    df = nodes_from_segmentation(seg, intensity)
    assert df.size == 0

    columns = [
        NodeAttr.time,
        "z",
        "y",
        "x",
        NodeAttr.mean_intensity,
        NodeAttr.area,
        "label",
    ]
    for column in columns:
        assert column in df.columns

    seg[0, 0:10, 0:10, 0:10] = 1
    df = nodes_from_segmentation(seg, intensity)
    for column in columns:
        assert column in df.columns

    value_dict = df.iloc[0].to_dict()
    assert value_dict[NodeAttr.time] == 0
    assert value_dict["z"] == 4.5
    assert value_dict["y"] == 4.5
    assert value_dict["x"] == 4.5
    assert value_dict[NodeAttr.mean_intensity] > 0
    assert value_dict[NodeAttr.area] == 1000
    assert value_dict["label"] == 1
