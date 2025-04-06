from __future__ import annotations

from pydantic import BaseModel, Field


class CandGraphParams(BaseModel):
    """The set of candidate graph parameters supported in the motile tracker.
    Used to build the UI as well as store parameters for runs.
    """

    max_edge_distance: float = Field(
        50.0,
        title="Max Move Distance",
        description=r"""The maximum distance an object center can move between time frames.  # noqa E501
Objects further than this cannot be matched, but making this value larger will increase solving time.""",  # noqa E501
    )
    area_threshold: float | None = Field(
        500.0,
        title="Volume Threshold",
        description=r"""Cells with volume smaller than this value will be excluded from the optimization and results.""",  # noqa E501
    )
    use_gt: bool = Field(
        False,
        title="Use Ground Truth",
        description=r"""Use ground truth detections instead of cell pose segmentations""",
    )
