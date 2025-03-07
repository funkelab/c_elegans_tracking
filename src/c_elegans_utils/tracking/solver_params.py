from __future__ import annotations

from pydantic import BaseModel, Field


class SolverParams(BaseModel):
    """The set of solver parameters supported in the motile tracker.
    Used to build the UI as well as store parameters for runs.
    """

    max_edge_distance: float = Field(
        50.0,
        title="Max Move Distance",
        description=r"""The maximum distance an object center can move between time frames.
Objects further than this cannot be matched, but making this value larger will increase solving time.""",
    )
    edge_selection_cost: float | None = Field(
        -20.0,
        title="Edge Selection",
        description=r"""Cost for selecting an edge. The more negative the value, the more edges will be selected.""",
    )
    appear_cost: float | None = Field(
        30,
        title="Appear",
        description=r"""Cost for starting a new track. A higher value means fewer and longer selected tracks.""",
    )
    division_cost: float | None = Field(
        20,
        title="Division",
        description=r"""Cost for a track dividing. A higher value means fewer divisions.
If this cost is higher than the appear cost, tracks will likely never divide.""",
    )
    distance_cost: float | None = Field(
        1,
        title="Distance",
        description=r"""Use the distance between objects as a feature for selecting edges.
The value is multiplied by the edge distance to create a cost for selecting that edge.""",
    )
