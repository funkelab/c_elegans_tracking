from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import zarr
from scipy.interpolate import CubicSpline


class CubicSpline3D:
    def __init__(self, s: Iterable, locations: np.ndarray, bc_type: str = "natural"):
        """A helper class for combining three splines, one for each dimension x y and z.

        Args:
            s (Iterable): The cell "locations" along the main worm axis. Defined by the
            index of the seam cells in the following list (usually 0 to 10)
                a0 h0 h1 h2 v1 v2 v3 v4 v5 v6 t
            locations (np.ndarray): an array with shape (n, 3), where n is usually 11,
                representing the 3D locations of the points along the parameterization
                of the spline
            bc_type (str, optional): boundary condition passed to scipy.interpolation.CubicSpline.
                Defaults to "natural".
        """
        self.splines: Iterable[CubicSpline] = [
            CubicSpline(s, locations[:, dim], bc_type=bc_type)
            for dim in range(locations.shape[1])
        ]

    def interpolate(self, index: float):
        """Find the point at a given index along the paramaterization of the spline
        (E.g. at that location between the lattice points 0 to 11)

        Args:
            index (float): A number between 0 and 10 representing the location along
            the worm's main axis paramaterized by the lattice points.

        Returns:
            list[float]: The [x, y, z] location of that point on the 3D spline.
        """
        return [spline([index])[0] for spline in self.splines]


def compute_central_spline_csv(csvfile: Path):
    df = pd.read_csv(csvfile, index_col="name")
    df.index = df.index.str.lower()
    names = ["a0", "h0", "h1", "h2", "v1", "v2", "v3", "v4", "v5", "v6", "t"]
    centers = []
    for name in names:
        right = _get_loc(df, name + "R")
        left = _get_loc(df, name + "L")
        center = [(right[d] + left[d]) / 2 for d in range(3)]
        centers.append(center)

    indices = list(range(len(names)))
    return CubicSpline3D(indices, np.array(centers))


def _get_loc(df: pd.DataFrame, name: str):
    row = df.loc[name.lower()]
    return row["x_voxels"], row["y_voxels"], row["z_voxels"]


def compute_central_spline(
    lattice_array_path: Path, time_range=None
) -> dict[int, CubicSpline3D]:
    zarray = zarr.open(lattice_array_path)

    # ["a0", "h0", "h1", "h2", "v1", "v2", "v3", "v4", "v5", "v6", "t"]
    names = zarray.attrs["lattice_point_names"]
    if time_range is not None:
        data = zarray[time_range[0] : time_range[1]]
    else:
        data = zarray[:]

    splines = {}
    for time, locations in enumerate(data):
        centers = (locations[:, 0] + locations[:, 1]) / 2
        indices = list(range(len(names)))
        splines[time] = CubicSpline3D(indices, centers)
    return splines
