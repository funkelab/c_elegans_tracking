from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from .cubic_spline_3d import CubicSpline3D


def compute_central_spline_csv(csvfile: Path) -> CubicSpline3D:
    """Given a csv with lattice point annotations, compute the worm's central spline.
    Lattice point names are:
        ["a0", "h0", "h1", "h2", "v1", "v2", "v3", "v4", "v5", "v6", "t"]
    All other additional lattice points (e.g. those added manually to adjust the contour
    of the worm) are ignored.
    The lattice points are used to parameterize the spline, using indices from 0 to 10,
    and are represented as equally spaced along the spline parameterization.
    Note: this can read the original file format provided by the shroff lab.

    Args:
        csvfile (Path): A path to a csv file containg the lattice point locations and
            names

    Returns:
        CubicSpline3D: A 3d cubic spline object containing splines in each dimension
    """
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


def compute_central_splines(
    lattice_array_path: Path, time_range: tuple[int, int] | None = None
) -> dict[int, CubicSpline3D]:
    """Compute a spline for each time point of lattices saved in a zarr array.

    The lattice array should have dimensions (t, 11, 3, 2) for time points, the 11
    lattice points in the below order, location, and left/right sides.

    ["a0", "h0", "h1", "h2", "v1", "v2", "v3", "v4", "v5", "v6", "t"]

    Args:
        lattice_array_path (Path): the path to the zarr group with the lattice array
            (include groups in the path)
        time_range (tuple[int, int] | None, optional): Limit the returned splines to the
            given time range. Output times will start at zero. Defaults to None.

    Returns:
        dict[int, CubicSpline3D]: A dictionary from time point (starting at 0) to center
            spline objects
    """
    zarray = zarr.open(lattice_array_path)

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
