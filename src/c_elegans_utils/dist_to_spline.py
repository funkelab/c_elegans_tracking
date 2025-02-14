import numpy as np
from .spline_computation import CubicSpline3D
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.typing import ArrayLike

def dist_to_spline(
        target_points: ArrayLike, spline: CubicSpline3D, plot_path: str | Path | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """Compute the distance to each point along a spline for a set of target points.
    Considers the range of possible locations along the spline to be [-1, 11],
    since the [0,10] range is the range defined by the lattice points and some cells may
    lie outside this range.

    Args:
        target_points (ArrayLike): array-like object with shape (N, 3), containing N
            three dimensional points
        spline (CubicSpline3D): The spline to compute the euclidean distance along
        plot_dir (Path | None, optional): A Path to save a matplotlib plot to.
            Defaults to None, which will not save a plot at all.

    Returns:
        tuple[np.ndarray, np.ndarray]: A 1D array of length L containing parameterized locations 
            along the spline in increasing order, and a (N, L) array containing the
            distances from each target_point to those parameterized locations.
    """
    spacing = 0.1
    query_range = [-1, 11]
    num_points = int((query_range[1] - query_range[0]) // spacing)
    cand_locations = np.linspace(query_range[0], query_range[1], num=num_points)
    spline_points = spline.interpolate(cand_locations)
    distances = cdist(target_points, spline_points, metric="euclidean")

    if plot_path is not None:
        save_dir = Path(plot_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        for dist in distances:
            plt.plot(cand_locations, dist)
        plt.savefig(plot_path)
        plt.close()

    return cand_locations, distances
    