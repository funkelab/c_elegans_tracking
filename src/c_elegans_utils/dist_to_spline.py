from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

from .compute_central_spline import CubicSpline3D


def dist_to_spline(
    target_points: ArrayLike,
    spline: CubicSpline3D,
    query_range: tuple[float, float],
    plot_path: str | Path | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Compute the distance to each point along a spline for a set of target points.
    Considers the range of possible locations along the spline to be [-1, 11],
    since the [0,10] range is the range defined by the lattice points and some cells may
    lie outside this range, but does not return minima at the endpoints of the range.

    Args:
        target_points (ArrayLike): array-like object with shape (N, 3), containing N
            three dimensional points
        spline (CubicSpline3D): The spline to compute the euclidean distance along
        plot_dir (Path | None, optional): A Path to save a matplotlib plot to.
            Defaults to None, which will not save a plot at all.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
            A 1D array of length L containing parameterized locations
            along the spline in increasing order.
            A (N, L) array containing the distances from each target_point to those
            parameterized locations.
            A tuple of 1D integer arrays containing the indices of the local minima for
            for each target point.
    """
    spacing = 0.1
    num_points = int((query_range[1] - query_range[0]) // spacing)
    cand_ap_pos = np.linspace(query_range[0], query_range[1], num=num_points)
    spline_points = spline.interpolate(cand_ap_pos)
    distances = cdist(target_points, spline_points, metric="euclidean")

    # non-maximal suppression within 5 values on either side, which with spacing .1 is
    # within one seam cell.
    def get_local_minima(arr):
        height = -1 * threshold if threshold is not None else None
        peaks, _ = find_peaks(-1 * arr, distance=5, height=height)
        peaks = list(peaks)
        # remove endpoints
        if 0 in peaks:
            peaks.remove(0)
        if len(arr) - 1 in peaks:
            peaks.remove(len(arr) - 1)
        return np.array(peaks)

    local_minima_indices = tuple(
        get_local_minima(distances[n]) for n in range(len(distances))
    )
    # local_minima_indices = argrelextrema(distances, np.less, order=5, axis=1)
    # if 0 in local_minima_indices:

    if plot_path is not None:
        save_dir = Path(plot_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        for dist, local_min_idx in zip(distances, local_minima_indices):
            plt.plot(cand_ap_pos, dist)
            plt.plot(cand_ap_pos[local_min_idx], dist[local_min_idx], marker="*")
        plt.savefig(plot_path)
        plt.close()

    return cand_ap_pos, distances, local_minima_indices
