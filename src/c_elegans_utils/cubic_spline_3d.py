from typing import Iterable

import numpy as np
from scipy.interpolate import CubicSpline


class CubicSpline3D:
    """A helper class for combining three splines, one for each dimension x y and z.

    Args:
        s (Iterable[float]): The cell "locations" along the main worm axis. Defined by
        the index of the seam cells in the following list (usually 0 to 10)
            a0 h0 h1 h2 v1 v2 v3 v4 v5 v6 t
        locations (np.ndarray): an array with shape (n, 3), where n is usually 11,
            representing the 3D locations of the points along the parameterization
            of the spline
        bc_type (str, optional): boundary condition passed to
            scipy.interpolation.CubicSpline. Defaults to "natural".
    """

    def __init__(
        self, s: Iterable[float], locations: np.ndarray, bc_type: str = "natural"
    ):
        self.splines: Iterable[CubicSpline] = [
            CubicSpline(s, locations[:, dim], bc_type=bc_type)
            for dim in range(locations.shape[1])
        ]

    def interpolate(self, indices: np.ndarray):
        """Find the point at a given index along the paramaterization of the spline
        (E.g. at that location between the lattice points 0 to 11)

        Args:
            index (float): A number between 0 and 10 representing the location along
            the worm's main axis paramaterized by the lattice points.

        Returns:
            list[float]: The [x, y, z] location of that point on the 3D spline.
        """
        return np.stack([spline(indices) for spline in self.splines], axis=1)

    def get_tan_vec(self, s: float) -> list[float]:
        # get derivative (gives you A, B, C)
        derivatives = []
        for spline in self.splines:
            der = spline.derivative()
            derivatives.append(float(der(s)))
        return derivatives

    def get_dist_along_spline(self, start, end, num_samples=20) -> float:
        sample_points = np.linspace(start, end, num=num_samples)
        sample_locs = self.interpolate(sample_points)

        o1 = sample_locs[:-1]
        o2 = sample_locs[1:]
        diff = np.abs(o1 - o2)
        norms = np.linalg.norm(diff, axis=1)
        total_dist = np.sum(norms)
        return total_dist.item()
