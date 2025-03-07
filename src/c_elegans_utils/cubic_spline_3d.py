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

    def get_normal_plane(self, s: float) -> tuple[float, float, float, float]:
        """Get the four coordinates that define the plane normal to the curve at s.
        Ax + By + Cz + D = 0
        Args:
            s (float): The locaiton along the curve to get the plane normal to

        Returns:
            tuple[float, float, float, float]: (A, B, C, D) in the above equation. A is
                normalized to 1 (unless it is zero, then there is no normalization)
        """
        # get the center point
        center = [spline(s) for spline in self.splines]
        derivatives = self.get_tan_vec(s)
        # find D by setting dot product to 0
        d = -1 * sum([der * cen for der, cen in zip(derivatives, center)])
        derivatives.append(d)

        a = derivatives[0]
        plane_params = tuple(d / a for d in derivatives) if a != 0 else tuple(derivatives)
        return plane_params

    def get_tan_vec(self, s: float) -> list[float]:
        # get derivative (gives you A, B, C)
        derivatives = []
        for spline in self.splines:
            der = spline.derivative()
            derivatives.append(der(s))
        return derivatives
