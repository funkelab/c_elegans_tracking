import math
from warnings import warn

import numpy as np

from .cubic_spline_3d import CubicSpline3D
from .dist_to_spline import dist_to_spline


class WormSpace:
    def __init__(self, lattice_points: np.ndarray):
        """A worm space definition for one time point of data.

        The worm space is computed from the annotated lattice points.
        Worm space has three axes:
            - Anterior/Posterior (AP): parameterized from 0 (the anterior lattice point)
            to 10 (the tail lattice point). Lattice points are spaced equally along the
            axis. Target points can extend beyond the endpoint values a small distance.
            - Medial/Lateral (ML): Defined by the line connecting the left and right
            lattice points, or left/right splines between the lattice points, with 0
            value at the middle location and right side being positive. The axis is
            normal to the central spline.
            - Dorsal/Ventral (DV): Defined by the line perpendicular to the ML axis and
            to the central spline. The 0 value is at the central spline, and "up"
            is positive.


        Args:
            lattice_points (np.ndarray): a numpy array with dims (11, 2, 3) for the
                11 lattice points, 2 sides, and 3 spatial dimensions of the worm
        """
        self.lattice_points = lattice_points
        right = lattice_points[:, 0]
        left = lattice_points[:, 1]
        center = (right + left) / 2

        self.valid_range = (0, lattice_points.shape[0])

        indices = list(range(*self.valid_range))

        self.right_spline = CubicSpline3D(indices, right)
        self.left_spline = CubicSpline3D(indices, left)
        self.center_spline = CubicSpline3D(indices, center)
        self.reparameterize()

    def get_candidate_locations(
        self, target_point: np.ndarray, threshold: float | None = None
    ) -> list[tuple[float, float, float]]:
        """Get the possible worm space locations for a given point in input pixel space.

        First computes the distance to the center spline along the length of the worm.
        Then finds local minima where distance is above threshold. (Warns if none exist).
        For each local minima, computes the worm space coordiantes of the point.

        # TODO: endpoints look bad, don't use this strategy past 0 and 10

        Args:
            target_point (np.ndarray): The input space location of the point
            threshold (float | None, optional): Exclude candidates further than
                threshold from the worm center spline. Defaults to None, which will
                return candidate locations at all local distance minima.

        Returns:
            list[tuple[float, float, float]]: All possible worm space coordiantes
            of the point, in order (AP, ML, DV)
        """
        ap_locs, distances, local_minima_indices = dist_to_spline(
            target_point, self.center_spline, self.valid_range, threshold=threshold
        )

        if len(local_minima_indices) == 0:
            warn(
                f"No candidate locations found for {target_point} "
                f"with threshold {threshold}.",
                stacklevel=2,
            )
            return []
        else:
            cand_ap_locs = ap_locs[local_minima_indices]
            return [self.get_worm_coords(target_point, s) for s in cand_ap_locs]

    def get_best_candidate(
        self, target_point: np.ndarray
    ) -> tuple[float, float, float] | None:
        """Get the single best candidate for a target point. Currently chooses the
        candidate with the minimum distance to the spline.

        Args:
            target_point (np.ndarray): The pixel location to be converted to worm space

        Returns:
            tuple[float, float, float] | None: The most likely worm space coordinates
                for the given pixel location, based on nearness to the center spline
        """
        ap_locs, distances, local_minima_indices = dist_to_spline(
            target_point, self.center_spline, self.valid_range, threshold=None
        )
        if len(local_minima_indices) == 0:
            warn(
                f"No candidate locations found for {target_point}.",
                stacklevel=2,
            )
            return None
        min_distances: list = list(distances[local_minima_indices])
        min_dist = min(min_distances)
        min_idx = local_minima_indices[min_distances.index(min_dist)]
        ap = ap_locs[min_idx]
        return self.get_worm_coords(target_point, ap)

    def get_worm_coords(
        self,
        target_point: tuple[float, float, float],
        ap: float,
    ) -> tuple[float, float, float]:
        """Get the worm coordinates for a given point in input space and ap axis value.
        The ap axis value must be a local minima of the distance to central curve
        function so that the input point is on the plane normal to the central curve.

        Gets the plane normal to the center spline at the AP value.
        Computes the ML basis vector on that plane (the unit vector centered at the
        center spline intersection pointing toward the right spline intersection).
        Computes the DV basis vector on that plane (normal to the ML vector, pointing up)
        Converts the input point (which is on the plane) to the new basis.


        Args:
            target_point (tuple[float, float, float]): target point in input space
            ap (float): ap value to use to compute the other two axis values. Must be
                a local minima of the distance to central spline function.

        Returns:
            tuple[float, float, float]: The worm space location of the point:
                (AP, ML, DV)
        """
        try:
            self.sanity_check(ap)
        except AssertionError as e:
            print(e)
        center_point = self.center_spline.interpolate([ap])[0]
        ml_basis, dv_basis, tan_vec = self.get_basis_vectors(ap)
        # get the vector from the center point to the target point
        target_vec = np.array(target_point) - center_point
        # convert that vector to the new basis space
        ml = np.dot(target_vec, ml_basis)
        dv = np.dot(target_vec, dv_basis)
        return ap, ml, dv

    def get_basis_vectors(self, ap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        right_point: np.ndarray = self.right_spline.interpolate([ap])[0]
        center_point: np.ndarray = self.center_spline.interpolate([ap])[0]
        tan_vec = self.center_spline.get_tan_vec(ap)
        ml_basis = center_point - right_point
        ml_basis = ml_basis / np.linalg.norm(ml_basis)
        dv_basis = np.cross(ml_basis, tan_vec)
        dv_basis = dv_basis / np.linalg.norm(dv_basis)
        return ml_basis, dv_basis, tan_vec

    def sanity_check(self, ap):
        center_point = self.center_spline.interpolate([ap])[0]

        # Make sure the three points form a line
        left_point = self.left_spline.interpolate([ap])[0]
        right_point = self.right_spline.interpolate([ap])[0]
        vec1 = left_point - center_point
        vec2 = right_point - center_point
        assert math.isclose(
            abs(np.dot(vec1, vec2)),
            abs(np.linalg.norm(vec1) * np.linalg.norm(vec2)),
            abs_tol=0.01,
        ), f"Left and right points at {ap} are not colinear with center point"

    def get_max_side_spline_distance(self):
        max_dist = 0
        for ap in np.linspace(*self.valid_range, num=25):
            center_point = self.center_spline.interpolate([ap])[0]
            right_point = self.right_spline.interpolate([ap])[0]
            dist = np.linalg.norm(center_point - right_point)
            if dist > max_dist:
                max_dist = dist
        return max_dist

    def reparameterize(self):
        right = self.lattice_points[:, 0]
        left = self.lattice_points[:, 1]
        center = (right + left) / 2

        position = 0
        indices = [0]
        for i in range(1, 11):
            distance = self.center_spline.get_dist_along_spline(i - 1, i, num_samples=25)
            position += distance
            indices.append(position)

        self.right_spline = CubicSpline3D(indices, right)
        self.left_spline = CubicSpline3D(indices, left)
        self.center_spline = CubicSpline3D(indices, center)
        self.valid_range = (0, position)
