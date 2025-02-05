import os
from pathlib import Path

import numpy as np

from c_elegans_utils.spline_computation import CubicSpline3D, compute_central_spline_csv


def test_cubic_spline_3d():
    indices = np.arange(0, 11)
    locations = np.zeros(shape=(11, 3))
    spline = CubicSpline3D(indices, locations)
    assert spline.interpolate(2) == [0, 0, 0]

    for i in range(11):
        locations[i] = [i, i, i]

    spline = CubicSpline3D(indices, locations)
    assert spline.interpolate(3) == [3, 3, 3]

    func1 = lambda x: x**2 - 2 * x + 1
    func2 = lambda x: -5 * x**2 + x - 100

    for i in range(11):
        locations[i] = [func1(i), func2(i), func1(i)]

    spline = CubicSpline3D(indices, locations)
    assert spline.interpolate(1) == [func1(1), func2(1), func1(1)]


def test_get_center_spline():
    curr_path = Path(os.path.abspath(__file__))
    csv_path = curr_path.parent / "resources" / "lattice.csv"
    spline = compute_central_spline_csv(csv_path)
    mid_point_a0 = [
        (377.0514 + 370.98718) / 2,
        (120.69736 + 106.26485) / 2,
        (113.73046 + 179.51422) / 2,
    ]
    assert spline.interpolate(0) == mid_point_a0
    test_point = spline.interpolate(0.5)
    assert test_point[0] < 375
    assert test_point[0] > 340
