import math
from typing import Iterable, Tuple

import numpy as np


def dot(v1: Iterable[float], v2: Iterable[float]) -> float:
    """
    Dot Product of vector v1 and v2.
    """
    assert len(v1) == len(v2), "Mismatched Dimension of v1 and v2!"
    ans = 0
    for l, r in zip(v1, v2):
        ans += l * r
    return ans


def perpendicular_vectors(vector: Iterable[float]) -> Tuple[Iterable[float], Iterable[float]]:
    """
    Given a 2D vector, return two vectors, one pointing to its left-hand direction
    and the other to its right-hand direction.
    """
    left_vector = [vector[1], -vector[0]]
    right_vector = [-vector[1], vector[0]]
    return left_vector, right_vector


def norm(point):
    return math.sqrt(point[0] ** 2 + point[1] ** 2)


def centroid(points):
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)


def transform_heading(v, center, heading_vector):
    #print("From {} to {}".format(heading_vector, v))
    # Calculate the angle of the heading vector from the X-axis
    heading_angle = np.arctan2(heading_vector[1], heading_vector[0])
    # print("ego_heading", np.rad2deg(heading_angle))
    v_angle = np.arctan2(v[1], v[0])
    # print("v_heading:", np.rad2deg(v_angle))
    angle_differential = v_angle - heading_angle
    # print("differential:", np.rad2deg(angle_differential))
    ## rotated_vector = rotate_vector(v, angle_differential)
    #print(np.rad2deg(angle_differential))
    return angle_differential  # rotated_vector
