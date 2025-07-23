from typing import Iterable, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from vqa.vqagen.math_utils import dot, perpendicular_vectors


def get_distance(pos_1: list, pos_2: list) -> float:
    return np.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)


def transform_to_world(points_ego, ego_world_position, heading_vector):
    points_ego = np.array(points_ego)
    ego_world_position = np.array(ego_world_position)
    heading_vector = np.array(heading_vector)
    # Extract components of the unit vector
    u_x, u_y = heading_vector

    # Create the rotation matrix from the unit vector
    R = np.array([
        [u_x, -u_y],
        [u_y, u_x]
    ])

    # Rotate the points
    points_rotated = np.dot(R, points_ego.T).T

    # Translate the points
    points_world = points_rotated + ego_world_position

    return points_world.tolist()


def extend_bbox(bbox: Iterable[Iterable[float]], z: float) -> Iterable[Iterable[float]]:
    """
    Provided bbox of from the top-down view and the height of the bx, return a 3-d bounding box
    from bottom to the top
    """
    assert len(bbox) == 4, "Expecting 4 vertices of bbox in the xy plane!"
    result = []
    for h in (0, z):
        for box in bbox:
            result.append(box + [h])
    return result


def find_extremities(ref_heading: Iterable[float], bboxes: Iterable[Iterable], center: Iterable[float]) -> Tuple[Iterable[float], ...]:
    """
    Find the two vertice of bbox that are the extremeties along the provided positive axis.

    :param ref_heading: The positive axis. Should be heading of the ego vehicle we want to compare with
    :param bboxes: The bounding boxes of the object.
    :param center: The center of the object.
    :return: The two vertices of the bounding box that are the extremeties along the positive axis.
    """
    recentered_bboxes = [[bbox[0] - center[0], bbox[1] - center[1]] for bbox in bboxes]
    max_dot, min_dot = float("-inf"), float("inf")
    left_bbox, right_bbox = bboxes[0], bboxes[1]
    for bbox in recentered_bboxes:
        dotp = dot(bbox, ref_heading)
        if dotp > max_dot:
            left_bbox = bbox
            max_dot = dotp
        if dotp < min_dot:
            right_bbox = bbox
            min_dot = dotp
    left_bbox[0] += center[0]
    left_bbox[1] += center[1]
    right_bbox[0] += center[0]
    right_bbox[1] += center[1]

    return left_bbox, right_bbox


def position_frontback_relative_to_obj1(obj1_heading: Iterable[float], obj1_position: Iterable[float], obj2_bbox: Iterable[Iterable]) -> str:
    """
    Determine if obj2 is behind, in front of, or overlapping with obj1.

    Obj2 is behind obj1 if both of its extremities of bounding box are behind obj1's position.
    Obj2 is in front of obj1 if both of its extremities of bounding box are in front of obj1's position.
    Obj2 is overlapping with obj1 if neither of the above conditions are met.

    :param obj1_heading: The heading of obj1.
    :param obj1_position: The position of obj1.
    :param obj2_bbox: The bounding box of obj2.
    :return: "back", "front", or "overlap"
    """
    # Find extremities of obj2's bbox along obj1's heading
    obj2_extremes = find_extremities(obj1_heading, obj2_bbox, obj1_position)

    # Check the positions of obj2's extremes relative to obj1
    behind_count, front_count = 0, 0
    for point in obj2_extremes:
        relative_position = point[0] - obj1_position[0], point[1] - obj1_position[1]
        relative_position = dot(relative_position, obj1_heading)
        if relative_position < 0:
            behind_count += 1
        elif relative_position > 0:
            front_count += 1

    # Determine the relative position
    if behind_count == 2:
        return "back"
    elif front_count == 2:
        return "front"
    else:
        return "overlap"


def position_left_right_relative_to_obj1(obj1_heading: Iterable[float], obj1_position: Iterable[float], obj2_bbox: Iterable[Iterable]) -> str:
    """
    Determine if obj2 is to the left or right of obj1.

    Obj2 is to the left of obj1 if both of its extremities of bounding box are to the left of obj1's heading.
    Obj2 is to the right of obj1 if both of its extremities of bounding box are to the right of obj1's heading.
    Obj2 is overlapping with obj1 if neither of the above conditions are met.

    :param obj1_heading: The heading of obj1.
    :param obj1_position: The position of obj1.
    :param obj2_bbox: The bounding box of obj2.
    :return: "left", "right", or "overlap"
    """
    # Calculate the perpendicular vectors for obj1's heading
    left_vector, right_vector = perpendicular_vectors(obj1_heading)

    # Find extremities of obj2's bbox along the left and right vectors
    left_extreme, right_extreme = find_extremities(left_vector, obj2_bbox, obj1_position)

    # Check the positions of obj2's extremes relative to obj1
    left_count, right_count = 0, 0
    for point in [left_extreme, right_extreme]:
        relative_point = point[0] - obj1_position[0], point[1] - obj1_position[1]
        relative_position_left = dot(relative_point, left_vector)
        if relative_position_left > 0:
            left_count += 1
        else:
            right_count += 1

    # Determine the relative position
    if left_count == 2:
        return "left"
    elif right_count == 2:
        return "right"
    else:
        return "overlap"


def order_points_clockwise(points):
    """
    Take in a list of (n,2)
    """
    points = np.array(points)
    center = np.mean(points, axis=0)
    arctan2 = lambda s, c: angle if (angle := np.arctan2(s, c)) >= 0 else 2 * np.pi + angle

    def clockwise_around_center(point):
        diff = point - center
        rcos = np.dot(diff, center)
        rsin = np.cross(diff, center)
        return arctan2(rsin, rcos)

    sorted_points = sorted(points, key=clockwise_around_center)
    return [list(point) for point in sorted_points]


def generate_smooth_spline(waypoints, num_points=100):
    """
    Generate smooth splines through the given waypoints.

    :param waypoints: List of (x, y) tuples representing waypoints.
    :param num_points: Number of points to sample along the spline.
    :return: x and y coordinates of the sampled points.
    """
    waypoints = np.array(waypoints)
    x = waypoints[:, 0]
    y = waypoints[:, 1]

    # Fit cubic splines
    cs_x = CubicSpline(np.arange(len(x)), x, bc_type='natural')
    cs_y = CubicSpline(np.arange(len(y)), y, bc_type='natural')

    # Generate new points
    t = np.linspace(0, len(waypoints) - 1, num_points)
    x_spline = cs_x(t)
    y_spline = cs_y(t)

    return x_spline, y_spline


def sample_keypoints(original_trajectory, num_points=2, sqrt_std_max=2):
    """
    Will generate a smooth trajectory that lands roughly at the original end.
    """
    size = len(original_trajectory)
    freq = int(np.floor(size / num_points))
    waypoints = original_trajectory[::freq]
    last = original_trajectory[-1]
    last = last[np.newaxis, ...]
    waypoints = np.vstack([waypoints, last])
    num_points = len(waypoints)
    noise = np.zeros_like(waypoints)
    for i in range(num_points):
        std = np.sqrt(sqrt_std_max * (i / (num_points - 1)))
        noise[i, :] = np.random.normal(0, std, (1, 2))
    waypoints += noise
    new_trajectory = generate_smooth_spline(waypoints, size)
    return new_trajectory


def generate_stopped_trajectory(stop_step, original_trajectory):
    leftover = len(original_trajectory) - stop_step
    tails = [original_trajectory[stop_step]] * (leftover - 1)
    return original_trajectory[:stop_step + 1] + tails


def extrapolate_bounding_boxes(centers, initial_angle, initial_box):
    bounding_boxes = [initial_box]
    coordinate = centers[0], initial_angle
    for i, waypoint in enumerate(centers[1:]):
        origin, heading = coordinate
        # find the rotated angle
        translation_vector = waypoint[0] - origin[0], waypoint[1] - origin[1]
        diff = transform_vec(origin, [np.cos(heading), np.sin(heading)], [waypoint])[0]
        # print(np.rad2deg(np.arctan2(diff[1], )))
        delta_angle = np.arctan2(diff[1], diff[0])
        cur_box = bounding_boxes[-1]
        # rotate_box
        rotated_box = rotate_box(cur_box, origin, delta_angle)
        # translate_box
        translated_box = translate_box(rotated_box, translation_vector)
        bounding_boxes.append(translated_box)
        # update coordinate
        coordinate = waypoint, coordinate[1] + delta_angle
    return bounding_boxes


def translate_box(box, offset):
    new_box = []
    for vertice in box:
        x, y = vertice
        new_x = x + offset[0]
        new_y = y + offset[1]
        new_box.append([new_x, new_y])
    return new_box


def rotate_box(vertices, origin, angle):
    """
      rotate clockwise by angle around origin.
      Vertices in world cooridnate.
      Origin in world coordinate.
      Return in world coordinate.
      """
    return [rotate_point(np.array(vertex), np.array(origin), -angle) for vertex in vertices]


def rotate_point(point, origin, angle):
    """
    rotate clockwise by angle around origin
    """
    relative_point = point - origin
    relative_point_rotated = np.dot(relative_point,
                                    np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
    return (relative_point_rotated + origin).tolist()


def box_trajectories_collide(bboxes1, bboxes2):
    """
    Given two box trajectories, determine if collision will ever happen. This requires two boxes intersect at some synchronized timestamp.
    """
    for box1, box2 in zip(bboxes1, bboxes2):
        if box_overlap(box1, box2):
            return True
    return False


def box_trajectories_intersect(bboxes1, bboxes2):
    """
    Given two box trajectories, determine if the trajectories will ever intersect/overlap.
    """
    records = []
    for id1, box1 in enumerate(bboxes1):
        for id2, box2 in enumerate(bboxes2):
            if box_overlap(box1, box2):
                records.append((id1, id2))
    records = sorted(records, key=lambda x: min(x))
    if len(records) > 0:
        """if records[0][0] < records[0][1]:
            print("2 run into 1")
        else:
            print("1 run into 2")"""
        return True
    else:
        return False


def box_overlap(box1, box2):
    """
    Given two boxes, determine if they overlap.
    """
    def project_polygon(axis, polygon):
        """ Project a polygon onto an axis and return the minimum and maximum points of projection. """
        dots = [np.dot(vertex, axis) for vertex in polygon]
        return min(dots), max(dots)

    def overlap_on_axis(axis, poly1, poly2):
        """ Check if projections of two polygons on a given axis overlap. """
        min1, max1 = project_polygon(axis, poly1)
        min2, max2 = project_polygon(axis, poly2)
        return max(min1, min2) <= min(max1, max2)

    def separating_axis_theorem(poly1, poly2):
        """ Use the Separating Axis Theorem to determine if two polygons overlap. """
        # Get the list of axes to check
        num_vertices = len(poly1)
        axes = [np.subtract(poly1[(i + 1) % num_vertices], poly1[i]) for i in range(num_vertices)]
        num_vertices = len(poly2)
        axes += [np.subtract(poly2[(i + 1) % num_vertices], poly2[i]) for i in range(num_vertices)]

        # Normalize the axes
        axes = [np.array([-axis[1], axis[0]]) for axis in axes]  # Perpendicular vector

        # Check overlap for all axes
        for axis in axes:
            if not overlap_on_axis(axis, poly1, poly2):
                return False  # Found a separating axis
        return True  # No separating axis found, polygons must overlap

    return separating_axis_theorem(np.array(box1), np.array(box2))


def transform_vec(origin, positive_x, bbox):
    def change_bases(x, y):
        relative_x, relative_y = x - origin[0], y - origin[1]
        new_x = positive_x
        new_y = (-new_x[1], new_x[0])
        x = (relative_x * new_x[0] + relative_y * new_x[1])
        y = (relative_x * new_y[0] + relative_y * new_y[1])
        return [x, y]

    return [change_bases(*point) for point in bbox]
