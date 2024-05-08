import math
from typing import Iterable, Tuple
from metadrive.base_class.base_object import BaseObject
import numpy as np
from scipy.interpolate import CubicSpline
from object_node import transform_vec


def dot(v1: Iterable[float], v2: Iterable[float]) -> float:
    """
    Dot Product of vector v1 and v2.
    """
    assert len(v1) == len(v2), "Mismatched Dimension of v1 and v2!"
    ans = 0
    for l, r in zip(v1, v2):
        ans += l * r
    return ans


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


def l2_distance(o1: BaseObject, o2: BaseObject) -> float:
    pos_1, pos_2 = o1.position, o2.position
    pos_1, pos_2 = pos_1.tolist(), pos_2.tolist()
    return np.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)


def get_distance(pos_1: list, pos_2: list) -> float:
    return np.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)


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


def find_overlap_episodes(objects_info, obj_id1, obj_id2):
    '''
    Given two object ids, find the episodes in which both objects exist.

    :param objects_info: Dictionary of object information of the form: {obj_id: {episode: {step: obj_info}}}
    :param obj_id1: ID of the first object
    :param obj_id2: ID of the second object
    :return: List of overlapping episodes
    '''
    overlapping_episodes = []

    # Iterate through the episodes of the first object
    for episode in objects_info.get(obj_id1, {}):
        if episode in objects_info.get(obj_id2, {}):
            # Check if both objects exist at every timestep within this episode
            all_timesteps_exist = all(
                timestep in objects_info[obj_id2][episode] for timestep in objects_info[obj_id1][episode])

            if all_timesteps_exist:
                overlapping_episodes.append(episode)

    return overlapping_episodes


def find_extremities(ref_heading: Iterable[float],
                     bboxes: Iterable[Iterable], center: Iterable[float]) -> Tuple[Iterable[float], ...]:
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


def position_frontback_relative_to_obj1(obj1_heading: Iterable[float], obj1_position: Iterable[float],
                                        obj2_bbox: Iterable[Iterable]) -> str:
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


def perpendicular_vectors(vector: Iterable[float]) -> Tuple[Iterable[float], Iterable[float]]:
    """
    Given a 2D vector, return two vectors, one pointing to its left-hand direction
    and the other to its right-hand direction.
    """
    left_vector = [vector[1], -vector[0]]
    right_vector = [-vector[1], vector[0]]
    return left_vector, right_vector


def position_left_right_relative_to_obj1(obj1_heading: Iterable[float], obj1_position: Iterable[float],
                                         obj2_bbox: Iterable[Iterable]) -> str:
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


def norm(point):
    return math.sqrt(point[0] ** 2 + point[1] ** 2)


def centroid(points):
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)


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


def majority_true(things, creterion=lambda x: x, threshold=0.8):
    num_things = len(things)
    num_true = 0
    for thing in things:
        if creterion(thing):
            num_true += 1
    return num_true / num_things >= threshold


def transform_heading(v, center, heading_vector):
    # Calculate the angle of the heading vector from the X-axis
    heading_angle = np.arctan2(heading_vector[1], heading_vector[0])
    # print("ego_heading", np.rad2deg(heading_angle))
    v_angle = np.arctan2(v[1], v[0])
    # print("v_heading:", np.rad2deg(v_angle))
    angle_differential = v_angle - heading_angle
    # print("differential:", np.rad2deg(angle_differential))
    ## rotated_vector = rotate_vector(v, angle_differential)
    return [np.cos(angle_differential), np.sin(angle_differential)]  # rotated_vector


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


def sample_keypoints(original_trajectory, num_points=2, sqrt_std_max = 2):
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
