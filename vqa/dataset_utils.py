from typing import Iterable, Tuple
import numpy as np
from metadrive.base_class.base_object import BaseObject

def dot(v1: Iterable[float], v2:Iterable[float])->float:
    """
    Dot Product of vector v1 and v2.
    """
    assert len(v1) == len(v2), "Mismatched Dimension of v1 and v2!"
    ans = 0
    for l,r in zip(v1,v2):
        ans += l*r
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

def l2_distance(o1:BaseObject, o2:BaseObject) -> float:
    pos_1,pos_2 = o1.position, o2.position
    pos_1, pos_2 = pos_1.tolist(), pos_2.tolist()
    return np.sqrt((pos_1[0]-pos_2[0])**2 + (pos_1[1]-pos_2[1])**2)

def get_distance(pos_1:list, pos_2:list) -> float:
    return np.sqrt((pos_1[0]-pos_2[0])**2 + (pos_1[1]-pos_2[1])**2)

def extend_bbox(bbox: Iterable[Iterable[float]], z: float)->Iterable[Iterable[float]]:
    """
    Provided bbox of from the top-down view and the height of the bx, return a 3-d bounding box
    from bottom to the top
    """
    assert len(bbox) == 4, "Expecting 4 vertices of bbox in the xy plane!"
    result = []
    for h in (0,z):
        for box in bbox:
            result.append(box+[h])
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
        relative_position = dot(point, obj1_heading)
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
        relative_position_left = dot(point[0], left_vector)
        relative_position_right = dot(point[1], right_vector)
        if relative_position_left > 0:
            left_count += 1
        if relative_position_right > 0:
            right_count += 1

    # Determine the relative position
    if left_count == 2:
        return "left"
    elif right_count == 2:
        return "right"
    else:
        return "overlap"