from typing import Iterable
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
