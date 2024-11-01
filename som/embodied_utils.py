import json
import numpy as np
from vqa.dataset_utils import find_extremities
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
TRAJECTORIES = json.load(open(os.path.join(current_directory,"trajectories_collection.json","r")))
PROCESSED = json.load(open(os.path.join(current_directory,"processed_trajectories.json","r")))
EGO_WORLD_POS = TRAJECTORIES["a_10_0"]["boxs"][0]
EGO_WORLD_HEADING = TRAJECTORIES["a_10_0"]["headings"][0]


class ACTION:
    TURN_LEFT=0#[0.5, 0.8],
    TURN_RIGHT=1#[-0.5, 0.8],
    SLOW_DOWN=2
    BRAKE=3#[0, -0.3],
    KEEP_STRAIGHT=4#[0,0.2]

    @classmethod
    def get_control(cls, action):
        if action == ACTION.TURN_LEFT:
            return [0.5, 0.8]
        elif action == ACTION.TURN_RIGHT:
            return [-0.5, 0.8]
        elif action == ACTION.SLOW_DOWN:
            return [0, -0.135]
        elif action == ACTION.BRAKE:
            return [0, -0.26]
        elif action == ACTION.KEEP_STRAIGHT:
            return [0, 0.15]
        else:
            raise ValueError("Unknown action: {}".format(action))

    @classmethod
    def get_action(cls, action):
        if action == ACTION.TURN_LEFT:
            return "TURN_LEFT"
        elif action == ACTION.TURN_RIGHT:
            return "TURN_RIGHT"
        elif action == ACTION.SLOW_DOWN:
            return "SLOW_DOWN"
        elif action == ACTION.BRAKE:
            return "BRAKE"
        elif action == ACTION.KEEP_STRAIGHT:
            return "KEEP_STRAIGHT"
        else:
            raise ValueError("Unknown action: {}".format(action))



def transform_trajectory(trajectory, start_pos, heading_angle):
    """
    Transforms the given trajectory to the coordinate system defined by the start position and heading direction.

    Parameters:
    - trajectory: np.ndarray of shape (N, 2), where N is the number of points, representing (x, y) positions in world coordinates.
    - start_pos: tuple or np.ndarray of shape (2,), representing the (x, y) starting position in world coordinates.
    - heading_angle: float, representing the heading direction angle in radians from the positive x-axis.

    Returns:
    - transformed_trajectory: np.ndarray of shape (N, 2), representing the transformed (x, y) positions.
    """
    # Translate the trajectory by the start position
    translated_trajectory = trajectory - np.array(start_pos)
    # Create the rotation matrix for -heading_angle
    cos_theta = np.cos(-heading_angle)
    sin_theta = np.sin(-heading_angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])
    # Apply the rotation
    transformed_trajectory = translated_trajectory.dot(rotation_matrix.T)
    return transformed_trajectory

def boxes_transform(boxes, start_pos, heading_angle):
    """
    Transforms the given
    """
    return [transform_trajectory(np.array(box), start_pos, heading_angle) for box in boxes]

def l2_distance(p1, p2):
    """
    Assuming p1 and p2 are iterable of length 2

    Returns:
    -l2-distance in np.float64
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def classify_distance(d):
    """
    Classifies the distance into one of the following categories:
    """
    if 0<=d<=2:
      return "very close"
    elif 2<d<=10:
      return "close"
    elif 10<d<=30:
      return "medium"
    elif 30<d:
      return "far"
    else:
      return "unknown"



def side(box, ego_box, ego_pos, ref_heading):
    """
        return 1 for right, -1 for left, and 0 for in the middle
    """
    normal = -ref_heading[1], ref_heading[0]
    ego_left, ego_right = find_extremities(normal, ego_box, ego_pos)
    node_bbox = box
    left_cross = []
    right_cross = []
    for node_vertice in node_bbox:
        l_vector = (node_vertice[0] - ego_left[0], node_vertice[1] - ego_left[1])
        r_vector = (node_vertice[0] - ego_right[0], node_vertice[1] - ego_right[1])
        l_cross = l_vector[0] * ref_heading[1] - l_vector[1] * ref_heading[0]
        r_cross = r_vector[0] * ref_heading[1] - r_vector[1] * ref_heading[0]
        left_cross.append(l_cross)
        right_cross.append(r_cross)
    l = True
    r = True
    for val in left_cross:
        if val > 0:
            l = False
            break
    for val in right_cross:
        if val < 0:
            r = False
            break
    if l and not r:
        return -1
    elif r and not l:
        return 1
    else:
        return 0

def longitudinal(box, ego_box, ego_pos, ref_heading):
    """
        return 1 for front, -1 for back, and 0 for in the middle
    """
    # Decide Front or Back relationships base on the bounding box of the tested object and the front/back boundary
    # of the compared object. If all vertices are in front of the front boundary, then we way the tested object
    # is in front of the compared object(and vice versa for back).
    # node w.r.t to me
    ego_front, ego_back = find_extremities(ref_heading, ego_box, ego_pos)
    node_bbox = box
    front_dot = []
    back_dot = []
    for node_vertice in node_bbox:
        f_vector = (node_vertice[0] - ego_front[0], node_vertice[1] - ego_front[1])
        b_vector = (node_vertice[0] - ego_back[0], node_vertice[1] - ego_back[1])
        f_dot = f_vector[0] * ref_heading[0] + f_vector[1] * ref_heading[1]
        b_dot = b_vector[0] * ref_heading[0] + b_vector[1] * ref_heading[1]
        front_dot.append(f_dot)
        back_dot.append(b_dot)
    f = True
    b = True
    for val in back_dot:
        if val > 0:
            b = False
            break
    for val in front_dot:
        if val < 0:
            f = False
            break
    if b and not f:
        return -1
    elif f and not b:
        return 1
    else:
        return 0

def find_center(box):
    """
    Finds the center of the given box. Assuming box is a (4,2) np.array
    """
    return np.mean(np.array(box), axis=0)


def compute_relation_string(relation) -> str:
    side = relation['side']
    front = relation['front']
    if side == -1 and front == 0:
        return 'l'
    elif side == -1 and front == -1:
        return 'lb'
    elif side == -1 and front == 1:
        return 'lf'
    elif side == 0 and front == -1:
        return 'b'
    elif side == 0 and front == 1:
        return 'f'
    elif side == 1 and front == 0:
        return 'r'
    elif side == 1 and front == -1:
        return 'rb'
    elif side == 1 and front == 1:
        return 'rf'
    else:
        return 'm'


def find_sector(box, ego_box, ego_heading):
    ego_pos = find_center(ego_box)
    relation = {
        "side": side(box, ego_box, ego_pos, ego_heading),
        "front": longitudinal(box, ego_box, ego_pos, ego_heading)
    }
    return compute_relation_string(relation)



def discretize_trajectory(bboxes, start_angle, start_heading):
    """
    Assuming bboxes is a (n, 4, 2) list in world coordinates.
    Assuming start_heading is the top-down yaw angle in radian.
    """
    starting_pos = find_center(bboxes[0])
    transformed_boxes = boxes_transform(bboxes, starting_pos, start_angle)
    centers = [find_center(box) for box in transformed_boxes]
    #print(centers)
    dists = [l2_distance(np.zeros(2), center) for center in centers]
    #print(dists)
    #exit()
    discretized_dist = [classify_distance(dist) for dist in dists]
    #print(discretized_dist)
    #
    sectors = [find_sector(box, bboxes[0], start_heading) for box in bboxes]
    #print(sectors)
    #exit(0)

    return transformed_boxes, discretized_dist, sectors



def find_end(bboxes, start_heading):
    _, dists, sectors = discretize_trajectory(bboxes, start_heading)
    return dists[-1], sectors[-1]



def interpret_actions(traj_dicts):
    #in the old record, it's action_duration_init_speed
    new_dict = {}
    for setting, record in traj_dicts.items():
        #print(setting)
        boxes, init_heading = record["boxs"], record["headings"][0]
        init_angle = np.arctan2(init_heading[1], init_heading[0])
        transformed_boxes, discrete_dists, sectors = discretize_trajectory(boxes[0], init_angle, init_heading)
        new_dict[setting] = dict(
            egocentric_boxes = [box.tolist() for box in transformed_boxes],
            discrete_pos = discrete_dists,
            sectors = sectors
        )
    json.dump(new_dict, open("processed_trajectories.json","w"), indent=2)


from collections import defaultdict
def sort_by_end(traj_dicts):
    ends = defaultdict(lambda:[])
    for setting, record in traj_dicts.items():
        ends[record["sectors"][-1]].append(setting)
    return ends

def sort_by_dist(traj_dicts):
    ends = defaultdict(lambda: [])
    for setting, record in traj_dicts.items():
        ends[record["discrete_pos"][-1]].append(setting)
    return ends





def end_with_average(traj_dicts, action, speed, duration, bucket_size=10):
    assert 0 <= speed < 50
    assert duration in [5, 10, 15, 20]
    get_speed = lambda s: float(s.split("_")[-1])
    signature = lambda s: s.split("_")
    keys = list(traj_dicts.keys())
    keys_of_interests = [key for key in keys if signature(key)[0] == action and signature(key)[1] == str(duration)]
    keys_by_speed = sorted(keys_of_interests, key = get_speed)
    #print(keys_by_speed)
    assert len(keys_by_speed)==50, f"{len(keys_by_speed)}"
    assert len(traj_dicts) % bucket_size == 0
    start_index = speed //bucket_size
    counter = defaultdict(lambda:0)
    slice = keys_by_speed[bucket_size*start_index: bucket_size*start_index + bucket_size]
    print(slice)
    for traj_key in slice:
        sector = (traj_dicts[traj_key]["discrete_pos"][-1], traj_dicts[traj_key]["sectors"][-1])
        counter[sector] += 1
    print(counter)
    return max(counter, key=counter.get)



def get_end_sector(action, speed, duration, bucket_size=10):
    processed = json.load(open("./processed_trajectories.json", "r"))
    return end_with_average(processed, action=action, speed=speed, duration=duration, bucket_size=bucket_size)


from vqa.object_node import box_overlap
from vqa.dataset_utils import transform_to_world
def determine_collisions(obj_box,  action, speed, duration, bucket_size=10, traj_dicts = PROCESSED):
    """
    obj_box in ego frame. Loading ego trajectories.

    """
    assert 0 <= speed < 50
    assert duration in [5, 10, 15, 20]
    get_speed = lambda s: float(s.split("_")[-1])
    signature = lambda s: s.split("_")
    keys = list(traj_dicts.keys())
    keys_of_interests = [key for key in keys if signature(key)[0] == action and signature(key)[1] == str(duration)]
    keys_by_speed = sorted(keys_of_interests, key=get_speed)
    assert len(keys_by_speed) == 50, f"{len(keys_by_speed)}"
    assert len(traj_dicts) % bucket_size == 0
    start_index = speed // bucket_size
    slice = keys_by_speed[bucket_size * start_index: bucket_size * start_index + bucket_size]
    box_trajectories = []
    for traj_key in slice:
        box_trajectories.append(slice[traj_key]["boxs"])
    results = []
    for timestamp, box in enumerate(box_trajectories):
        if box_overlap(box, obj_box):
            results.append(timestamp)
    if len(results) > 0:
        return True, round(sum(results)/len(results))
    else:
        return False, None












if __name__ == "__main__":
    interpret_actions(TRAJECTORIES)
    #processed = json.load(open("./processed_trajectories.json","r"))
    ##sortedd = sort_by_end(processed)
    ##json.dump(sortedd, open("stat_end.json","w"), indent=2)
    #sortedd = sort_by_dist(processed)
    #json.dump(sortedd, open("stat_dist.json", "w"), indent=2)

    #result = end_with_average(processed, action="a", speed=10, duration=15)

    #for bucket_size in [2, 5, 10]:
    #    result = end_with_average(processed, action="a", speed=10, duration=10, bucket_size=bucket_size)
    #    print(result)

