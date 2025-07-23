from collections import defaultdict
from typing import Iterable, Tuple, List, Union

import numpy as np

from vqa.utils.common_utils import majority_true
from vqa.vqagen.math_utils import dot, norm
from vqa.vqagen.geometric_utils import get_distance, find_extremities, position_frontback_relative_to_obj1, position_left_right_relative_to_obj1


class ObjectNode:
    def __init__(self,
                 pos,
                 color,
                 speed,
                 heading,
                 id,
                 bbox,
                 type,
                 height,
                 observing_cameras,
                 visible_criteria,
                 collisions):
        '''
        Apparently I need more comments
        '''
        # More properties could be defined.
        self.pos = pos  # (x,y) in world coordinate
        self.color = color  # as annotated
        self.speed = speed  # in m/s
        self.heading = heading  # (dx, dy), in world coordinate
        self.id = id  # as defined in the metadrive env
        self.bbox = bbox  # bounding box with
        self.type = type  # as annotated
        self.height = height  # The height retrieved from the asset's convex hull.
        self.observing_cameras = observing_cameras
        self.visible = visible_criteria(observing_cameras)
        self.collision = collisions

    def compute_relation(self, node, ref_heading: Iterable[float]) -> dict:
        """
        node: AgentNode. The node you wish to examine its relation w.r.t. to this node
        ref_heading: The direction of the +x-axis of the coordinate you wish to examine the spatial relationships, expressed in world coordinate.
        Encode spatial relation with ints.
        Indicators:
            Left  | Right | colinear:  -1 | 1 | 0
            Back  | Front | colinear:  -1 | 1 | 0
        """
        assert node is not None, "node is None in agent_node.AgentNode.compute_relation"
        relation = {
            'side': self.leftORright(node, ref_heading),
            'front': self.frontORback(node, ref_heading),
        }
        return relation

    def compute_relation_string(self, node, ref_heading: Iterable[float]) -> str:
        assert node is not None, "node is None in agent_node.AgentNode.compute_relation_string"
        relation = self.compute_relation(node, ref_heading)
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

    def __str__(self) -> str:
        dictionary = {
            'pos': self.pos,
            'color': self.color,
            'speed': self.speed,
            'heading': self.heading,
            'type': self.type,
            "id": self.id,
            "visible": self.visible
        }
        return dictionary.__str__()

    def leftORright(self, node, ref_heading: Iterable[float]) -> int:
        """
        return 1 for right, -1 for left, and 0 for in the middle
        """
        # Decide Left or Right relationships base on the bounding box of the tested object and the left/right boundary
        # of the compared object. If all vertices are to the left of the front boundary, then we way the tested object
        # is to the left of the compared object(and vice versa for right).
        # node w.r.t to me
        # ref_heading is the direction of ego_vehicle.This direction may not be the heading for me
        normal = -ref_heading[1], ref_heading[0]
        ego_left, ego_right = find_extremities(normal, self.bbox, self.pos)
        node_bbox = node.bbox
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

    def frontORback(self, node, ref_heading: Iterable[float]) -> int:
        """
        return 1 for front, -1 for back, and 0 for in the middle
        """
        # Decide Front or Back relationships base on the bounding box of the tested object and the front/back boundary
        # of the compared object. If all vertices are in front of the front boundary, then we way the tested object
        # is in front of the compared object(and vice versa for back).
        # node w.r.t to me

        ego_front, ego_back = find_extremities(ref_heading, self.bbox, self.pos)
        node_bbox = node.bbox
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


class TemporalNode:
    def __init__(self, now_frame, id, type, height, positions, color, speeds, headings, bboxes, observing_cameras,
                 collisions, interaction=None):
        # time-invariant properties
        self.now_frame = now_frame  # 0-indexed.indicate when is "now". The past is history and inclusive of now.
        self.id = id  # as defined in the metadrive env
        self.type = type  # as annotated
        self.height = height  # The height retrieved from the asset's convex hull.
        self.color = color  # as annotated

        # time-dependent properties
        self.positions = positions  # (x,y) in world coordinate
        self.speeds = speeds  # in m/s
        self.headings = headings  # (dx, dy), in world coordinate
        self.bboxes = bboxes  # bounding box with
        self.observing_cameras = observing_cameras  # record which ego camera observed the object.
        self.collisions = collisions  # record collisions(if any) along time.
        if interaction is not None:
            self.interactions = interaction
        else:
            self.interactions = defaultdict(list)  # interaction record is (other_node, action_name)
        self.actions = self.summarize_action(self.now_frame)  # intrinsic actions that can be summarized independently.

    def future_positions(self, now_frame, pred_frames=None):
        if pred_frames is None:
            return self.positions[now_frame + 1:]
        else:
            return self.positions[now_frame + 1: now_frame + 1 + pred_frames]

    def future_bboxes(self, now_frame, pred_frames=None):
        if pred_frames is None:
            return self.bboxes[now_frame + 1:]
        else:
            return self.bboxes[now_frame + 1: now_frame + 1 + pred_frames]

    def summarize_action(self, now_frame) -> list[str]:
        """
        Apparently, in real envs, we don't have any second-order information. We only have speed.
        So, all actions must be summarized leveraging speed/positions
        """
        actions = []
        front_vector = self.headings[0]
        left_vector = -front_vector[1], front_vector[0]
        final_pos = self.positions[now_frame]
        init_pos = self.positions[0]
        displacement = final_pos[0] - init_pos[0], final_pos[1] - init_pos[1]
        if dot(displacement, left_vector) > 3:
            actions.append("turn_left")
        elif dot(displacement, left_vector) < -3:
            actions.append("turn_right")
        pos_differential = 0
        prev_pos = self.positions[0]
        for pos in self.positions[1:now_frame + 1]:
            new_differential = get_distance(pos, prev_pos)
            pos_differential += new_differential
            prev_pos = pos
        if pos_differential < 0.5:
            actions.append("parked")
        else:
            actions.append("moving")
        std_speed = np.std([self.speeds[:now_frame + 1]])
        speed_differentials = []
        prev_speed = self.speeds[0]
        for speed in self.speeds[1:now_frame + 1]:
            speed_differentials.append(speed - prev_speed)
            prev_speed = speed
        if majority_true(speed_differentials, lambda x: x > 0) and self.speeds[now_frame] > self.speeds[
            0] and std_speed > 1:
            actions.append("accelerating")
        elif majority_true(speed_differentials, lambda x: x < 0) and self.speeds[now_frame] < self.speeds[
            0] and std_speed > 1:
            actions.append("decelerating")
        return actions

    def analyze_interaction(self, other, now_frame):
        def pass_by(center, other):
            def sign_onestep(obj1, obj2, step_id):
                # get the sign of obj2-obj1 dot obj1.heading for single step
                obj1_pos = obj1.positions[step_id]
                obj2_pos = obj2.positions[step_id]
                obj1_heading = obj1.headings[step_id]
                obj2_obj1 = [obj2_pos[0] - obj1_pos[0], obj2_pos[1] - obj1_pos[1]]
                prod = dot(obj1_heading, obj2_obj1)
                if prod > 0:
                    return 1
                elif prod == 0:
                    return 0
                else:
                    return -1

            def close_point(obj1, obj2):
                min_distance = float("inf")
                for step in range(now_frame + 1):
                    min_distance = min(min_distance, get_distance(obj1.positions[step], obj2.positions[step]))
                return min_distance

            closest_distance = close_point(center, other)
            signs = [sign_onestep(center, other, i) for i in range(now_frame + 1)]
            return signs[0] < 0 < signs[-1] and (closest_distance < 5)

        def head_toward(center, other):
            def head_toward_onestep(obj1, obj2, step_id):
                # check if single step satisfy the follow condition
                obj1_pos = obj1.positions[step_id]
                obj2_pos = obj2.positions[step_id]
                obj1_heading = obj1.headings[step_id]
                obj2_heading = obj2.headings[step_id]
                obj2_obj1 = [obj2_pos[0] - obj1_pos[0], obj2_pos[1] - obj1_pos[1]]
                heading_checker = dot(obj2_heading, obj2_obj1) < 0 and dot(obj1_heading, obj2_obj1) > 0
                return heading_checker and obj2.speeds[step_id] > 0

            signs = [head_toward_onestep(center, other, i) for i in range(now_frame + 1)]
            return majority_true(signs)  # all(signs)

        def follow(center, other):

            def follow_one_step(obj1, obj2, step_id):
                obj1_pos = obj1.positions[step_id]
                obj2_pos = obj2.positions[step_id]
                obj1_heading = obj1.headings[step_id]
                obj2_heading = obj2.headings[step_id]
                obj2_obj1 = [obj2_pos[0] - obj1_pos[0], obj2_pos[1] - obj1_pos[1]]
                heading_checker = dot(obj2_heading, obj2_obj1) < 0 and dot(obj1_heading, obj2_obj1) < -2 and norm(
                    obj2_obj1) < 10
                return heading_checker and obj1.speeds[step_id] > 0 and obj2.speeds[step_id] > 0

            flags = [follow_one_step(center, other, i) for i in range(now_frame + 1)]
            return majority_true(flags)  # all(flags)

        def drive_alongside(center, other):
            def alongside_onestep(obj1, obj2, step_id):
                # check if single step satisfy the follow condition
                obj1_pos = obj1.positions[step_id]
                obj2_pos = obj2.positions[step_id]
                obj1_heading = obj1.headings[step_id]
                obj2_heading = obj2.headings[step_id]
                obj2_bbox = obj2.bboxes[step_id]
                distance_checker = get_distance(obj1_pos, obj2_pos) < 5
                heading_checker = dot(obj1_heading, obj2_heading) > 0.8
                frontback_checker = position_frontback_relative_to_obj1(obj1_heading, obj1_pos, obj2_bbox) == "overlap"
                rightleft_checker = position_left_right_relative_to_obj1(obj1_heading, obj1_pos, obj2_bbox) != "overlap"
                return distance_checker and heading_checker and frontback_checker and rightleft_checker and obj1.speeds[
                    step_id] > 0 and obj2.speeds[step_id] > 0

            flags = [alongside_onestep(center, other, i) for i in range(now_frame + 1)]
            # print(flags)
            return majority_true(flags)  # all(flags)

        pass_by_flag = pass_by(self, other)
        if pass_by_flag:
            self.interactions["passed_by"].append(other.id)
            other.interactions["pass_by"].append(self.id)
        head_toward_flag = head_toward(self, other)
        if head_toward_flag:
            self.interactions["headed_toward"].append(other.id)
            other.interactions["head_toward"].append(self.id)
        follow_flag = follow(self, other)
        if follow_flag:
            self.interactions["followed"].append(other.id)
            other.interactions["follow"].append(self.id)
        alongside_flag = drive_alongside(self, other)
        if alongside_flag:
            self.interactions["accompanied_by"].append(other.id)
            other.interactions["move_alongside"].append(self.id)

    def get_all_interactions(self):
        all_interaction = []
        for interaction, value in self.interactions.items():
            if interaction in ["pass_by", "head_toward", "follow", "move_alongside"] and len(value) > 0:
                all_interaction.append(interaction)
        return all_interaction

    @property
    def visible(self):
        return len(self.observing_cameras[self.now_frame]) > 0

    @property
    def bbox(self):
        return self.bboxes[self.now_frame]

    @property
    def pos(self):
        return self.positions[self.now_frame]

    @property
    def heading(self):
        return self.headings[self.now_frame]

    @property
    def speed(self):
        return self.speeds[self.now_frame]

    def leftORright(self, node, ref_heading: Iterable[float]) -> int:
        """
        return 1 for right, -1 for left, and 0 for in the middle.
        By using the decorator, we are always referring to the key frame.
        """
        # Decide Left or Right relationships base on the bounding box of the tested object and the left/right boundary
        # of the compared object. If all vertices are to the left of the front boundary, then we way the tested object
        # is to the left of the compared object(and vice versa for right).
        # node w.r.t to me
        # ref_heading is the direction of ego_vehicle.This direction may not be the heading for me
        normal = -ref_heading[1], ref_heading[0]
        ego_left, ego_right = find_extremities(normal, self.bbox, self.pos)
        node_bbox = node.bbox
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

    def frontORback(self, node, ref_heading: Iterable[float]) -> int:
        """
        return 1 for front, -1 for back, and 0 for in the middle.
        By using the decorator, we are always referring to the key frame.
        """
        # Decide Front or Back relationships base on the bounding box of the tested object and the front/back boundary
        # of the compared object. If all vertices are in front of the front boundary, then we way the tested object
        # is in front of the compared object(and vice versa for back).
        # node w.r.t to me

        ego_front, ego_back = find_extremities(ref_heading, self.bbox, self.pos)
        node_bbox = node.bbox
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

    def compute_relation(self, node, ref_heading: Iterable[float]) -> dict:
        """
        node: AgentNode. The node you wish to examines its relation w.r.t. to this node
        ref_heading: The direction of the +x axis of the coordinate you wish to examine the spatial relationships, expressed in world coordinate.
        Encode spatial relation with ints.
        Indicators:
            Left  | Right | colinear:  -1 | 1 | 0
            Back  | Front | colinear:  -1 | 1 | 0
        This relation is only true for the key frame.
        """
        assert node is not None, "node is None in agent_node.AgentNode.compute_relation"
        relation = {
            'side': self.leftORright(node, ref_heading),
            'front': self.frontORback(node, ref_heading),
        }
        return relation

    def compute_relation_string(self, node, ref_heading):
        assert node is not None, "node is None in agent_node.AgentNode.compute_relation_string"
        relation = self.compute_relation(node, ref_heading)
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

    def __str__(self):
        return self.id


def nodify(scene_dict: dict, multiview=True) -> Tuple[str, List[ObjectNode]]:
    """
    Read world JSON file into nodes. 
    Return <ego id, list of ObjectNodes>
    """
    ego_dict = scene_dict['ego']
    ego_id = scene_dict['ego']['id']
    nodes = []
    if multiview:
        visible_criteria = lambda x: len(x) > 0
    else:
        visible_criteria = lambda x: "front" in x
    for info in scene_dict['objects']:
        nodes.append(ObjectNode(
            pos=info["pos"],
            color=info["color"],
            speed=info["speed"],
            heading=info["heading"],
            id=info['id'],
            bbox=info['bbox'],
            height=info['height'],
            type=info['type'],
            observing_cameras=info["observing_camera"],
            visible_criteria=visible_criteria,
            collisions=info["collisions"]
        )
        )
    nodes.append(
        ObjectNode(
            pos=ego_dict["pos"],
            color=ego_dict["color"],
            speed=ego_dict["speed"],
            heading=ego_dict["heading"],
            id=ego_dict['id'],
            bbox=ego_dict['bbox'],
            height=ego_dict['height'],
            type=ego_dict['type'],
            observing_cameras=ego_dict["observing_camera"],
            visible_criteria=visible_criteria,
            collisions=ego_dict["collisions"])
    )
    return ego_id, nodes


def transform(ego: Union[ObjectNode | TemporalNode], bbox: Iterable[Iterable[float]]) -> Iterable:
    """
    Coordinate system transformation from world coordinate to ego's coordinate.
    +x being ego's heading, +y being +x rotate 90 degrees counterclockwise.
    """

    # assert len(bbox) == 4 ,"bbox has more than four points in agent_node.transform"
    def change_bases(x, y):
        relative_x, relative_y = x - ego.pos[0], y - ego.pos[1]
        new_x = ego.heading
        new_y = (-new_x[1], new_x[0])
        x = (relative_x * new_x[0] + relative_y * new_x[1])
        y = (relative_x * new_y[0] + relative_y * new_y[1])
        return [x, y]

    return [change_bases(*point) for point in bbox]


def distance(node1: ObjectNode, node2: ObjectNode) -> float:
    """
    Return the Euclidean distance between two AgentNodes
    """
    dx, dy = node1.pos[0] - node2.pos[0], node1.pos[1] - node2.pos[1]
    return np.sqrt(dx ** 2 + dy ** 2)


