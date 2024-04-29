import os
from vqa.dataset_utils import dot, get_distance, l2_distance, find_overlap_episodes, \
    position_frontback_relative_to_obj1, position_left_right_relative_to_obj1
import math

from vqa.scene_graph import SceneGraph
from vqa.object_node import ObjectNode
from collections import defaultdict
from typing import Iterable
import numpy as np


class DynamicFilter:
    def __init__(self, scene_folder, sample_frequency: int, episode_length: int, skip_length: int):
        self.scene_folder = scene_folder
        self.scene_info = self.process_episodes(sample_frequency, episode_length, skip_length)

    def __init__(self, episodes_folder):
        self.scene_folder = episodes_folder
        self.scene_info = self.process_episodes()

    def process_episodes(self, sample_frequency: int, episode_length: int, skip_length: int) -> dict:
        '''
        Read the world json file for specified episodes, and return a dictionary of object information.

        Should Match to scenario_generation.py. Skip the first skip_length steps, and sample every sample_frequency steps
        for episode_length steps. (This is looping )

        The returned dictionary is of the form: {obj_id: {episode: {step: obj_info}}}, where obj_id is the unique id
        of the object, episode is the episode number in the format of "{episode begin step}_{episode end step}",
        step is the global step number

        :param sample_frequency: The frequency of sampling the world json file.
        :param episode_length: The length of each episode.
        :param skip_length: The length of each skip.
        :return: A dictionary of object information.
        '''
        objects_info = {}

        # Iterate over each subfolder in the main folder
        for subfolder in os.listdir(self.scene_folder):
            if os.path.isdir(os.path.join(self.scene_folder, subfolder)):
                if subfolder == "dynamic":
                    continue
                step = int(subfolder.split('_')[-1])

                # Check if the step should be skipped or processed
                # We shouldn't meet any step that is both skipped and processed
                assert not step % (skip_length + episode_length) < skip_length
                # We should only meet steps that are processed
                assert (step % sample_frequency == 0 and
                        (step % (skip_length + episode_length) - skip_length) < episode_length)
                json_file_path = os.path.join(self.scene_folder, subfolder, f"world_{subfolder}.json")

                if os.path.exists(json_file_path):
                    with open(json_file_path, 'r') as file:
                        data = json.load(file)
                    # get the object information
                    for obj in data.get("objects", []):
                        obj_id = obj.get("id")
                        if obj_id is not None:
                            if obj_id not in objects_info:
                                objects_info[obj_id] = {}
                            # get the episode number
                            episode_start = step - (step % (skip_length + episode_length)) + skip_length + 1
                            episode_end = episode_start + episode_length - 1
                            episode = f"{episode_start}-{episode_end}"
                            if episode not in objects_info[obj_id]:
                                objects_info[obj_id][episode] = {}

                            objects_info[obj_id][episode][step] = obj
                    for ego in data["ego"]:
                        ego_id = ego["id"]
                        if ego_id not in objects_info:
                            objects_info[ego_id] = {}
                        episode_start = step - (step % (skip_length + episode_length)) + skip_length + 1
                        episode_end = episode_start + episode_length - 1
                        episode = f"{episode_start}-{episode_end}"
                        if episode not in objects_info[ego_id]:
                            objects_info[ego_id][episode] = {}
                        objects_info[ego_id][episode][step] = obj

        return objects_info

    def process_episodes(self) -> dict:
        objects_info = {}
        for episode_folder in os.listdir(self.scene_folder):
            # splitted = episode_folder.split("_")
            # episode_start, episode_end = splitted[1], splitted[2]
            full_path = os.path.join(self.scene_folder, episode_folder)
            for frame_folder in os.listdir(full_path):
                if os.path.isdir(os.path.join(full_path, frame_folder)):
                    step = int(frame_folder.split("_")[-1])
                    json_file_path = os.path.join(self.scene_folder, episode_folder, frame_folder,
                                                  f"world_{frame_folder}.json")
                    try:
                        with open(json_file_path, "r") as file:
                            data = json.load(file)
                        for obj in data.get("objects", []):
                            obj_id = obj.get("id")
                            if obj_id is not None:
                                if obj_id not in objects_info:
                                    objects_info[obj_id] = {}
                                episode = episode_folder  # f"{episode_start}-{episode_end}"
                                if episode not in objects_info[obj_id]:
                                    objects_info[obj_id][episode] = {}
                                objects_info[obj_id][episode][step] = obj
                        ego = data.get("ego")
                        ego_id = ego["id"]
                        if ego_id not in objects_info:
                            objects_info[ego_id] = {}
                        if episode not in objects_info[ego_id]:
                            objects_info[ego_id][episode] = {}
                        objects_info[ego_id][episode][step] = obj
                    except Exception as e:
                        raise e
        return objects_info

    def follow(self, obj1: str, obj2: str, distance_threshold: int = 10, strickness: float = 0.8) -> list:
        '''
        Return a list of episodes where obj1 is followed by obj2
        obj1 is followed by obj2 if:
            |obj1-obj2| < some value, for the entirety of the observation episode
            obj1.heading dot (obj2-obj1) is negative, for the entirety of the observation episode
            obj1 can shoot a ray to obj2, and obj2 can shoot a ray to obj1, for he entirety of the observation episode
                    |-> requrire further annotation: for each vehicle, record the lidar-observable vehicles.
            obj2 must be almost exactly behind obj1
        :param obj1: the id of the object that is followed
        :param obj2: the id of the object that follows
        :param distance_threshold: the distance threshold for the first condition
        :param strickness: the percentage of the steps needed for predicate to be true on the episode.
        :return: a list of episodes where obj1 is followed by obj2
        '''

        def follow_onestep(episode, step_id):
            # check if single step satisfy the follow condition
            obj1_pos = self.scene_info[obj1][episode][step_id]["pos"]
            obj2_pos = self.scene_info[obj2][episode][step_id]["pos"]
            obj1_heading = self.scene_info[obj1][episode][step_id]["heading"]
            obj2_bbox = self.scene_info[obj2][episode][step_id]["bbox"]
            distance_checker = get_distance(obj1_pos, obj2_pos) < distance_threshold
            obj2_obj1 = [obj2_pos[0] - obj1_pos[0], obj2_pos[1] - obj1_pos[1]]
            heading_checker = obj1_heading[0] * obj2_obj1[0] + obj1_heading[1] * obj2_obj1[1] < 0
            frontback_checker = position_frontback_relative_to_obj1(obj1_heading, obj1_pos, obj2_bbox) == "back"
            rightleft_checker = position_left_right_relative_to_obj1(obj1_heading, obj1_pos, obj2_bbox) == "overlap"
            # TODO: Implement Lidar
            return distance_checker and heading_checker and frontback_checker and rightleft_checker

        episodes_overlap = find_overlap_episodes(self.scene_info, obj1, obj2)
        match_episode = []
        # for all continuous episodes, check if all steps satisfy the follow condition

        for episode in episodes_overlap:
            # episode_flag = True
            satisfy_counter = 0
            counter = 0
            # for all time steps in the episode, check if the follow condition is satisfied
            for step_id in self.scene_info[obj1][episode].keys():
                counter += 1
                if follow_onestep(episode, step_id):
                    satisfy_counter += 1
            if satisfy_counter / counter >= strickness:
                match_episode.append(episode)
        return match_episode

    def pass_by(self, obj1: str, obj2: str) -> list:
        '''
        Return a list of episodes where obj2 pass by obj1
        obj2 pass_by obj1 if:
            obj2-obj1 dot obj1.heading experience exactly one sign reversal for he entirety of the observation episode
        :param obj1: the id of the object that is passed by
        :param obj2: the id of the object that passes by
        :return: a list of episodes where obj2 pass by obj1
        '''

        def sign_onestep(episode, step_id):
            # get the sign of obj2-obj1 dot obj1.heading for single step
            obj1_pos = self.scene_info[obj1][episode][step_id]["pos"]
            obj2_pos = self.scene_info[obj2][episode][step_id]["pos"]
            obj1_heading = self.scene_info[obj1][episode][step_id]["heading"]
            obj2_obj1 = [obj2_pos[0] - obj1_pos[0], obj2_pos[1] - obj1_pos[1]]
            prod = dot(obj1_heading, obj2_obj1)
            if prod > 0:
                return 1
            elif prod == 0:
                return 0
            else:
                return -1

        episodes_overlap = find_overlap_episodes(self.scene_info, obj1, obj2)
        match_episode = []
        for episode in episodes_overlap:
            prev_sign = None  # Track previous sign
            signature = []
            flag = False
            for step_id in self.scene_info[obj1][episode].keys():
                curr_sign = sign_onestep(episode, step_id)
                if prev_sign is not None and curr_sign != prev_sign:
                    signature.append(prev_sign)
                prev_sign = curr_sign
            if not signature:
                continue
            if curr_sign != signature[-1]:
                signature.append(curr_sign)
            if len(signature) == 2:
                if (signature[0] == -1 and signature[-1] == 1) or (signature[0] == 1 and signature[-1] == -1):
                    flag = True
            if len(signature) == 3:
                if (signature[0] == -1 and signature[-1] == 1) or (signature[0] == 1 and signature[-1] == -1):
                    flag = True
            if flag:
                match_episode.append(episode)
        return match_episode

    def collide_with(self, obj1: str, obj2: str) -> list:
        '''
        Return a list of episodes where obj1 collide with obj2
        obj1 collide with obj2 if:
            one collision record exists between obj1 and obj2
        :param obj1: the id of the object that collides with
        :param obj2: the id of the object that is collided with
        :return: a list of episodes where obj1 collide with obj2
        '''

        episodes_overlap = find_overlap_episodes(self.scene_info, obj1, obj2)
        match_episode = []
        # for all continuous episodes, check if all steps satisfy the follow condition
        for episode in episodes_overlap:
            episode_flag = True
            # for all time steps in the episode, check if the follow condition is satisfied
            for step_id in self.scene_info[obj1][episode].keys():
                # TODO: Implement Collision
                pass
            if episode_flag:
                match_episode.append(episode)
        return match_episode

    def head_toward(self, obj1: str, obj2: str, strickness: float = 0.8) -> list:
        '''
        Return a list of episodes where obj2 head toward obj1
        obj2 head toward obj1 if:
            obj2-obj1 dot obj2.heading is negative, for the entirety of the observation episode
        :param obj1: the id of the object that is head toward
        :param obj2: the id of the object that head toward
        :return: a list of episodes where obj2 head toward obj1
        '''

        def head_toward_onestep(episode, step_id):
            # check if single step satisfy the follow condition
            obj1_pos = self.scene_info[obj1][episode][step_id]["pos"]
            obj2_pos = self.scene_info[obj2][episode][step_id]["pos"]
            obj2_heading = self.scene_info[obj2][episode][step_id]["heading"]
            obj2_obj1 = [obj2_pos[0] - obj1_pos[0], obj2_pos[1] - obj1_pos[1]]
            heading_checker = dot(obj2_heading, obj2_obj1) < 0
            return heading_checker

        episodes_overlap = find_overlap_episodes(self.scene_info, obj1, obj2)
        match_episode = []
        # for all continuous episodes, check if all steps satisfy the follow condition
        for episode in episodes_overlap:
            satisfy_counter = counter = 0
            # for all time steps in the episode, check if the follow condition is satisfied
            for step_id in self.scene_info[obj1][episode].keys():
                if head_toward_onestep(episode, step_id):
                    satisfy_counter += 1
                counter += 1
            if satisfy_counter / counter >= strickness:
                match_episode.append(episode)
        return match_episode

    def drive_alongside(self, obj1: str, obj2: str, distance_threshold=10, heading_threshold=0.7,
                        stricktness=0.8) -> list:
        '''
        Return a list of episodes where obj2 drive alongside obj1
        obj2 drive alongside obj1 if:
            |obj1-obj2| < some value, for the entirety of the observation episode
            obj1 and obj2 heading dot product is near 1
            obj2 is about directly to the right(left) of obj1 for the entirety of the observation episode.

        '''

        # obj2 drive alongside obj1 if obj2 remains about directly to the right(left) of obj1 for the entirety of the
        # observation episode.
        def alongside_onestep(episode, step_id):
            # check if single step satisfy the follow condition
            obj1_pos = self.scene_info[obj1][episode][step_id]["pos"]
            obj2_pos = self.scene_info[obj2][episode][step_id]["pos"]
            obj1_heading = self.scene_info[obj1][episode][step_id]["heading"]
            obj2_heading = self.scene_info[obj2][episode][step_id]["heading"]
            obj2_bbox = self.scene_info[obj2][episode][step_id]["bbox"]
            distance_checker = get_distance(obj1_pos, obj2_pos) < distance_threshold
            heading_checker = dot(obj1_heading, obj2_heading) > heading_threshold
            frontback_checker = position_frontback_relative_to_obj1(obj1_heading, obj1_pos, obj2_bbox) == "overlap"
            rightleft_checker = position_left_right_relative_to_obj1(obj1_heading, obj1_pos, obj2_bbox) != "overlap"
            return distance_checker and heading_checker and frontback_checker and rightleft_checker

        episodes_overlap = find_overlap_episodes(self.scene_info, obj1, obj2)
        match_episode = []
        # for all continuous episodes, check if all steps satisfy the follow condition
        for episode in episodes_overlap:
            satisfy_counter = True
            counter = True
            # for all time steps in the episode, check if the follow condition is satisfied
            for step_id in self.scene_info[obj1][episode].keys():
                if alongside_onestep(episode, step_id):
                    satisfy_counter += 1
                counter += 1
            if satisfy_counter / counter >= stricktness:
                match_episode.append(episode)
        return match_episode


class TemporalNode:
    def __init__(self, now_frame, id, type, height, positions, color, speeds, headings, bboxes, observing_cameras,
                 states,
                 collisions, interaction=[]):
        # time-invariant properties
        self.now_frame = now_frame  # 0-indexed.indicate when is "now". The past is history and inclusive of now.
        self.id = id  # as defined in the metadrive env
        self.states = states  # accleration, thurning
        self.type = type  # as annotated
        self.height = height  # The height retrieved from the asset's convex hull.
        self.color = color  # as annotated

        # time-dependent properties
        self.positions = positions  # (x,y) in world coordinate
        self.speeds = speeds  # in m/s
        self.headings = headings  # (dx, dy), in world coordinate
        self.bboxes = bboxes  # bounding box with
        self.observing_cameras = observing_cameras  # record which ego camera observed the object.
        self.collision = collisions  # record collisions(if any) along time.
        self.interaction = interaction  # interaction record is (other_node, action_name)
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

    def summarize_action(self, now_frame):
        """
        Apparently, in real envs, we don't have any second-order information. We only have speed.
        So, all actions must be summarized leveraging speed/positions.

        """
        actions = []
        front_vector = self.headings[0]
        left_vector = -front_vector[1], front_vector[0]
        final_pos = self.positions[now_frame]
        init_pos = self.positions[0]
        displacement = final_pos[0] - init_pos[0], final_pos[1] - init_pos[1]
        print(self.id, dot(displacement, left_vector))
        if dot(displacement, left_vector) > 1.5:
            actions.append("turn_left")
        elif dot(displacement, left_vector) < -1.5:
            actions.append("turn_right")
        else:
            actions.append("go straight")
        pos_differential = 0
        prev_pos = self.positions[0]
        for pos in self.positions[1:now_frame + 1]:
            new_differential = get_distance(pos, prev_pos)
            pos_differential += new_differential
            prev_pos = pos
        if pos_differential < 0.5:
            actions.append("parked")
        std_speed = np.std([self.speeds[:now_frame + 1]])
        speed_differentials = []
        prev_speed = self.speeds[0]
        for speed in self.speeds[1:now_frame + 1]:
            speed_differentials.append(speed-prev_speed)
            prev_speed = speed
        if majority_true(speed_differentials, lambda x : x > 0) and self.speeds[now_frame] > self.speeds[0] and std_speed > 1:
            actions.append("acceleration")
        elif majority_true(speed_differentials, lambda x : x < 0) and self.speeds[now_frame] < self.speeds[0] and std_speed > 1:
            actions.append("deceleration")
        return actions
    def __str__(self):
        return self.id

def majority_true(things, creterion, threshold=0.8):
    num_things = len(things)
    num_true = 0
    for thing in things:
        if creterion(thing):
            num_true += 1
    return num_true/num_things >= threshold


class TemporalGraph:
    def __init__(self, framepaths, observable_at_key=True, tolerance=0.8):
        """
        We ask questions based on the observation_phase and return answr for the prediction phase
        Note that in MetaDrive each step is 0.1s
        Will give attempt to give 2 seconds of observation and half seconds of prediction phase
        So, observation phase = 20 and prediction phase = 5
        Each graph will store the path to the original annotation("For statistics purpose") and also the loaded information

        """

        self.observation_phase = 0.8  # This is the percentage of frames belonging into observation. The last frame
        # is "present"
        self.prediction_phase = 1.0 - self.observation_phase
        self.tolerance = tolerance  # The percentage(of observation phase) of being observable for objects to be
        # considered in the graph.
        self.observable_at_key = observable_at_key  # enforce that the objects referred must be observable at the key
        # frame.
        self.framepaths: list[str] = framepaths
        self.frames: list[dict] = [json.load(open(framepath, "r")) for framepath in framepaths]

        self.num_observation_frames = math.floor(len(self.framepaths) * self.observation_phase)
        self.num_prediction_frames = len(self.framepaths) - self.num_observation_frames
        self.idx_key_frame = self.num_observation_frames - 1  # since 0-indexed. The past is [0,self._idx_key_frame]
        self.node_ids: Iterable[str] = self.find_node_ids(self.frames, self.observable_at_key, self.tolerance)
        self.ego_id: str = self.frames[0]["ego"]["id"]
        assert all([self.frames[i]["ego"]["id"] == self.ego_id for i in
                    range(len(self.frames))]), "Ego changed during this period."
        self.nodes = self.build_nodes(self.node_ids, self.frames)
        self.key_frame_graph: SceneGraph = self.build_key_frame()

    def build_nodes(self, node_ids, frames) -> dict[str, TemporalNode]:
        positions = defaultdict(list)
        headings = defaultdict(list)
        colors = defaultdict(str)
        types = defaultdict(str)
        visibles = defaultdict(list)
        observing_cameras = defaultdict(list)
        speeds = defaultdict(list)
        bboxes = defaultdict(list)
        heights = defaultdict(list)
        states = defaultdict(list)
        collisions = defaultdict(list)
        actions = defaultdict(dict)  # need to analyze inter-object actions later.
        temporal_nodes = {}
        for timestamp, frame in enumerate(frames):
            for object in frame["objects"]:
                if object["id"] in node_ids:
                    positions[object["id"]].append(object["pos"])
                    headings[object["id"]].append(object["heading"])
                    colors[object["id"]] = object["color"]
                    types[object["id"]] = object["type"]
                    visibles[object["id"]].append(object["visible"])
                    observing_cameras[object["id"]].append(object["observing_camera"])
                    speeds[object["id"]].append(object["speed"])
                    bboxes[object["id"]].append(object["bbox"])
                    heights[object["id"]].append(object["height"])
                    states[object["id"]].append(object["states"])
                    collisions[object["id"]].append(object["collisions"])

            positions[self.ego_id].append(frame["ego"]["pos"])
            headings[self.ego_id].append(frame["ego"]["heading"])
            colors[self.ego_id] = frame["ego"]["color"]
            types[self.ego_id] = frame["ego"]["type"]
            visibles[self.ego_id].append(frame["ego"]["visible"])
            observing_cameras[self.ego_id].append(frame["ego"]["observing_camera"])
            speeds[self.ego_id].append(frame["ego"]["speed"])
            bboxes[self.ego_id].append(frame["ego"]["bbox"])
            heights[self.ego_id].append(frame["ego"]["height"])
            states[self.ego_id].append(frame["ego"]["states"])
            collisions[self.ego_id].append(frame["ego"]["collisions"])

        for node_id in node_ids:
            temporal_node = TemporalNode(
                id=node_id, now_frame=self.idx_key_frame, type=types[node_id], height=heights[node_id],
                positions=positions[node_id],
                color=colors[node_id], speeds=speeds[node_id], headings=headings[node_id], bboxes=bboxes[node_id],
                observing_cameras=bboxes[node_id], states=states[node_id], collisions=collisions[node_id],
            )
            temporal_nodes[node_id] = temporal_node
        return temporal_nodes

    def find_node_ids(self, frames, observable_at_key=True, noise_tolerance=0.8) -> Iterable[str]:
        """
        Not including the ego_id
        We consider objects that are observable for the noise_tolerance amount
        of time(also, must be observable at key frame) for the observation period and still exist in the prediction phase
        """
        observation_frames = self.num_observation_frames
        prediction_frames = self.num_prediction_frames
        observable_at_t = {
            t: set() for t in range(observation_frames)
        }
        exist_at_t = {
            t: set() for t in range(observation_frames, observation_frames + prediction_frames)
        }
        all_nodes = set()
        for i in range(observation_frames):
            for obj in frames[i]["objects"]:
                if obj["visible"]:
                    observable_at_t[i].add(obj["id"])
                all_nodes.add(obj["id"])
        for i in range(observation_frames, observation_frames + prediction_frames):
            for obj in frames[i]["objects"]:
                exist_at_t[i].add(obj["id"])
                all_nodes.add(obj["id"])
        final_nodes = set()
        for node in all_nodes:
            observable_frame = 0
            for nodes_observable in observable_at_t.values():
                if node in nodes_observable:
                    observable_frame += 1
            if observable_frame / observation_frames >= noise_tolerance:
                if (not observable_at_key) or (node in observable_at_t[self.idx_key_frame]):
                    final_nodes.add(node)
        for nodes_exist in exist_at_t.values():
            final_nodes = final_nodes.intersection(nodes_exist)
        final_nodes.add(frames[0]["ego"]["id"])
        return final_nodes

    def get_nodes(self):
        return self.nodes

    def build_key_frame(self) -> SceneGraph:
        key_frame_annotation = self.frames[self.idx_key_frame]  # json.load(open(key_frame_path, "r"))
        nodes = []
        ego_id = key_frame_annotation["ego"]["id"]
        ego_dict = key_frame_annotation['ego']
        ego_node = ObjectNode(
            pos=ego_dict["pos"],
            color=ego_dict["color"],
            speed=ego_dict["speed"],
            heading=ego_dict["heading"],
            id=ego_dict['id'],
            bbox=ego_dict['bbox'],
            height=ego_dict['height'],
            type=ego_dict['type'],
            lane=ego_dict['lane'],
            visible=ego_dict["visible"],
            states=ego_dict["states"],
            collisions=ego_dict["collisions"]
        )
        nodes.append(ego_node)
        for info in key_frame_annotation["objects"]:
            if info["id"] in self.node_ids:
                nodes.append(ObjectNode(
                    pos=info["pos"],
                    color=info["color"],
                    speed=info["speed"],
                    heading=info["heading"],
                    id=info['id'],
                    bbox=info['bbox'],
                    height=info['height'],
                    type=info['type'],
                    lane=info['lane'],
                    visible=info['visible'],
                    states=info["states"],
                    collisions=info["collisions"])
                )
        key_graph = SceneGraph(
            ego_id=ego_id, nodes=nodes, folder=self.framepaths[self.idx_key_frame]
        )
        return key_graph

if __name__ == "__main__":
    """scene_folder = "E:\\Bolei\\MetaVQA\\temporal"
    dynamic_filter = DynamicFilter(scene_folder)
    objects_info = dynamic_filter.load_scene()
    print(objects_info[list(objects_info.keys())[0]])"""
    episode_folder = "C:/school/Bolei/Merging/MetaVQA/verification_multiview/95_150_179/**/world*.json"
    import glob
    import json

    frame_files = sorted(glob.glob(episode_folder, recursive=True))
    # frames = [json.load(open(file, "r")) for file in frames_files]
    graph = TemporalGraph(frame_files)
    #print(graph.node_ids)
    #print(graph.ego_id)
    for node in graph.get_nodes().values():
        print(node.id, node.actions)

    #print(graph.get_nodes())
    #print(graph.key_frame_graph)

    # print(graph.num_observation_frames, graph.num_prediction_frames)
