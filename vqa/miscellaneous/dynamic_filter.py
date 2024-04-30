import os
from vqa.dataset_utils import dot, find_overlap_episodes, \
    position_frontback_relative_to_obj1, position_left_right_relative_to_obj1

from vqa.scene_graph import TemporalGraph
from vqa.dataset_utils import get_distance


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


if __name__ == "__main__":
    """scene_folder = "E:\\Bolei\\MetaVQA\\temporal"
    dynamic_filter = DynamicFilter(scene_folder)
    objects_info = dynamic_filter.load_scene()
    print(objects_info[list(objects_info.keys())[0]])"""
    episode_folder = "C:/school/Bolei/Merging/MetaVQA/verification_multiview/95_210_239/**/world*.json"
    #episode_folder = "E:/Bolei/MetaVQA/multiview/0_30_54/**/world*.json"
    import glob
    import json

    frame_files = sorted(glob.glob(episode_folder, recursive=True))
    graph = TemporalGraph(frame_files)
    for node in graph.get_nodes().values():
        print(node.id, node.actions)
    for node_id, node in graph.get_nodes().items():
        print(node.id, node.interactions)
