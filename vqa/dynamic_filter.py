import json
import os
from vqa.dataset_utils import l2_distance, find_overlap_episodes, position_frontback_relative_to_obj1, position_left_right_relative_to_obj1
class DynamicFilter:
    def __init__(self, scene_folder):
        self.scene_folder = scene_folder
        self.scene_info = self.load_scene()

    def process_episodes(self, sample_frequency:int, episode_length:int, skip_length:int)->dict:
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

        return objects_info



    def follow(self, obj1:str, obj2:str, distance_threshold = 10)->list:
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
        :return: a list of episodes where obj1 is followed by obj2
        '''
        def follow_onestep(episode, step_id):
            # check if single step satisfy the follow condition
            obj1_pos = self.scene_info[obj1][episode][step_id]["position"]
            obj2_pos = self.scene_info[obj2][episode][step_id]["position"]
            obj1_heading = self.scene_info[obj1][episode][step_id]["heading"]
            obj2_bbox = self.scene_info[obj2][episode][step_id]["bbox"]
            distance_checker = l2_distance(obj1_pos, obj2_pos) < distance_threshold
            obj2_obj1 = [obj2_pos[0]-obj1_pos[0], obj2_pos[1]-obj1_pos[1]]
            heading_checker = obj1_heading[0]*obj2_obj1[0] + obj1_heading[1]*obj2_obj1[1] < 0
            frontback_checker = position_frontback_relative_to_obj1(obj1_heading, obj1_pos, obj2_bbox) == "back"
            rightleft_checker = position_left_right_relative_to_obj1(obj1_heading, obj1_pos, obj2_bbox) == "overlap"
            # TODO: Implement Lidar
            return distance_checker and heading_checker and frontback_checker and rightleft_checker
        episodes_overlap = find_overlap_episodes(self.scene_info, obj1, obj2)
        match_episode = []
        # for all continuous episodes, check if all steps satisfy the follow condition
        for episode in episodes_overlap:
            episode_flag = True
            # for all time steps in the episode, check if the follow condition is satisfied
            for step_id in self.scene_info[obj1][episode].keys():
                if not follow_onestep(episode, step_id):
                    episode_flag = False
            if episode_flag:
                match_episode.append(episode)
        return match_episode


    def pass_by(self, obj1:str, obj2:str)->list:
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
            obj1_pos = self.scene_info[obj1][episode][step_id]["position"]
            obj2_pos = self.scene_info[obj2][episode][step_id]["position"]
            obj1_heading = self.scene_info[obj1][episode][step_id]["heading"]
            obj2_obj1 = [obj2_pos[0]-obj1_pos[0], obj2_pos[1]-obj1_pos[1]]
            return obj1_heading[0]*obj2_obj1[0] + obj1_heading[1]*obj2_obj1[1]

        episodes_overlap = find_overlap_episodes(self.scene_info, obj1, obj2)
        match_episode = []
        for episode in episodes_overlap:
            prev_sign = 0 # Track previous sign
            changed_sign = False # Whether we have changed sign in this episode
            match = False # Whether this episode is a match for "only one sign reversal
            for step_id in self.scene_info[obj1][episode].keys():
                curr_sign = sign_onestep(episode, step_id)
                if curr_sign > 0 and prev_sign < 0:
                    if changed_sign: # if we have changed sign before, then this episode is not a match
                        match = False
                    else:
                        match = True
                        changed_sign = True
                        prev_sign = curr_sign

                elif curr_sign < 0 and prev_sign > 0:
                    if changed_sign: # if we have changed sign before, then this episode is not a match
                        match = False
                    else:
                        match = True
                        changed_sign = True
                        prev_sign = curr_sign
            if match:
                match_episode.append(episode)
        return match_episode

    def collide_with(self, obj1:str, obj2:str)->list:
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

    def head_toward(self, obj1:str, obj2:str)->list:
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
            obj1_pos = self.scene_info[obj1][episode][step_id]["position"]
            obj2_pos = self.scene_info[obj2][episode][step_id]["position"]
            obj2_heading = self.scene_info[obj2][episode][step_id]["heading"]
            obj2_obj1 = [obj2_pos[0]-obj1_pos[0], obj2_pos[1]-obj1_pos[1]]
            heading_checker = obj2_heading[0]*obj2_obj1[0] + obj2_heading[1]*obj2_obj1[1] < 0
            return heading_checker
        episodes_overlap = find_overlap_episodes(self.scene_info, obj1, obj2)
        match_episode = []
        # for all continuous episodes, check if all steps satisfy the follow condition
        for episode in episodes_overlap:
            episode_flag = True
            # for all time steps in the episode, check if the follow condition is satisfied
            for step_id in self.scene_info[obj1][episode].keys():
                if not head_toward_onestep(episode, step_id):
                    episode_flag = False
            if episode_flag:
                match_episode.append(episode)
        return match_episode

    def drive_alongside(self, obj1:str, obj2:str, distance_threshold=10, heading_threshold=0.7)->list:
        '''
        Return a list of episodes where obj2 drive alongside obj1
        obj2 drive alongside obj1 if:
            |obj1-obj2| < some value, for the entirety of the observation episode
            obj1 and obj2 heading dot product is near 1
            obj2 is about directly to the right(left) of obj1 for the entirety of the observation episode.

        '''
        #obj2 drive alongside obj1 if obj2 remains about directly to the right(left) of obj1 for the entirety of the
        #observation episode.
        def alongside_onestep(episode, step_id):
            # check if single step satisfy the follow condition
            obj1_pos = self.scene_info[obj1][episode][step_id]["position"]
            obj2_pos = self.scene_info[obj2][episode][step_id]["position"]
            obj1_heading = self.scene_info[obj1][episode][step_id]["heading"]
            obj2_heading = self.scene_info[obj2][episode][step_id]["heading"]
            obj2_bbox = self.scene_info[obj2][episode][step_id]["bbox"]
            distance_checker = l2_distance(obj1_pos, obj2_pos) < distance_threshold
            heading_checker = obj1_heading[0]*obj2_heading[0] + obj1_heading[1]*obj2_heading[1] > heading_threshold
            frontback_checker = position_frontback_relative_to_obj1(obj1_heading, obj1_pos, obj2_bbox) == "overlap"
            rightleft_checker = position_left_right_relative_to_obj1(obj1_heading, obj1_pos, obj2_bbox) != "overlap"
            return distance_checker and heading_checker and frontback_checker and rightleft_checker

        episodes_overlap = find_overlap_episodes(self.scene_info, obj1, obj2)
        match_episode = []
        # for all continuous episodes, check if all steps satisfy the follow condition
        for episode in episodes_overlap:
            episode_flag = True
            # for all time steps in the episode, check if the follow condition is satisfied
            for step_id in self.scene_info[obj1][episode].keys():
                if not alongside_onestep(episode, step_id):
                    episode_flag = False
            if episode_flag:
                match_episode.append(episode)
        return match_episode

if __name__=="__main__":
    scene_folder = "D:\\research\\metavqa-merge\\MetaVQA\\vqa\\verification"
    dynamic_filter = DynamicFilter(scene_folder)
    objects_info = dynamic_filter.load_scene()
    print(objects_info[list(objects_info.keys())[0]])

#TODO: Episodic Reocrding
#TODO: Record Collision Event.  In scenario_generation.py