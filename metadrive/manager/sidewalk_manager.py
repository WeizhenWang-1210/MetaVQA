# manager that adds items (currently pedestrian) on the sidewalk.
# Note: currently you need to change path in the init function.
import math
import os
from collections import defaultdict
from asset.read_config import configReader
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.ramp import InRampOnStraight, OutRampOnStraight
from metadrive.component.pgblock.straight import Straight
from metadrive.component.road_network import Road
from metadrive.component.static_object.test_new_object import TestObject, TestGLTFObject
from metadrive.component.static_object.traffic_object import TrafficCone, TrafficWarning, TrafficBarrier
from metadrive.engine.engine_utils import get_engine
from metadrive.manager.base_manager import BaseManager
from metadrive.constants import DrivableAreaProperty

import json
import random
# class SidewalkManager(BaseManager):
#     """
#     This class is used to spawn static objects (currently pedestrian) on the sidewalk
#     """
#     PRIORITY = 9
#
#
#     def __init__(self):
#         super(SidewalkManager, self).__init__()
#         self.num_pedestrian_per_road = 2
#         self.pede1 = self.load_json_file("C:\\research\\gitplay\\MetaVQA\\asset\\newpede-dennis_posed_004_-_male_standing_business_model.json")
#         self.pede2 = self.load_json_file(
#             "C:\\research\\gitplay\\MetaVQA\\asset\\newpede-mei_posed_001_-_female_walking_business_model.json")
#
#     @staticmethod
#     def load_json_file(filepath):
#         with open(filepath, 'r') as f:
#             return json.load(f)
#     def before_reset(self):
#         """
#         Clear all objects in th scene
#         """
#         super(SidewalkManager, self).before_reset()
#         # self.num_pedestrian_per_road = self.engine.global_config["num_pedestrian_per_road"]
#     def quickSpawn(self, obj, lane, longitude, lateral):
#         """
#         Spawns an object quickly on a given lane at the specified position.
#
#         Args:
#         - obj (Object): The object to spawn.
#         - lane (Lane): The lane where the object should be spawned.
#         - longitude (float): The longitudinal position.
#         - lateral (float): The lateral position.
#         """
#         self.spawn_object(
#             obj,
#             lane=lane,
#             position=lane.position(longitude, lateral),
#             static=self.engine.global_config["static_traffic_object"],
#             heading_theta=lane.heading_theta_at(longitude)
#         )
#     def randomspawn(self, obj_json_list, lane, long_range, lateral_range):
#         """
#         Spawns a random object from the provided JSON list on a given lane within specified positional ranges.
#
#         Args:
#         - obj_json_list (list): A list of JSON objects to choose from.
#         - lane (Lane): The lane where the object should be spawned.
#         - long_range (tuple): The range for the longitudinal position.
#         - lateral_range (tuple): The range for the lateral position.
#         """
#         random_obj_json = random.choice(obj_json_list)
#         random_heading = random.choice([0, math.pi])
#         random_long = random.uniform(*long_range)  # Unpack the tuple
#         random_lateral = random.uniform(*lateral_range)  # Unpack the tuple
#         self.spawn_object(
#             TestGLTFObject,
#             lane=lane,
#             position=lane.position(random_long, random_lateral),
#             static=self.engine.global_config["static_traffic_object"],
#             heading_theta=lane.heading_theta_at(random_long)+random_heading,
#             asset_metainfo=random_obj_json,
#         )
#     def reset(self):
#         """
#         Fill the map with static objects for each block. Specifically:
#         1. Skips the `FirstPGBlock` type blocks.
#         2. Places `TrafficWarning` objects at the start and end of each lane. (serve as boundary checker)
#         3. Spawns random pedestrians (or other objects from the obj_json_list) at random positions along the lane.
#
#         Returns:
#         - None
#         """
#         engine = get_engine()
#         for block in engine.current_map.blocks:
#             if type(block) is FirstPGBlock:
#                 continue
#             # positive lane refers to the current lane, negative refer to the reverse lane on your left
#             for lane in [block.positive_basic_lane, block.negative_basic_lane]:
#                 # min range of longtitude
#                 longitude = 0
#                 # min range for lateral for the sidewalk
#                 lateral = 0 + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
#                 # place a trafficwarning to tell the boundary
#                 self.quickSpawn(TrafficWarning, lane, longitude, lateral)
#                 # max range for long
#                 longitude = lane.length
#                 # max range for lateral for the sidewalk
#                 lateral = lane.width + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
#                 self.quickSpawn(TrafficWarning, lane, longitude, lateral)
#                 long_range = (0, lane.length)
#                 # valid lateral ranges
#                 lateral_range = (0 + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST,
#                                  lane.width + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST)
#                 obj_json_list = [self.pede1, self.pede2]
#                 for i in range(5):
#                     # for each lane block, we randomly spawn 5 pedestrians.
#                     self.randomspawn(obj_json_list, lane, long_range, lateral_range)
#
#
#     def set_state(self, state: dict, old_name_to_current=None):
#         """
#         Copied from super(). Restoring some states before reassigning value to spawned_objets
#         """
#         assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
#         if old_name_to_current is None:
#             old_name_to_current = {key: key for key in state.keys()}
#         spawned_objects = state["spawned_objects"]
#         ret = {}
#         for name, class_name in spawned_objects.items():
#             current_name = old_name_to_current[name]
#             name_obj = self.engine.get_objects([current_name])
#             assert current_name in name_obj and name_obj[current_name
#                                                          ].class_name == class_name, "Can not restore mappings!"
#             # Restore some internal states
#             name_obj[current_name].lane = self.engine.current_map.road_network.get_lane(
#                 name_obj[current_name].lane.index
#             )
#
#             ret[current_name] = name_obj[current_name]
#         self.spawned_objects = ret


class SidewalkManager(BaseManager):
    """
    This class is used to spawn static objects (currently pedestrian) on the sidewalk
    """
    PRIORITY = 9


    def __init__(self):
        super(SidewalkManager, self).__init__()
        self.config = configReader()
        self.path_config = self.config.loadPath()
        self.init_static_adj_list()
        self.get_num_and_pos()
        self.spawned_obj_positions = defaultdict(list)

    def init_static_adj_list(self):
        self.type_metainfo_dict = defaultdict(list)
        for root, dirs, files in os.walk(self.path_config["adj_parameter_folder"]):
            for file in files:
                if not file.lower().startswith("car"):
                    # print(file)
                    with open(os.path.join(root, file), 'r') as f:
                        loaded_metainfo = json.load(f)
                        self.type_metainfo_dict[loaded_metainfo['general']['detail_type']].append(loaded_metainfo)
    def get_num_and_pos(self):
        self.num_dict = dict()
        self.pos_dict = dict()
        self.heading_dict = dict()
        for detail_type in self.type_metainfo_dict.keys():
            self.num_dict[detail_type] = self.config.getSpawnNum(detail_type)
            self.pos_dict[detail_type]  = self.config.getSpawnPos(detail_type)
            if self.config.getSpawnHeading(detail_type):
                self.heading_dict[detail_type] = [heading * math.pi for heading in self.config.getSpawnHeading(detail_type)]
            else:
                self.heading_dict[detail_type] = [0, math.pi]

    @staticmethod
    def load_json_file(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    def quickSpawn(self, obj, lane, longitude, lateral):
        """
        Spawns an object quickly on a given lane at the specified position.

        Args:
        - obj (Object): The object to spawn.
        - lane (Lane): The lane where the object should be spawned.
        - longitude (float): The longitudinal position.
        - lateral (float): The lateral position.
        """
        self.spawn_object(
            obj,
            lane=lane,
            position=lane.position(longitude, lateral),
            static=self.engine.global_config["static_traffic_object"],
            heading_theta=lane.heading_theta_at(longitude)
        )

    def does_overlap(self, lane, new_long, new_lat, new_width, new_length):
        for obj in self.spawned_obj_positions[lane]:
            obj_long, obj_lat, obj_width, obj_length = obj
            if (abs(new_long - obj_long) < (new_width + obj_width) / 2) and \
                    (abs(new_lat - obj_lat) < (new_length + obj_length) / 2):
                return True
        return False
    def randomObjSpawn(self, obj_json_list, lane, long_range, lateral_range, heading_list):
        """
        Spawns a random object from the provided JSON list on a given lane within specified positional ranges.

        Args:
        - obj_json_list (list): A list of JSON objects to choose from.
        - lane (Lane): The lane where the object should be spawned.
        - long_range (tuple): The range for the longitudinal position.
        - lateral_range (tuple): The range for the lateral position.
        """
        random_obj_json = random.choice(obj_json_list)
        random_heading = random.choice(heading_list)

        max_attempts = 10
        for _ in range(max_attempts):
            random_long = random.uniform(*long_range)
            random_lateral = random.uniform(*lateral_range)

            # Check for overlaps
            # width = random_obj_json["general"]["width"]
            # length = random_obj_json["general"]["length"]
            width = 3
            length = 3
            if not self.does_overlap(lane, random_long, random_lateral, width, length):
                self.spawned_obj_positions[lane].append((random_long, random_lateral, width, length))

                self.spawn_object(
                    TestObject,
                    lane=lane,
                    position=lane.position(random_long, random_lateral),
                    static=self.engine.global_config["static_traffic_object"],
                    heading_theta=lane.heading_theta_at(random_long) + random_heading,
                    asset_metainfo=random_obj_json,
                )
                return
        print("Failed to find a non-overlapping position after {} attempts.".format(max_attempts))
    def randomSpawn(self, random_obj_json, lane, long_range, lateral_range, heading_list):
        """
        Spawns a random object from the provided JSON list on a given lane within specified positional ranges.

        Args:
        - obj_json_list (list): A list of JSON objects to choose from.
        - lane (Lane): The lane where the object should be spawned.
        - long_range (tuple): The range for the longitudinal position.
        - lateral_range (tuple): The range for the lateral position.
        """
        random_heading = random.choice(heading_list)
        random_long = random.uniform(*long_range)  # Unpack the tuple
        random_lateral = random.uniform(*lateral_range)  # Unpack the tuple
        self.spawn_object(
            TestObject,
            lane=lane,
            position=lane.position(random_long, random_lateral),
            static=self.engine.global_config["static_traffic_object"],
            heading_theta=lane.heading_theta_at(random_long)+random_heading,
            asset_metainfo=random_obj_json,
        )
    def spawn_on_block(self, block):
        for lane in [block.positive_basic_lane, block.negative_basic_lane]:
            long_range = (0, lane.length)
            # valid lateral ranges
            sidewalk_lateral_range = (0 + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST,
                             lane.width + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST)
            out_lateral_range = (1.5 * lane.width + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST,
                             2 * lane.width + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST)
            for detail_type, meta_info_list in self.type_metainfo_dict.items():
                position = self.pos_dict[detail_type]
                num = self.num_dict[detail_type]
                heading = self.heading_dict[detail_type]
                print("Type: {}, heading: {}".format(detail_type, heading))
                if num <= 0:
                    continue
                elif position == "onsidewalk":
                    for _ in range(num):
                        self.randomObjSpawn(obj_json_list=meta_info_list,
                                         lane=lane,
                                         long_range=long_range,
                                         lateral_range=sidewalk_lateral_range,
                                         heading_list = heading)
                elif position == "outsidewalk":
                    for _ in range(num):
                        self.randomObjSpawn(obj_json_list=meta_info_list,
                                         lane=lane,
                                         long_range=long_range,
                                         lateral_range=out_lateral_range,
                                         heading_list = heading)


    def reset(self):
        """
        Fill the map with static objects for each block. Specifically:
        1. Skips the `FirstPGBlock` type blocks.
        2. Places `TrafficWarning` objects at the start and end of each lane. (serve as boundary checker)
        3. Spawns random pedestrians (or other objects from the obj_json_list) at random positions along the lane.

        Returns:
        - None
        """
        engine = get_engine()
        for block in engine.current_map.blocks:
            if type(block) is FirstPGBlock:
                continue
            # positive lane refers to the current lane, negative refer to the reverse lane on your left
            for lane in [block.positive_basic_lane, block.negative_basic_lane]:
                # min range of longtitude
                longitude = 0
                # min range for lateral for the sidewalk
                lateral = 0 + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
                # place a trafficwarning to tell the boundary
                self.quickSpawn(TrafficWarning, lane, longitude, lateral)
                # max range for long
                longitude = lane.length
                # max range for lateral for the sidewalk
                lateral = lane.width + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
                self.quickSpawn(TrafficWarning, lane, longitude, lateral)
                self.spawn_on_block(block)


    def set_state(self, state: dict, old_name_to_current=None):
        """
        Copied from super(). Restoring some states before reassigning value to spawned_objets
        """
        assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
        if old_name_to_current is None:
            old_name_to_current = {key: key for key in state.keys()}
        spawned_objects = state["spawned_objects"]
        ret = {}
        for name, class_name in spawned_objects.items():
            current_name = old_name_to_current[name]
            name_obj = self.engine.get_objects([current_name])
            assert current_name in name_obj and name_obj[current_name
                                                         ].class_name == class_name, "Can not restore mappings!"
            # Restore some internal states
            name_obj[current_name].lane = self.engine.current_map.road_network.get_lane(
                name_obj[current_name].lane.index
            )

            ret[current_name] = name_obj[current_name]
        self.spawned_objects = ret