import math

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
class SidewalkManager(BaseManager):
    """
    This class is used to manager all static object, such as traffic cones, warning tripod.
    """
    PRIORITY = 9


    def __init__(self):
        super(SidewalkManager, self).__init__()
        self.num_pedestrian_per_road = 2
        self.pede1 = self.load_json_file("C:\\research\\gitplay\\MetaVQA\\asset\\newpede-dennis_posed_004_-_male_standing_business_model.json")
        self.pede2 = self.load_json_file(
            "C:\\research\\gitplay\\MetaVQA\\asset\\newpede-mei_posed_001_-_female_walking_business_model.json")

    @staticmethod
    def load_json_file(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    def before_reset(self):
        """
        Clear all objects in th scene
        """
        super(SidewalkManager, self).before_reset()
        # self.num_pedestrian_per_road = self.engine.global_config["num_pedestrian_per_road"]
    def quickSpawn(self, obj, lane, longitude, lateral):
        self.spawn_object(
            obj,
            lane=lane,
            position=lane.position(longitude, lateral),
            static=self.engine.global_config["static_traffic_object"],
            heading_theta=lane.heading_theta_at(longitude)
        )
    def randomspawn(self, obj_json_list, lane, long_range, lateral_range):
        random_obj_json = random.choice(obj_json_list)
        random_heading = random.choice([0, math.pi])
        random_long = random.uniform(*long_range)  # Unpack the tuple
        random_lateral = random.uniform(*lateral_range)  # Unpack the tuple
        self.spawn_object(
            TestGLTFObject,
            lane=lane,
            position=lane.position(random_long, random_lateral),
            static=self.engine.global_config["static_traffic_object"],
            heading_theta=lane.heading_theta_at(random_long)+random_heading,
            asset_metainfo=random_obj_json,
        )
    def reset(self):
        """
        Generate an accident scene or construction scene on block
        :return: None
        """
        engine = get_engine()
        for block in engine.current_map.blocks:
            if type(block) is FirstPGBlock:
                continue
            for lane in [block.positive_basic_lane, block.negative_basic_lane]:
                # lane = block.positive_basic_lane
                longitude = 0
                lateral = 0 + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
                self.quickSpawn(TrafficWarning, lane, longitude, lateral)
                # longitude = self.np_random.rand() * lane.length / 2 + lane.length / 2
                longitude = lane.length
                lateral = lane.width + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
                self.quickSpawn(TrafficWarning, lane, longitude, lateral)
                long_range = (0, lane.length)
                lateral_range = (0 + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST,
                                 lane.width + DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST)
                obj_json_list = [self.pede1, self.pede2]
                for i in range(5):
                    self.randomspawn(obj_json_list, lane, long_range, lateral_range)


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
