import copy
from metadrive.engine.logger import get_logger
from metadrive.utils.math import norm
import numpy as np
import os
import json
import random
from asset.read_config import configReader
from metadrive.component.static_object.traffic_object import TrafficCone, TrafficBarrier
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.vehicle.vehicle_type import CustomizedCar
from metadrive.component.vehicle.vehicle_type import SVehicle, LVehicle, MVehicle, XLVehicle, \
    TrafficDefaultVehicle
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.scenario.parse_object_state import parse_object_state, get_idm_route, get_max_valid_indicis
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from metadrive.utils.math import wrap_to_pi

logger = get_logger()


class ScenarioDiverseTrafficManager(ScenarioTrafficManager):
    def __init__(self):
        super(ScenarioDiverseTrafficManager, self).__init__()
        self.config = configReader()
        self.path_config = self.config.loadPath()
        self.adj_folder = self.path_config['adj_parameter_folder']
        self.init_car_adj_list()
        self.original_car_ratio = 0.4

    def init_car_adj_list(self):
        self.car_asset_metainfos = []  # List to store the file paths
        for root, dirs, files in os.walk(self.adj_folder):
            for file in files:
                if file.lower().startswith("car"):
                    with open(os.path.join(root, file), 'r') as file:
                        loaded_metainfo = json.load(file)
                        print(loaded_metainfo)
                        self.car_asset_metainfos.append(loaded_metainfo)

    def randomCustomizedCar(self):
        return CustomizedCar, random.choice(self.car_asset_metainfos)

    def spawn_vehicle(self, v_id, track):
        state = parse_object_state(track, self.episode_step)

        # for each vehicle, we would like to know if it is static
        if v_id not in self._static_car_id and v_id not in self._moving_car_id:
            valid_points = track["state"]["position"][np.where(track["state"]["valid"])]
            moving = np.max(np.std(valid_points, axis=0)[:2]) > self.STATIC_THRESHOLD
            set_to_add = self._moving_car_id if moving else self._static_car_id
            set_to_add.add(v_id)

        # don't create in these two conditions
        if not state["valid"] or (self.engine.global_config["no_static_vehicles"] and v_id in self._static_car_id):
            return

        # if collision don't generate, unless ego car is in replay mode
        ego_pos = self.ego_vehicle.position
        heading_dist, side_dist = self.ego_vehicle.convert_to_local_coordinates(state["position"], ego_pos)
        if not self.is_ego_vehicle_replay and self._filter_overlapping_car and \
                abs(heading_dist) < self.GENERATION_FORWARD_CONSTRAINT and \
                abs(side_dist) < self.GENERATION_SIDE_CONSTRAINT:
            return

        # create vehicle
        use_original_vehicle = random.random() < self.original_car_ratio
        if use_original_vehicle:
            if state["vehicle_class"]:
                vehicle_class = state["vehicle_class"]
            else:
                vehicle_class = get_vehicle_type(
                    float(state["length"]), None if self.even_sample_v else self.np_random, self.need_default_vehicle
                )
        else:
            vehicle_class, asset_info = self.randomCustomizedCar()
        obj_name = v_id if self.engine.global_config["force_reuse_object_name"] else None
        v_cfg = copy.copy(self._traffic_v_config)
        if self.engine.global_config["top_down_show_real_size"]:
            v_cfg["top_down_length"] = track["state"]["length"][self.episode_step]
            v_cfg["top_down_width"] = track["state"]["width"][self.episode_step]
            if v_cfg["top_down_length"] < 1 or v_cfg["top_down_width"] < 0.5:
                logger.warning(
                    "Scenario ID: {}. The top_down size of vehicle {} is weird: "
                    "{}".format(self.engine.current_seed, v_id, [v_cfg["length"], v_cfg["width"]])
                )
        if use_original_vehicle:
            v = self.spawn_object(
                vehicle_class,
                position=state["position"],
                heading=state["heading"],
                vehicle_config=v_cfg,
                name=obj_name
            )
        else:
            v = self.spawn_object(
                vehicle_class,
                position=state["position"],
                heading=state["heading"],
                vehicle_config=v_cfg,
                name=obj_name,
                test_asset_meta_info=asset_info
            )
        self._scenario_id_to_obj_id[v_id] = v.name
        self._obj_id_to_scenario_id[v.name] = v_id

        # add policy
        start_index, end_index = get_max_valid_indicis(track, self.episode_step)  # real end_index is end_index-1
        moving = track["state"]["position"][start_index][..., :2] - track["state"]["position"][end_index - 1][..., :2]
        length_ok = norm(moving[0], moving[1]) > self.IDM_CREATE_MIN_LENGTH
        heading_ok = abs(wrap_to_pi(self.ego_vehicle.heading_theta - state["heading"])) < np.pi / 2
        idm_ok = heading_dist < self.IDM_CREATE_FORWARD_CONSTRAINT and abs(
            side_dist
        ) < self.IDM_CREATE_SIDE_CONSTRAINT and heading_ok
        need_reactive_traffic = self.engine.global_config["reactive_traffic"]
        if not need_reactive_traffic or v_id in self._static_car_id or not idm_ok or not length_ok:
            policy = self.add_policy(v.name, ReplayTrafficParticipantPolicy, v, track)
            policy.act()
        else:
            idm_route = get_idm_route(track["state"]["position"][start_index:end_index][..., :2])
            # only not static and behind ego car, it can get reactive policy
            self.add_policy(
                v.name, TrajectoryIDMPolicy, v, self.generate_seed(), idm_route,
                self.idm_policy_count % self.IDM_ACT_BATCH_SIZE
            )
            # no act() is required for IDMPolicy
            self.idm_policy_count += 1


type_count = [0 for i in range(3)]


def get_vehicle_type(length, np_random=None, need_default_vehicle=False):
    if np_random is not None:
        if length <= 4:
            return SVehicle
        elif length <= 5.5:
            return [LVehicle, SVehicle, MVehicle][np_random.randint(3)]
        else:
            return [LVehicle, XLVehicle][np_random.randint(2)]
    else:
        global type_count
        # evenly sample
        if length <= 4:
            return SVehicle
        elif length <= 5.5:
            type_count[1] += 1
            vs = [LVehicle, MVehicle, SVehicle]
            # vs = [SVehicle, LVehicle, MVehicle]
            if need_default_vehicle:
                vs.append(TrafficDefaultVehicle)
            return vs[type_count[1] % len(vs)]
        else:
            type_count[2] += 1
            vs = [LVehicle, XLVehicle]
            return vs[type_count[2] % len(vs)]


def reset_vehicle_type_count(np_random=None):
    global type_count
    if np_random is None:
        type_count = [0 for i in range(3)]
    else:
        type_count = [np_random.randint(100) for i in range(3)]
