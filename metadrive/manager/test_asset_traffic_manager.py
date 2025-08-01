import copy
import logging
from collections import namedtuple
from typing import Dict

import math
import os
import json
import random
import numpy as np
from asset.read_config import configReader
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.vehicle.vehicle_type import CustomizedCar
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.traffic_manager import PGTrafficManager
from metadrive.utils import merge_dicts

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class TrafficMode:
    # Traffic vehicles will be respawned, once they arrive at the destinations
    Respawn = "respawn"

    # Traffic vehicles will be triggered only once
    Trigger = "trigger"

    # Hybrid, some vehicles are triggered once on map and disappear when arriving at destination, others exist all time
    Hybrid = "hybrid"


class NewAssetPGTrafficManager(PGTrafficManager):
    VEHICLE_GAP = 10  # m

    def __init__(self):
        """
        Control the whole traffic flow
        """
        super(NewAssetPGTrafficManager, self).__init__()

        self._traffic_vehicles = []

        # triggered by the event. TODO(lqy) In the future, event trigger can be introduced
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = self.engine.global_config["traffic_mode"]
        self.random_traffic = self.engine.global_config["random_traffic"]
        self.density = self.engine.global_config["traffic_density"]
        self.respawn_lanes = None
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

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import custom_random_vehicle_type
        vehicle_type = custom_random_vehicle_type(self.np_random, [0.2, 0.2, 0.2, 0.2, 0.0, 0.2])
        return vehicle_type

    def randomCustomizedCar(self):
        return CustomizedCar, random.choice(self.car_asset_metainfos)

    def after_step(self, *args, **kwargs):
        """
        Update all traffic vehicles' states,
        """
        v_to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if not v.on_lane:
                if self.mode == TrafficMode.Trigger:
                    v_to_remove.append(v)
                elif self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
                    v_to_remove.append(v)
                else:
                    raise ValueError("Traffic mode error: {}".format(self.mode))
        for v in v_to_remove:
            vehicle_type = type(v)
            if vehicle_type == CustomizedCar:
                asset_metainfo = v.get_asset_metainfo()
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)
            if self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
                lane = self.respawn_lanes[self.np_random.randint(0, len(self.respawn_lanes))]
                lane_idx = lane.index
                long = self.np_random.rand() * lane.length / 2
                traffic_v_config = {"spawn_lane_index": lane_idx, "spawn_longitude": long}
                if vehicle_type == CustomizedCar:
                    new_v = self.spawn_object(
                        vehicle_type, vehicle_config=traffic_v_config, test_asset_meta_info=asset_metainfo
                    )
                else:
                    new_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                from metadrive.policy.idm_policy import IDMPolicy
                self.add_policy(new_v.id, IDMPolicy, new_v, self.generate_seed())
                self._traffic_vehicles.append(new_v)

        return dict()

    def _create_respawn_vehicles(self, map: BaseMap, traffic_density: float):
        total_num = len(self.respawn_lanes)
        for lane in self.respawn_lanes:
            _traffic_vehicles = []
            total_num = int(lane.length / self.VEHICLE_GAP)
            vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
            self.np_random.shuffle(vehicle_longs)
            for long in vehicle_longs[:int(np.ceil(traffic_density * len(vehicle_longs)))]:
                # if self.np_random.rand() > traffic_density and abs(lane.length - InRampOnStraight.RAMP_LEN) > 0.1:
                #     # Do special handling for ramp, and there must be vehicles created there
                #     continue
                use_original_vehicle = random.random() < self.original_car_ratio
                if use_original_vehicle:
                    vehicle_type = self.random_vehicle_type()
                else:
                    vehicle_type, asset_info = self.randomCustomizedCar()
                traffic_v_config = {"spawn_lane_index": lane.index, "spawn_longitude": long}
                traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
                if use_original_vehicle:
                    random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                else:
                    random_v = self.spawn_object(
                        vehicle_type, vehicle_config=traffic_v_config, test_asset_meta_info=asset_info
                    )
                from metadrive.policy.idm_policy import IDMPolicy
                self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                self._traffic_vehicles.append(random_v)

    def _create_vehicles_once(self, map: BaseMap, traffic_density: float) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        vehicle_num = 0
        for block in map.blocks[1:]:

            # Propose candidate locations for spawning new vehicles
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # How many vehicles should we spawn in this block?
            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            # Generate vehicles!
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]
            # print("We have {} candidates! We are spawning {} vehicles!".format(total_vehicles, len(selected)))

            from metadrive.policy.idm_policy import IDMPolicy
            for v_config in selected:
                use_original_vehicle = random.random() < self.original_car_ratio
                if use_original_vehicle:
                    vehicle_type = self.random_vehicle_type()
                    v_config.update(self.engine.global_config["traffic_vehicle_config"])
                    random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
                else:
                    vehicle_type, asset_info = self.randomCustomizedCar()
                    v_config.update(self.engine.global_config["traffic_vehicle_config"])
                    random_v = self.spawn_object(vehicle_type, vehicle_config=v_config, test_asset_meta_info=asset_info)
                self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()
