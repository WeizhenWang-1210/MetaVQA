import copy
import logging
from collections import namedtuple
from typing import Dict

import math
import os
import json
import numpy as np
from asset.read_config import configReader
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
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


class TestAssetTrafficManager(BaseManager):
    VEHICLE_GAP = 10  # m

    SPAW_LANE_INDEX = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)
    def _propose_vehicle_configs(self, lane: AbstractLane):
        potential_vehicle_configs = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        # Only choose given number of vehicles
        for long in vehicle_longs:
            random_vehicle_config = {"spawn_lane_index": lane.index, "spawn_longitude": long, "enable_reverse": False}
            potential_vehicle_configs.append(random_vehicle_config)
        return potential_vehicle_configs
    def _init_additional_model(self):
        self.config = configReader()
        self.path_config = self.config.loadPath()
        self.adj_params_folder = self.path_config["adj_parameter_folder"]
        self.car_configs = []  # List to store the file paths
        for root, dirs, files in os.walk(self.adj_params_folder ):
            for file in files:
                if file.lower().startswith("car"):
                    os.path.join(root, file)
                    with open(os.path.join(root, file), 'r') as file:
                        loaded_metainfo = json.load(file)
                        loaded_metainfo["MODEL_SCALE"] = tuple(loaded_metainfo["MODEL_SCALE"])
                        loaded_metainfo["MODEL_OFFSET"] = tuple(loaded_metainfo["MODEL_OFFSET"])
                        loaded_metainfo["MODEL_SCALE"] = tuple(loaded_metainfo["MODEL_SCALE"])
                    self.car_configs.append(loaded_metainfo)
    def randomCustomizedCar(self):

    def _create_vehicles_once(self, map: BaseMap, traffic_density: float) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        vehicle_num = 0
        SPAW_LANE_INDEX = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)
        selected = [{"spawn_lane_index": SPAW_LANE_INDEX, "spawn_longitude": 0, "enable_reverse": False},
                    {"spawn_lane_index": SPAW_LANE_INDEX, "spawn_longitude": 10, "enable_reverse": False}]
        from metadrive.policy.idm_policy import IDMPolicy
        vehicles_on_block = []
        for v_config in selected:
            vehicle_type = self.random_vehicle_type()
            v_config.update(self.engine.global_config["traffic_vehicle_config"])
            random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
            self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
            vehicles_on_block.append(random_v.name)

        trigger_road = FirstPGBlock.pre_block_socket.positive_road
        block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)
        self.block_triggered_vehicles.append(block_vehicles)
        vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        # vehicle_type = random_vehicle_type(self.np_random, [0.2, 0.15, 0.3, 0.2, 0.0, 0.0, 0.0,0.15])
        vehicle_type = random_vehicle_type(self.np_random, [0,0,0,0,0,0,0,1])
        return vehicle_type