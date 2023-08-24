import copy
from metadrive.policy.idm_policy import TrajectoryIDMPOlicy
from typing import Dict

from gymnasium.spaces import Box, Dict, MultiDiscrete, Discrete

from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.agent_manager import AgentManager
from metadrive.policy.AI_protect_policy import AIProtectPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy


class TestAssetAgentManager(AgentManager):
    """
    This class maintain the relationship between active agents in the environment with the underlying instance
    of objects.

    Note:
    agent name: Agent name that exists in the environment, like agent0, agent1, ....
    object name: The unique name for each object, typically be random string.
    """
    INITIALIZED = False  # when vehicles instances are created, it will be set to True

    def __init__(self, init_observations, init_action_space, test_asset_meta_info):
        """
        The real init is happened in self.init(), in which super().__init__() will be called
        """
        # BaseVehicles which can be controlled by policies when env.step() called
        super().__init__(init_observations, init_action_space)
        self.test_asset_meta_info = test_asset_meta_info
        self.saved_test_asset_obj = None
    def _remove_vehicle(self, vehicle):
        vehicle_name = vehicle.name
        # assert vehicle_name not in self._active_objects
        self.clear_objects([vehicle_name])
        if vehicle_name in self._object_to_agent:
            self._agent_to_object.pop(self._object_to_agent[vehicle_name])
            self._object_to_agent.pop(vehicle_name)
        if vehicle_name in self._active_objects:
            del self._active_objects[vehicle_name]
    def set_test_asset_config_dict(self, newdict):
        self.test_asset_meta_info = newdict
        if self.saved_test_asset_obj is not None:
            self.saved_test_asset_obj.update_asset_metainfo(newdict)
            remove_dict = self.spawned_objects
            for vals in list(remove_dict.values()):
                self._remove_vehicle(vals)
            self.episode_created_agents = self._get_vehicles(
                config_dict=self.engine.global_config["target_vehicle_configs"],
                test_asset_meta_info=self.test_asset_meta_info
            )
    def _get_vehicles(self, config_dict: dict, test_asset_meta_info: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        for agent_id, v_config in config_dict.items():
            v_type = random_vehicle_type(self.np_random) if self.engine.global_config["random_agent_model"] else \
                vehicle_type[v_config["vehicle_model"] if v_config.get("vehicle_model", False) else "default"]

            obj_name = agent_id if self.engine.global_config["force_reuse_object_name"] else None
            if v_config.get("vehicle_model", False) and v_config["vehicle_model"] == "test":
                obj = self.spawn_object(v_type, vehicle_config=v_config, name=obj_name, test_asset_meta_info=test_asset_meta_info)
                self.saved_test_asset_obj = obj
            else:
                obj = self.spawn_object(v_type, vehicle_config=v_config, name=obj_name)
            ret[agent_id] = obj
            policy_cls = self.agent_policy
            args = [obj, self.generate_seed()]
            if policy_cls == TrajectoryIDMPOlicy or issubclass(policy_cls, TrajectoryIDMPOlicy):
                args.append(self.engine.map_manager.current_sdc_route)
            self.add_policy(obj.id, policy_cls, *args)
        return ret
    def reset(self):
        """
        Agent manager is really initialized after the BaseVehicle Instances are created
        """
        self.random_spawn_lane_in_single_agent()
        config = self.engine.global_config
        self._debug = config["debug"]
        self._delay_done = config["delay_done"]
        self._infinite_agents = config["num_agents"] == -1
        self._allow_respawn = config["allow_respawn"]
        self.episode_created_agents = self._get_vehicles(
            config_dict=self.engine.global_config["target_vehicle_configs"],
            test_asset_meta_info=self.test_asset_meta_info
        )

    def propose_new_vehicle(self):
        # Create a new vehicle.
        agent_name = self.next_agent_id()
        next_config = self.engine.global_config["target_vehicle_configs"]["agent0"]
        vehicle = self._get_vehicles({agent_name: next_config})[agent_name]
        new_v_name = vehicle.name
        self._agent_to_object[agent_name] = new_v_name
        self._object_to_agent[new_v_name] = agent_name
        self.observations[new_v_name] = self._init_observations["agent0"]
        self.observations[new_v_name].reset(vehicle)
        self.observation_spaces[new_v_name] = self._init_observation_spaces["agent0"]
        self.action_spaces[new_v_name] = self._init_action_spaces["agent0"]
        self._active_objects[vehicle.name] = vehicle
        self._check()
        step_info = vehicle.before_step([0, 0])
        vehicle.set_static(False)
        return agent_name, vehicle, step_info
    def for_each_active_agents(self, func, *args, **kwargs):
        """
        This func is a function that take each vehicle as the first argument and *arg and **kwargs as others.
        """
        # assert len(self.active_agents) > 0, "Not enough vehicles exist!"
        ret = dict()
        for k, v in self.active_agents.items():
            ret[k] = func(v, *args, **kwargs)
        return ret