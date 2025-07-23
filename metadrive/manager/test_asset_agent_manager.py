import vqa.vqagen.utils.qa_utils
from metadrive.engine.logger import get_logger
from metadrive.manager.agent_manager import VehicleAgentManager
from metadrive.policy.idm_policy import TrajectoryIDMPolicy

logger = get_logger()


class TestAssetAgentManager(VehicleAgentManager):
    """
    This class maintain the relationship between active agents in the environment with the underlying instance
    of objects.

    Note:
    agent name: Agent name that exists in the environment, like default_agent, agent0, agent1, ....
    object name: The unique name for each object, typically be random string.
    """
    INITIALIZED = False  # when vehicles instances are created, it will be set to True

    def __init__(self, init_observations, test_asset_meta_info, initpos):
        """
        The real init is happened in self.init(), in which super().__init__() will be called
        """
        super(TestAssetAgentManager, self).__init__(init_observations)
        self.test_asset_meta_info = test_asset_meta_info
        self.saved_test_asset_obj = None
        self.initpos = initpos

    def set_test_asset_config_dict(self, newdict):
        self.test_asset_meta_info = newdict
        if self.saved_test_asset_obj is not None:
            remove_dict = self.spawned_objects
            for vals in list(remove_dict.values()):
                self._remove_vehicle(vals)
            self.episode_created_agents = self._create_agents(
                config_dict=self.engine.global_config["agent_configs"], test_asset_meta_info=self.test_asset_meta_info
            )

    def _create_agents(self, config_dict: dict, test_asset_meta_info: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        for agent_id, v_config in config_dict.items():
            v_type = random_vehicle_type(self.np_random) if self.engine.global_config["random_agent_model"] else \
                vehicle_type[v_config["vehicle_model"] if vqa.vqagen.utils.qa_utils.get("vehicle_model", False) else "default"]

            obj_name = agent_id if self.engine.global_config["force_reuse_object_name"] else None

            # Note: we must use force spawn
            if vqa.vqagen.utils.qa_utils.get("vehicle_model", False) and v_config["vehicle_model"] == "test":
                obj = self.spawn_object(
                    v_type,
                    position=self.initpos,
                    heading=0,
                    vehicle_config=v_config,
                    name=obj_name,
                    force_spawn=True,
                    test_asset_meta_info=test_asset_meta_info
                )
                self.saved_test_asset_obj = obj
            else:
                obj = self.spawn_object(v_type, vehicle_config=v_config, name=obj_name)
            ret[agent_id] = obj
            policy_cls = self.agent_policy
            args = [obj, self.generate_seed()]
            if policy_cls == TrajectoryIDMPolicy or issubclass(policy_cls, TrajectoryIDMPolicy):
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
        self.episode_created_agents = self._create_agents(
            config_dict=self.engine.global_config["agent_configs"], test_asset_meta_info=self.test_asset_meta_info
        )

    def _remove_vehicle(self, vehicle):
        vehicle_name = vehicle.name
        # assert vehicle_name not in self._active_objects
        self.clear_objects([vehicle_name])
        if vehicle_name in self._active_objects:
            v = self._active_objects.pop(vehicle_name)
            self._agents_finished_this_frame[self._object_to_agent[vehicle_name]] = v.name
            # self._check()
        if vehicle_name in self._object_to_agent:
            self._agent_to_object.pop(self._object_to_agent[vehicle_name])
            self._object_to_agent.pop(vehicle_name)
