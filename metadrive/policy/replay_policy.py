import logging

from metadrive.policy.base_policy import BasePolicy
from metadrive.scenario.parse_object_state import parse_object_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplayTrafficParticipantPolicy(BasePolicy):
    """
       Replay policy from Real data. For adding new policy, overwrite get_trajectory_info()
       This policy is designed for Waymo Policy by default
       """
    DEBUG_MARK_COLOR = (3, 140, 252, 255)

    def __init__(self, control_object, track, random_seed=None):
        super(ReplayTrafficParticipantPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.start_index = 0
        self._velocity_local_frame = False
        self.traj_info = self.get_trajectory_info(track)

    @property
    def is_current_step_valid(self):
        return self.traj_info[self.episode_step] is not None

    def get_trajectory_info(self, track):
        ret = []
        for i in range(self.engine.data_manager.current_scenario_length):
            # a trick for saving computation
            if i < self.episode_step:
                ret.append(None)
            else:
                state = parse_object_state(track, i)
                if not state["valid"]:
                    ret.append(None)
                else:
                    ret.append(state)
        return ret

    def act(self, *args, **kwargs):
        index = max(int(self.episode_step), 0)
        if index >= len(self.traj_info):
            return None

        info = self.traj_info[index]

        # Before step
        # Warning by LQY: Don't call before step here! Before step should be called by manager
        # action = self.traj_info[int(self.episode_step)].get("action", None)
        # self.control_object.before_step(action)

        if not bool(info["valid"]):
            return None  # Return None action so the base vehicle will not overwrite the steering & throttle

        if "throttle_brake" in info:
            if hasattr(self.control_object, "set_throttle_brake"):
                self.control_object.set_throttle_brake(float(info["throttle_brake"].item()))
        if "steering" in info:
            if hasattr(self.control_object, "set_steering"):
                self.control_object.set_steering(float(info["steering"].item()))
        self.control_object.set_position(info["position"])
        self.control_object.set_velocity(info["velocity"], in_local_frame=self._velocity_local_frame)
        self.control_object.set_heading_theta(info["heading"])
        self.control_object.set_angular_velocity(info["angular_velocity"])

        return None  # Return None action so the base vehicle will not overwrite the steering & throttle


class ReplayEgoCarPolicy(ReplayTrafficParticipantPolicy):
    def get_trajectory_info(self, trajectory):
        # Directly get trajectory from data manager
        trajectory_data = self.engine.data_manager.current_scenario["tracks"]
        sdc_track_index = str(self.engine.data_manager.current_scenario["metadata"]["sdc_id"])
        ret = []
        for i in range(len(trajectory_data[sdc_track_index]["state"]["position"])):
            ret.append(parse_object_state(
                trajectory_data[sdc_track_index],
                i,
            ))
        return ret


class InterventionPolicy(ReplayEgoCarPolicy):
    def act(self, *args, **kwargs):
        # print(self.engine.external_actions)
        index = max(int(self.episode_step), 0)
        if index >= len(self.traj_info):
            print("no more trajectories")
            return None

        info = self.traj_info[index]
        # Before step
        # Warning by LQY: Don't call before step here! Before step should be called by manager
        # action = self.traj_info[int(self.episode_step)].get("action", None)
        # self.control_object.before_step(action)

        if not bool(info["valid"]):
            return None  # Return None action so the base vehicle will not overwrite the steering & throttle

        if "throttle_brake" in info:
            if hasattr(self.control_object, "set_throttle_brake"):
                self.control_object.set_throttle_brake(float(info["throttle_brake"].item()))
        if "steering" in info:
            if hasattr(self.control_object, "set_steering"):
                self.control_object.set_steering(float(info["steering"].item()))

        intervention = self.engine.external_actions is not None and 'default_agent' in self.engine.external_actions.keys() \
                       and not (self.engine.external_actions['default_agent'][0] == 0. \
                                and self.engine.external_actions['default_agent'][1] == 0)
        if not intervention:
            self.control_object.set_position(info["position"])
            self.control_object.set_velocity(info["velocity"], in_local_frame=self._velocity_local_frame)
            self.control_object.set_heading_theta(info["heading"])
            self.control_object.set_angular_velocity(info["angular_velocity"])
            return None
        else:
            return self.engine.external_actions['default_agent']

