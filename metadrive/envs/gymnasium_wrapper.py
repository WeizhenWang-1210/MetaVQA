import gymnasium
from gymnasium.spaces import Box

from metadrive.constants import TerminationState
from metadrive.envs.base_env import BaseEnv


class GymnasiumEnvWrapper:
    def __init__(self, *args, **kwargs):
        super(GymnasiumEnvWrapper, self).__init__(*args, **kwargs)
        self._skip_env_checking = True

    def step(self, actions):
        o, r, d, i = super(GymnasiumEnvWrapper, self).step(actions)
        truncated = True if i[TerminationState.MAX_STEP] else False
        return o, r, d, truncated, i

    @property
    def observation_space(self):
        obs_space = super(GymnasiumEnvWrapper, self).observation_space
        return Box(low=obs_space.low, high=obs_space.high, shape=obs_space.shape)

    @property
    def action_space(self):
        space = super(GymnasiumEnvWrapper, self).action_space
        return Box(low=space.low, high=space.high, shape=space.shape)

    def reset(self, *, seed=None, options=None):
        return super(GymnasiumEnvWrapper, self).reset(), {}

    @classmethod
    def build(cls, base_class):
        assert issubclass(base_class, BaseEnv), "The base class should be the subclass of BaseEnv!"
        return type("{}({})".format(cls.__name__, base_class.__name__), (cls, base_class), {})


if __name__ == '__main__':
    from metadrive.envs.scenario_env import ScenarioEnv

    env = GymnasiumEnvWrapper.build(ScenarioEnv)({"manual_control": True})
    o, i = env.reset()
    assert isinstance(env.observation_space, gymnasium.Space)
    assert isinstance(env.action_space, gymnasium.Space)
    for s in range(600):
        o, r, d, t, i = env.step([0, -1])
        env.vehicle.set_velocity([0, 0])
        if d:
            assert s == env.config["horizon"] and i["max_step"] and t
            break
