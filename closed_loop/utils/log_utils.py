from closed_loop.closed import RECORD_BUFFER
from closed_loop.navigation import dynamic_get_navigation_signal
from closed_loop.utils.prompt_utils import observe_som, observe


def capture_som(env):
    obs, id2label = observe_som(env)
    front = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, -15, 3], [0, -0.8, 0])
    RECORD_BUFFER[env.current_seed][env.engine.episode_step] = dict(action=None, front=front, obs=obs,
                                                                    navigation=dynamic_get_navigation_signal(
                                                                        env.engine.data_manager.current_scenario,
                                                                        timestamp=env.episode_step, env=env),
                                                                    state=[env.agent.position, env.agent.heading,
                                                                           env.agent.speed])
    return obs, front, id2label


def capture(env):
    obs = observe(env)
    front = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, -15, 3], [0, -0.8, 0])
    RECORD_BUFFER[env.current_seed][env.engine.episode_step] = dict(action=None, front=front, obs=obs,
                                                                    navigation=dynamic_get_navigation_signal(
                                                                        env.engine.data_manager.current_scenario,
                                                                        timestamp=env.episode_step, env=env),
                                                                    state=[env.agent.position, env.agent.heading,
                                                                           env.agent.speed])
    return obs, front
