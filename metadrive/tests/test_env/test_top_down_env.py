import numpy as np
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.envs.top_down_env import TopDownSingleFrameMetaDriveEnv, TopDownMetaDrive, TopDownMetaDriveEnvV2


def test_top_down_rendering():
    for env in [
            TopDownSingleFrameMetaDriveEnv(dict(num_scenarios=5, map="C", traffic_density=1.0)),
            TopDownMetaDrive(dict(num_scenarios=5, map="C", traffic_density=1.0)),
            TopDownMetaDrive(dict(num_scenarios=5, map="C", frame_stack=1, post_stack=2)),
            TopDownMetaDriveEnvV2(dict(num_scenarios=5, map="C", frame_stack=1, post_stack=2)),
            WaymoEnv(dict(num_scenarios=1, start_scenario_index=0)),
            WaymoEnv(dict(num_scenarios=1, start_scenario_index=1)),
            WaymoEnv(dict(num_scenarios=1, start_scenario_index=2)),
    ]:
        try:
            for _ in range(5):
                o, _ = env.reset()
                assert np.mean(o) > 0.0
                for _ in range(10):
                    o, *_ = env.step([0, 1])
                    assert np.mean(o) > 0.0
                for _ in range(10):
                    o, *_ = env.step([-0.05, 1])
                    assert np.mean(o) > 0.0
        finally:
            env.close()


def _vis_top_down_with_panda_render():
    env = TopDownMetaDrive(dict(use_render=True))
    try:
        o, _ = env.reset()
        for i in range(1000):
            o, r, tm, tc, i = env.step([0, 1])
            if tm or tc:
                break
    finally:
        env.close()


def _vis_top_down_with_panda_render_and_top_down_visualization():
    env = TopDownMetaDrive({"use_render": True})
    try:
        o, _ = env.reset()
        for i in range(2000):
            o, r, tm, tc, i = env.step([0, 1])
            if tm or tc:
                break
            env.render(mode="top_down")
    finally:
        env.close()


if __name__ == "__main__":
    # test_top_down_rendering()
    # _vis_top_down_with_panda_render()
    _vis_top_down_with_panda_render_and_top_down_visualization()
