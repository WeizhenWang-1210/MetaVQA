#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.scenario import utils as sd_utils
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, InterventionPolicy
from metadrive.scenario.parse_object_state import parse_object_state
import time

RENDER_MESSAGE = {
    "Quit": "ESC",
    "Switch perspective": "Q or B",
    "Reset Episode": "R",
    "Keyboard Control": "W,A,S,D",
}
ACTION_MAPPING = {
            "left": [0.5, 0.5],
            "right": [-0.5, 0.5],
            "stop": [0, -0.2],
            "none": [0, 0.3]
}

def get_trajectory_info(engine):
    # Directly get trajectory from data manager
    trajectory_data = engine.data_manager.current_scenario["tracks"]
    sdc_track_index = str(engine.data_manager.current_scenario["metadata"]["sdc_id"])
    ret = []
    for i in range(len(trajectory_data[sdc_track_index]["state"]["position"])):
        ret.append(parse_object_state(
            trajectory_data[sdc_track_index],
            i,
        ))
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactive_traffic", action="store_true")
    parser.add_argument("--top_down", "--topdown", action="store_true")
    parser.add_argument("--waymo", action="store_true")
    args = parser.parse_args()
    extra_args = dict(film_size=(2000, 2000)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)
    data_directory = "C:\school\Bolei\cat\cat"
    scenario_summary, scenes, _ = sd_utils.read_dataset_summary(data_directory)
    try:
        env = ScenarioEnv(
            {
                "sequential_seed": True,
                "reactive_traffic": True if args.reactive_traffic else False,
                "use_render": True if not args.top_down else False,
                "data_directory": data_directory,
                "num_scenarios": len(scenes),
                "agent_policy": InterventionPolicy,
            }
        )
        o, _ = env.reset()
        traj = get_trajectory_info(env.engine)
        max_step = len(traj)
        destination = traj[max_step - 1]['position']
        step = 0
        t = 0
        for i in range(1, 100000):
            o, r, tm, tc, info = env.step(ACTION_MAPPING["left"])
            step += 1
            t += 1
            if t % 5 == 0:
                time.sleep(5)
            env.render(
                mode="top_down" if args.top_down else None,
                text=None if args.top_down else RENDER_MESSAGE,
                **extra_args
            )
            if step > max_step:
                env.reset()
                step = 0
            if (tm or tc) and not info["out_of_road"]:
                env.reset()
                step=0


    finally:
        env.close()
