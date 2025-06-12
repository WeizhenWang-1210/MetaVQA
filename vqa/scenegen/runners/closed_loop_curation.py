#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import json

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario import utils as sd_utils

RENDER_MESSAGE = {
    "Quit": "ESC",
    "Switch perspective": "Q or B",
    "Reset Episode": "R",
    "Keyboard Control": "W,A,S,D",
}


def inspect():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactive_traffic", action="store_true")
    parser.add_argument("--top_down", "--topdown", action="store_true")
    parser.add_argument("--waymo", action="store_true")
    args = parser.parse_args()
    extra_args = dict(film_size=(2000, 2000)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)
    scenario_dir = "/data_weizhen/CAT"
    scenario_summary, _, _ = sd_utils.read_dataset_summary(scenario_dir)

    try:
        env = ScenarioEnv(
            {
                "sequential_seed": True,
                "reactive_traffic": False,
                "use_render": False,
                "data_directory": scenario_dir,
                "num_scenarios": len(scenario_summary),
                "agent_policy": ReplayEgoCarPolicy,
            }
        )
        records = dict()
        for seed in range(len(scenario_summary)):
            o, _ = env.reset(seed=seed)
            print(f"Working on scene-{seed}:{env.engine.data_manager.current_scenario_file_name}")
            run = True
            tmp = dict()
            step = 0
            collided = False
            while run:
                o, r, tm, tc, info = env.step([0, 0])
                step += 1
                if len(env.agent.crashed_objects) > 0 and not collided:
                    print(
                        f"Contains collision starting at {step}: scene-{seed}:{env.engine.data_manager.current_scenario_file_name}")
                    tmp["collision"] = True
                    tmp["collision_step"] = step
                    collided = True
                if tm or tc:
                    run = False
            print(f"Length is {step}: scene-{seed}:{env.engine.data_manager.current_scenario_file_name}")
            tmp["total_step"] = step
            records[env.engine.data_manager.current_scenario_file_name] = tmp
    finally:
        json.dump(
            records,
            open("/data_weizhen/CAT/summary.json", "w"),
            indent=2
        )
        env.close()


def filter():
    records = json.load(open("/data_weizhen/CAT/summary.json", "r"))
    selected_scenarios = set()
    for scene, record in records.items():
        if record["total_step"] >= 30 and "collision" in record.keys():
            selected_scenarios.add(scene)
    string = ";\n".join(list(selected_scenarios))
    with open("/data_weizhen/CAT/selected.txt", "w") as f:
        f.write(string)
        f.close()


import os
import shutil


def curate_cat():
    records = json.load(open("/data_weizhen/CAT/summary.json", "r"))
    src_dir = "/data_weizhen/CAT"
    target_dir = "/data_weizhen/scenarios"
    os.makedirs(target_dir, exist_ok=True)
    for filename, record in records.items():
        if record["total_step"] >= 30 and "collision" in record.keys():
            shutil.copy(
                os.path.join(src_dir, filename),
                os.path.join(target_dir, filename)
            )


if __name__ == "__main__":
    curate_cat()
