#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import os.path
import random
import time
import glob
import numpy as np
import cv2
import re
from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, InterventionPolicy
import json


def extract_numbers(filename):
    # print(filename)
    pattern = r"(.*?)_(\d.*).jpg$"
    index = re.findall(pattern, filename)[-1][-1]
    return int(index)


def get_actions(action_buffer_path):
    action_buffer = json.load(open(action_buffer_path))
    duration = 5
    actions = []
    navigations = []
    scene = None
    for key, value in action_buffer.items():
        actions += [value["action"] for _ in range(duration)]
        navigations += [value["navigation"] for _ in range(duration)]
        scene = value["scene"]
    return scene, actions, navigations


def fine_som_observations(action_buffer_path):
    folder = os.path.dirname(action_buffer_path)
    obs = glob.glob(os.path.join(folder, "obs*.jpg"))
    fronts = glob.glob(os.path.join(folder, "front*.jpg"))
    obs_ordered = sorted(obs, key=extract_numbers)
    fronts_ordered = sorted(fronts, key=extract_numbers)
    return obs_ordered, fronts_ordered


def get_trajectory(env):
    """
    n,2 array
    """
    scenario = env.engine.data_manager.current_scenario
    ego_id = scenario["metadata"]["sdc_id"]
    ego_track = scenario["tracks"][ego_id]
    ego_traj = ego_track["state"]["position"][..., :2]
    return ego_traj


from PIL import Image

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
    data_directory = "E:/Bolei/scenarios"
    num_scenarios = 120
    model = "E:/closed_loops/qwen2_waymonusc"
    template = f"{model}/*/action_buffer.json"
    action_buffers = glob.glob(template)
    tmp = dict()
    navs = dict()
    ades = dict()
    obs = dict()
    fronts = dict()

    for action_buffer in action_buffers:
        file, actions, navigations = get_actions(action_buffer)
        obs_ordered, fronts_ordered = fine_som_observations(action_buffer)
        tmp[file] = actions
        navs[file] = navigations
        obs[file] = obs_ordered
        fronts[file] = fronts_ordered
    try:
        env = ScenarioEnv(
            {
                "sequential_seed": True,
                "use_render": False,
                "data_directory": data_directory,
                "num_scenarios": num_scenarios,
                "agent_policy": InterventionPolicy,
                "vehicle_config": dict(vehicle_model="static_default")
            }
        )
        for seed in range(num_scenarios):
            env.reset(seed)
            filename = env.engine.data_manager.current_scenario_file_name
            gt_trajectory = get_trajectory(env)
            if filename not in tmp.keys():
                continue
            actions = tmp[filename]
            navigations = navs[filename]
            observations = obs[filename]
            front_cams = fronts[filename]
            trajectories = []
            for idx, action in enumerate(actions):
                trajectories.append(env.agent.position)
                o, r, tm, tc, info = env.step(action)
            while len(trajectories) < gt_trajectory.shape[0]:
                trajectories.append(trajectories[-1])
            while len(trajectories) > gt_trajectory.shape[0]:
                trajectories.pop()
            print(len(trajectories))
            trajectory_array = np.array(trajectories)
            assert trajectory_array.shape == gt_trajectory.shape, (trajectory_array.shape, gt_trajectory.shape)
            ade = np.linalg.norm(trajectory_array - gt_trajectory, axis=1).mean()
            ades[filename] = float(ade)


    finally:
        json.dump(
            dict(
                ADES = ades,
                avgADE= sum(list(ades.values()))/len((list(ades.values())))
            ),
            open(f"{model}/ades.json", "w"),
            indent=2,
        )
        env.close()
