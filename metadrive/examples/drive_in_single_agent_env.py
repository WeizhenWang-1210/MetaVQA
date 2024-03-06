#!/usr/bin/env python
"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import argparse
import logging
import random

import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE

if __name__ == "__main__":

    from metadrive.engine.asset_loader import AssetLoader

    asset_path = AssetLoader.asset_path
    use_waymo = False
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy

    from metadrive.scenario import utils as sd_utils

    dataset_path = "/bigdata/datasets/scenarionet/waymo/training/training_0"

    print(f"Reading the summary file from Waymo data at: {dataset_path}")

    waymo_dataset_summary = sd_utils.read_dataset_summary(dataset_path)
    print(f"The dataset summary is a {type(waymo_dataset_summary)}, with lengths {len(waymo_dataset_summary)}.")
    waymo_scenario_summary, waymo_scenario_ids, waymo_scenario_files = waymo_dataset_summary

    print(len(waymo_scenario_ids))
    assert len(waymo_scenario_summary) == len(waymo_scenario_ids)
    print(waymo_scenario_files.keys())
    """ print(
        f"The scenario summary is a dict with keys: {waymo_scenario_summary.keys()} \nwhere each value of the dict is the summary of a scenario.\n")"""

    config = {
        "sequential_seed": True,
        "reactive_traffic": True,
        "use_render": False,
        "data_directory": dataset_path,
        "agent_policy": ReplayEgoCarPolicy,
        "num_scenarios": len(waymo_scenario_summary.keys())
    }


    """config = dict(
        # controller="steering_wheel",
        #use_render=True,

        #manual_control=True,
        traffic_density=0.1,
        num_scenarios=10000,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=True, show_navi_mark=False, show_line_to_navi_mark=False),
        # debug=True,
        # debug_static_world=True,
        map=4,  # seven block
        start_seed=10,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(rgb_camera=(RGBCamera, 400, 300)),
                interface_panel=["rgb_camera", "dashboard"]
            )
        )"""

    env = ScenarioEnv(config)
    try:
        o, _ = env.reset(seed=0)
        print(HELP_MESSAGE)
        env.agent.expert_takeover = True
        """if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("The observation is an numpy array with shape: ", o.shape)"""
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            print(f"step {i}")
            """env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Current Observation": args.observation,
                    "Keyboard Control": "W,A,S,D",
                }
            )

            if args.observation == "rgb_camera":
                cv2.imshow('RGB Image in Observation', o["image"][..., -1])
                cv2.waitKey(1)"""
            if (tm or tc) and info["arrive_dest"]:
                env.reset(env.current_seed + 1)
                env.current_track_agent.expert_takeover = True
    finally:
        env.close()
