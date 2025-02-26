#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import random

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.scenario import utils as sd_utils
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from collections import deque
import numpy as np
import cv2
import imageio
import os
import json

RENDER_MESSAGE = {
    "Quit": "ESC",
    "Switch perspective": "Q or B",
    "Reset Episode": "R",
    "Keyboard Control": "W,A,S,D",
}



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
    data_dir = "D:\scenarios"
    scenario_summary, _, _ = sd_utils.read_dataset_summary(data_dir)
    try:
        env = ScenarioEnv(
            {
                "sequential_seed": True,
                "reactive_traffic": True if args.reactive_traffic else False,
                "use_render": True if not args.top_down else False,
                "data_directory": data_dir,
                "num_scenarios": 120,
                "agent_policy": ReplayEgoCarPolicy,
            }
        )

        #traj = json.load(open("F:/traj.json","r"))
        for seed in range(120):
            o, _ = env.reset(seed)
            run  = True
            while run:
                o, r, tm, tc, info = env.step([0,0])
                env.render(
                    mode="top_down", target_agent_heading_up=True
                )
                #env.render(
                #    mode="top_down" if args.top_down else None,
                #    text=None if args.top_down else RENDER_MESSAGE,
                #    **extra_args
                #)
                if tm or tc:
                    run = False
            #if env.engine.data_manger.current_scenario_filename not in traj["gts"].keys(
    finally:
        env.close()
