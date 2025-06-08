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
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.env_input_policy import EnvInputPolicy

Action_Mapping = {
    "a": [0.5, 0.8],  # roughly one second to change a lane.
    "b": [-0.5, 0.8],  # roughly one second to change to right lane.
    "c": [0, -0.135],  # brake deceleration = 0.3g
    "d": [0, -0.26],  # brake immediately deceleration = 0.55g
    "e": [0, 0.15]  # keep speed
}

if __name__ == "__main__":
    config = dict(
        # controller="steering_wheel",
        # use_render=True,
        traffic_density=0.1,
        num_scenarios=10000,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=False,
        vehicle_config=dict(show_lidar=True, show_navi_mark=False, show_line_to_navi_mark=False,
                            vehicle_model="static_default"),
        map=4,  # seven block
        agent_policy=EnvInputPolicy,
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
        )
    env = MetaDriveEnv(config)
    trajectories = {}

    limits = [4.47, 13.41, 22.35, 60]
    intervals = []
    l = 0
    for limit in limits:
        interval = np.linspace(l, limit, 10, endpoint=False).tolist()
        interval = [round(val,3) for val in interval]
        intervals += interval
        print(interval)
        l = limit
    initial_speeds = intervals
    controls = ["a", "b", "c", "d", "e"]
    durations = [5, 10, 15, 20]
    import itertools
    import tqdm

    for setting in tqdm.tqdm(itertools.product(controls, durations, initial_speeds), desc="Logging", unit="setting"):
        o, _ = env.reset(seed=21)
        action, duration, initial_speed = setting[0], setting[1], setting[2]
        print(action, duration, initial_speed)
        control_signal = Action_Mapping[action]
        env.reset(seed=21)
        waypoints = []
        speeds = []
        headings = []
        boxes = []
        env.agent.set_velocity(direction=np.array(env.agent.heading), value=initial_speed)
        for t in range(duration):
            waypoints.append(env.agent.position)
            speeds.append(env.agent.speed)
            headings.append(env.agent.heading)
            box = env.agent.bounding_box
            box = [vertex.tolist() for vertex in box]
            boxes.append(box)
            o, r, tm, tc, info = env.step(control_signal)
        setting = "_".join([str(c) for c in setting])
        trajectories[setting] = dict(
            waypoints=waypoints, speeds=speeds, headings=headings, boxs=[boxes]
        )
    import json

    json.dump(
        trajectories, open("trajectories_collection.json", "w"), indent=2, sort_keys=True
    )
