#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import random

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv, ScenarioDiverseEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
import cv2

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
    try:
        env = ScenarioDiverseEnv(
            {
                "agent_policy": ReplayEgoCarPolicy,
                "manual_control": True,
                "sequential_seed": True,
                "reactive_traffic": True if args.reactive_traffic else False,
                "use_render": True if not args.top_down else False,
                "data_directory": AssetLoader.file_path(
                    asset_path, "waymo" if use_waymo else "nuscenes", unix_style=False
                ),
                "sensors": dict(
                    rgb=(RGBCamera, 960, 540),
                    semantic=(SemanticCamera, 960, 540),
                    depth=(DepthCamera, 960, 540)
                ),
                "num_scenarios": 3 if use_waymo else 10
            }
        )
        o, _ = env.reset()

        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([1.0, 0.])
            env.render(
                mode="top_down" if args.top_down else None,
                text=None if args.top_down else RENDER_MESSAGE,
                **extra_args
            )
            #cv2.imwrite("./tmp/rgb.png", env.engine.sensors["rgb"].perceive(to_float=True, new_parent_node=env.agent.origin)*255)
            #cv2.imwrite("./tmp/depth.png",
            #            env.engine.sensors["depth"].perceive(to_float=True, new_parent_node=env.agent.origin)*255)
            #cv2.imwrite("./tmp/semantic.png",
            #            env.engine.sensors["semantic"].perceive(to_float=True, new_parent_node=env.agent.origin, hpr=[315, 0, 0])*255)

            #cv2.imshow("RGB",env.engine.sensors["rgb"].perceive(to_float=True, new_parent_node=env.agent.origin))
            cv2.imshow("DEPTH",
                       env.engine.sensors["depth"]
                       .perceive(to_float=True, new_parent_node=env.agent.origin, hpr=[45, 0, 0]))
            cv2.waitKey(1)
            #cv2.imshow("SEMANTIC", env.engine.sensors["semantic"].perceive(to_float=True, new_parent_node=env.agent.origin))
            #cv2.waitKey(1)
            if tm or tc:
                env.reset()
    finally:
        env.close()