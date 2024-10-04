#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import random

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
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
        env = ScenarioEnv(
            {
                #"manual_control": True,
                "agent_policy": ReplayEgoCarPolicy,
                "sequential_seed": True,
                "reactive_traffic": True if args.reactive_traffic else False,
                "use_render": True if not args.top_down else False,
                "data_directory": AssetLoader.file_path(
                    asset_path, "waymo" if use_waymo else "nuscenes", unix_style=False
                ),
                "num_scenarios": 3 if use_waymo else 10,
                #"debug": True,
                "use_bounding_box": True,
                "vehicle_config": {
                    "show_line_to_dest": True,
                    "show_line_to_navi_mark": True,
                },
                "disable_collision": True,
            }
        )
        o, _ = env.reset()

        depth_cam = DepthCamera(1600,1200, env.engine)
        depth_cam.lens.setFov(70)
        rgb_cam = RGBCamera(1600, 1200, env.engine)
        rgb_cam.lens.setFov(70)
        instance_cam = InstanceCamera(1600, 1200, env.engine)
        instance_cam.lens.setFov(70)
        semantic_cam = SemanticCamera(1600, 1200, env.engine)
        semantic_cam.lens.setFov(70)




        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([1.0, 0.])
            env.render(
                mode="top_down" if args.top_down else None,
                text=None if args.top_down else RENDER_MESSAGE,
                **extra_args
            )
            depth_capture    = depth_cam.perceive(to_float=False, new_parent_node=env.agent.origin)
            depth_capture = depth_cam.perceive(to_float=False, new_parent_node=env.agent.origin)
            rgb_capture      = rgb_cam.perceive(to_float=False, new_parent_node=env.agent.origin)
            semantic_capture = semantic_cam.perceive(to_float=False, new_parent_node=env.agent.origin)
            instance_capture = instance_cam.perceive(to_float=False, new_parent_node=env.agent.origin)

            cv2.imshow("Depth", depth_capture)
            cv2.imshow("RGB", rgb_capture)
            cv2.imshow("Semantic", semantic_capture)
            cv2.imshow("Instance", instance_capture)
            cv2.waitKey(1)


            if tm or tc:
                env.reset()
    finally:
        env.close()
