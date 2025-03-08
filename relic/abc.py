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
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from collections import deque
import numpy as np
import cv2
import imageio
import os
import json
from metadrive.component.sensors.depth_camera import DepthCamera


RENDER_MESSAGE = {
    "Quit": "ESC",
    "Switch perspective": "Q or B",
    "Reset Episode": "R",
    "Keyboard Control": "W,A,S,D",
}


class Buffer:
    def __init__(self, size=3, shape=(640, 960, 3)):
        self.dq = deque(maxlen=size)
        self.size = size
        self.shape = shape

    def insert(self, stuff):
        self.dq.append(stuff)

    def read(self):
        result = []
        for _ in range(self.size - len(self.dq)):
            result.append(None)
        for item in self.dq:
            result.append(item)
        return result

    def flush(self):
        self.dq.clear()

    def display(self):
        img_list = self.read()
        for i, obs in enumerate(img_list):
            if obs is None:
                img_list[i] = np.zeros(self.shape)
        concatenated_img = np.concatenate(img_list, axis=1)
        # print(concatenated_img)
        cv2.imshow("Debug", concatenated_img)
        cv2.waitKey(1)

    def export_nparray(self):
        img_list = self.read()
        for i, obs in enumerate(img_list):
            if obs is None:
                img_list[i] = np.zeros(self.shape)
        video_array = np.stack(img_list, axis=0)
        return video_array

    def export_frames(self, folder):
        os.makedirs(folder, exist_ok=True)
        for frame_id, im in enumerate(self.dq):
            frame_path = os.path.join(folder, str(frame_id))
            cv2.imwrite(f"{frame_path}.png", im)

    def export_video(self, filename):
        output_path = f'{filename}.mp4'
        fps = 10
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in self.dq:
            writer.append_data(frame[:, :, [2, 1, 0]])
        writer.close()


def record_accident(env, buffer, summary, countdown, camera):
    while countdown > 0:
        o, r, tm, tc, info = env.step([0, 0])
        im = camera.perceive(False, env.agent.origin, [0, -6, 2], [0, -0.5, 0])
        im_buffer.insert(im)
        env.render(
            mode="top_down" if args.top_down else None,
            text=None if args.top_down else RENDER_MESSAGE,
            **extra_args
        )
        if tm or tc:
            break
        countdown -= 1
    buffer.export_video("collision_{}_{}".format(env.current_seed, summary["incident_step"]))
    with open("collision_{}_{}.json".format(env.current_seed, summary["incident_step"]), "w") as file:
        json.dump(summary, file)
    return tm, tc


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
    true_path = "C:\school\Bolei\cat\cat"
    scenario_summary, _, _ = sd_utils.read_dataset_summary(
        true_path)

    try:
        env = ScenarioEnv(
            {
                "sequential_seed": True,
                "reactive_traffic": True if args.reactive_traffic else False,
                "use_render": False,#True if not args.top_down else False,
                "num_scenarios": len(scenario_summary),
                "agent_policy": ReplayEgoCarPolicy,
                "data_directory":true_path,
                "sensors": dict(
                    rgb=(RGBCamera, 960, 540),
                    instance=(InstanceCamera, 960, 540),
                    semantic=(SemanticCamera, 960, 540),
                    depth=(DepthCamera, 960, 540)
                ),

            }
        )
        o, _ = env.reset(seed=0)
        rgb_cam = env.engine.get_sensor("rgb")
        depth_cam = env.engine.get_sensor("depth")
        instance_cam = env.engine.get_sensor("instance")
        semantic_cam = env.engine.get_sensor("semantic")

        inception = False
        countdown = 5
        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([0, 0])
            #rgb = env.engine.get_sensor("rgb").perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5))
            #cv2.imshow("rgb_front.png", rgb)
            capture_1 = depth_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(0, 0, 0))
            capture_2 = depth_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(90, 0, 0))
            capture_3 = depth_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(180, 0, 0))
            capture_4 = depth_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(270, 0, 0))
            capture_5 = depth_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5))

            instance_1 =instance_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(0, 0, 0))
            instance_2 =instance_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(90, 0, 0))
            instance_3 =instance_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(180, 0, 0))
            instance_4 =instance_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(270, 0, 0))

            rgb_1 = rgb_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(0, 0, 0))
            rgb_2 = rgb_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(90, 0, 0))
            rgb_3 = rgb_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(180, 0, 0))
            rgb_4 = rgb_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(270, 0, 0))

            semantic_1 = semantic_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(0, 0, 0))
            semantic_2 = semantic_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(90, 0, 0))
            semantic_3 = semantic_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(180, 0, 0))
            semantic_4 = semantic_cam.perceive(new_parent_node=env.agent.origin, position=(0., 0.0, 1.5), hpr=(270, 0, 0))



            cv2.imshow("capture_1", capture_1)
            cv2.imshow("capture_2", capture_3)
            cv2.imshow("capture_3", capture_4)
            cv2.imshow("capture_4", capture_5)

            cv2.imshow("instance_1", instance_1)
            cv2.imshow("instance_2", instance_2)
            cv2.imshow("instance_3", instance_3)
            cv2.imshow("instance_4", instance_4)

            cv2.imshow("rgb_1", rgb_1)
            cv2.imshow("rgb_2", rgb_2)
            cv2.imshow("rgb_3", rgb_3)
            cv2.imshow("rgb_4", rgb_4)

            cv2.imshow("semantic_1", semantic_1)
            cv2.imshow("semantic_2", semantic_2)
            cv2.imshow("semantic_3", semantic_3)
            cv2.imshow("semantic_4", semantic_4)



            #cv2.imshow("capture_6", capture_6)
            print("here")
            if tm or tc:
                env.reset()
    finally:
        env.close()
