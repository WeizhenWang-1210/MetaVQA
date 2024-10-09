#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import PIL.Image
from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv, ScenarioDiverseEnv
from metadrive.scenario import utils as sd_utils
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from collections import deque
import torch
from vqa.online_eval import device
def load_and_process_images(images_lists, vis_processor):
    # image paths: 6x20(view x time) list of paths
    # return: 5x6x3x364x364 tensor (5 is the select time)
    # Define the views
    views = ["front", "leftf", "leftb", "rightf", "rightb", "back"]
    # Initialize a list to store processed images for each view
    # Iterate through the image paths
    all_views_images = [[] for _ in views]
    for i, images_list in enumerate(images_lists):
        for image in images_list:
            # for view_index, view_path in enumerate(view_paths):
            processed_image = vis_processor(image).to(device)
            all_views_images[i].append(processed_image)
    # Convert the lists of images into NumPy arrays
    all_views_images = [torch.stack(view_images, dim=0) for view_images in all_views_images]
    # Concatenate the processed images into the desired shape (t, views, height, width, channel)
    processed_images_tensor = torch.stack(all_views_images, dim=1)
    t, views, height, width, channel = processed_images_tensor.shape
    # If t is 1, repeat the tensor 20 times along the first dimension
    if t < 5:
        processed_images_tensor = processed_images_tensor.repeat(5-t+1, 1, 1, 1, 1)
    return processed_images_tensor


def vector_transform(origin, positive_x, point):
    def change_bases(x, y):
        relative_x, relative_y = x - origin[0], y - origin[1]
        new_x = positive_x
        new_y = (-new_x[1], new_x[0])
        x = (relative_x * new_x[0] + relative_y * new_x[1])
        y = (relative_x * new_y[0] + relative_y * new_y[1])
        return [x, y]

    return change_bases(*point)


class Buffer:
    def __init__(self, size=3, shape=(640, 960, 3)):
        self.dq = deque(maxlen=size)
        self.size = size
        self.shape = shape

    def insert(self, stuff):
        if len(self.dq) < self.size:
            self.dq.append(stuff)
        else:
            garbage = self.dq.popleft()
            del garbage
            self.dq.append(stuff)

    def read_tensor(self, vis_processor):
        """
        Export the content as a (5,6,3,364,364) processed tensor.
        """
        #data = []
        views = ["front", "leftf", "leftb", "rightf", "rightb", "back"]
        all_views_images = [[] for _ in views]
        """for _ in range(self.size - len(self.dq)):
            for idx,  perspective in enumerate(views):
                empty_img = np.zeros((3, 540, 960), dtype=np.uint8)
                img = PIL.Image.fromarray(empty_img)
                all_views_images[idx].append(img)"""
        for item, _, _ in self.dq:
            for idx, perspective in enumerate(views):
                img = PIL.Image.fromarray(item[perspective])
                all_views_images[idx].append(img)
        old_position, now_position = self.dq[0][1], self.dq[-1][1]
        now_heading = self.dq[-1][2]
        change = [now_position[0] - old_position[0], now_position[1] - old_position[1]]
        displacement = vector_transform(now_heading, now_heading, change)
        return load_and_process_images(all_views_images, vis_processor), displacement, now_position














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
    scenario_summary, _, _ = sd_utils.read_dataset_summary("")

    try:
        env = ScenarioEnv(
            {
                "sequential_seed": True,
                "reactive_traffic": True if args.reactive_traffic else False,
                "use_render": True if not args.top_down else False,
                "data_directory": "E:\Bolei\cat",
                "num_scenarios": len(scenario_summary),
                "agent_policy": ReplayEgoCarPolicy,
            }
        )
        o, _ = env.reset()
        camera = RGBCamera(960, 540, env.engine)
        im_buffer = Buffer(5)
        summary = {
            "incident_step": None,
            "incident_obj": None
        }
        inception = False
        countdown = 5
        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([0, 0])
            im = camera.perceive(False, env.agent.origin, [0, -6, 2], [0, -0.5, 0])
            im_buffer.insert(im)
            if not inception and len(env.agent.crashed_objects) > 0:
                print("Collision happened at step {}".format(i))
                inception = True
                summary["incident_step"] = env.engine.episode_step
                summary["incident_obj"] = [obj for obj in env.agent.crashed_objects]
                tm, tc = record_accident(env, im_buffer, summary, countdown, camera)
            env.render(
                mode="top_down" if args.top_down else None,
                text=None if args.top_down else RENDER_MESSAGE,
                **extra_args
            )
            if tm or tc:
                env.reset()
                inception = False
                im_buffer.flush()
    finally:
        env.close()
