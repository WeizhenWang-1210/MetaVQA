#!/usr/bin/env python
"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import argparse
import logging
import os
import random

import PIL.Image
import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE
import torch
from collections import deque
class Buffer:
    def __init__(self, size=3):
        self.dq = deque(maxlen=size)
        self.size = size
        self.shape = (640,960,3)

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
        concatenated_img = np.concatenate(img_list,axis = 1)
        #print(concatenated_img)
        cv2.imshow("Debug",concatenated_img)
        cv2.waitKey(1)
    def export(self):
        imgs = self.read()
        for i, obs in enumerate(imgs):
            if obs is None:
                imgs[i] = np.zeros(self.shape)
        for i, obs in enumerate(imgs):
            #print(obs.shape)
            imgs[i] = obs.astype(np.uint8)

        stacked = np.stack(imgs,axis=0)
        """ path = os.path.join(os.getcwd(),"out.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', 'MP4V', 'X264'
        out = cv2.VideoWriter(path, fourcc, 20.0, (self.shape[0], self.shape[1]))
        for img in imgs:
            out.write(img)
        out.release()"""
        return stacked#.transpose((3, 0, 1, 2))


from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
def use_llava(vid):
    disable_torch_init()
    video = vid
    print(video,"here")
    inp = 'Describe the scenario.'
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit,
                                                           device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs, model.device)


if __name__ == "__main__":
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.engine.asset_loader import AssetLoader
    asset_path = AssetLoader.asset_path
    env_config = {
        "sequential_seed": True,
        "reactive_traffic": True,
        "use_render": False,
        "data_directory": AssetLoader.file_path(
                    asset_path, "nuscenes", unix_style=False
                ),
        "num_scenarios": 3,
        "agent_policy": ReplayEgoCarPolicy,
        "sensors": dict(
            rgb=(RGBCamera, 960, 640),
        )
    }
    print("Finished reading")
    env = ScenarioEnv(env_config)
    try:
        o, _ = env.reset()
        env.agent.expert_takeover = True
        camera = env.engine.get_sensor("rgb")
        buffer = Buffer(256)
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            img = camera.perceive(False, env.agent.origin, [0, -6, 2], [0, -0.5, 0])
            #print(img.shape)
            buffer.insert(img)
            vid = buffer.export()
            #print(len(vid))
            use_llava(vid)
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_agent.expert_takeover = True
    finally:
        env.close()
