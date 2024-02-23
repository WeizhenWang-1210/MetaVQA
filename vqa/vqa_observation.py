from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.depth_camera import DepthCamera
import cv2
import gymnasium as gym
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.state_obs import StateObservation
import os

class MyObservation(BaseObservation):
    def __init__(self, config):
        super(MyObservation, self).__init__(config)
        self.rgb = ImageObservation(config, "rgb", config["norm_pixel"])
        self.instance = ImageObservation(config, "instance", config["norm_pixel"])

    @property
    def observation_space(self):
        os={o: getattr(self, o).observation_space for o in ["rgb", "instance"]}
        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        os={o: getattr(self, o).observe() for o in ["rgb", "instance"]}
        return os