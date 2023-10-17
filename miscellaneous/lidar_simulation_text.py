"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import argparse
import random
import json
import numpy as np
from numpy.linalg import inv
import pygame
from metadrive import MetaDriveEnv
import time
import cv2
import os
import re
import math
from panda3d.core import LPoint3f
from metadrive.constants import CamMask
from metadrive.constants import HELP_MESSAGE
from utils_testing import sample_bbox
from panda3d.core import NodePath
from direct.showbase.Loader import Loader
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.utils import get_object_from_node
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.static_object.traffic_object import TrafficBarrier, TrafficCone, TrafficWarning

def cloud_points(img_array, H,W, hfov, vfov):
    """
    W: num pixel in width; H: num pixel in Height.
    hfov: horizontal fov in deg; vfov: vertical fov in deg
    """
    def find_f(H,W,hfov,vfov):
        fx = (W/2)/(math.tan(np.deg2rad(hfov)/2))
        fy = (H/2)/(math.tan(np.deg2rad(vfov)/2))
        return fx,fy
    fx,fy = find_f(H,W,hfov,vfov)
    

    def pix2vox(img,fx,fy,H,W):
        def val2dist(img):
            return np.exp(np.log(16) * img/255)*5
        distance_buffer = val2dist(img)
        H,W = img.shape
        def convert_from_uvd( u, v, d, cx,cy,focalx,focaly):
            x_over_z = (u-cx) / focalx
            y_over_z = (v-cy) / focaly
            z = d / np.sqrt( 1+x_over_z**2 + y_over_z**2)
            x = x_over_z * z
            y = y_over_z * z
            return x, y, z
        points = []
        for v in range(H):
            for u in range(W):
                point = convert_from_uvd(u,v,distance_buffer[v,u],W/2,H/2,fx,fy)
                points.append(point)
        return np.asarray(points)
    vox = pix2vox(img_array,fx,fy,H,W)
    vox = vox[vox[:,2]<=50]
    #vox = vox[vox[:,1]<height_offset]
    vox = vox.tolist()
    random.shuffle(vox)
    return vox[:512]





if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        manual_control=True,
        traffic_density=0.6,
        num_scenarios=100,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        need_inverse_traffic = True,
        #debug=True,
        #debug_static_world=True,
        map=4,  # seven block
        start_seed=random.randint(0, 100),
        vehicle_config = {"image_source":"depth_camera", "depth_camera":(84,84, True),"show_navi_mark":False},
        #show_coordinates = True
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="rgb_camera", choices=["lidar", "rgb_camera"])
    parser.add_argument("--num_instance", type = int, default = 100)
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(dict(image_observation=True))
    env = MetaDriveEnv(config)
    save = True
    try:
        
        o, _ = env.reset()
        print(HELP_MESSAGE)
        AssetLoader.init_loader(env.engine)
        loadeer = AssetLoader.get_loader()
        env.vehicle.expert_takeover = True
        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("The observation is an numpy array with shape: ", o.shape)
        cycle = 1000
        sample_nps = []
        for i in range(512):
            sample_np = loadeer.load_model(AssetLoader.file_path("models", "sphere.egg"))
            sample_np.setColor(0,0,1)
            sample_np.setScale(0.15,0.15,0.15)
            sample_np.hide(CamMask.AllOn)
            sample_nps.append(sample_np)
        for i in range(1, 9000000000):
            
            agents = env.engine.agents
            agent = list(agents.values())[0]
            """obs = env.engine.get_objects(lambda o: o.id != agent.id)
            #print(obs)
            for k, v  in obs.items():
                v.origin.hide(CamMask.MainCam)
            """
            o, r, tm, tc, info = env.step([0, 0])
            
            dcam = env.vehicle.get_camera(env.vehicle.config["image_source"])
            depth = dcam.get_image(env.vehicle)[..., -1]    
            hfov,vfov = dcam.lens.fov 
            """cv2.imshow("dcam", depth)
            cv2.waitKey(1)
            cv2.imwrite("depth.png",depth)"""
            points  = cloud_points(depth,84,84,hfov,vfov)
            if len(points)>0:
                    for i in range(len(points)):
                        sample_np = sample_nps[i]
                        sample_np.show(CamMask.MainCam)
                        sample_np.reparentTo(dcam.cam)
                        sample_np.setPos(points[i][0],points[i][2], -points[i][1])
                        sample_np.wrtReparentTo(env.engine.render)

                        """wp = sample_np.getPos(env.engine.render)
                        sample_np.reparentTo(env.engine.render)
                        sample_np.setPos(wp)"""
                    """for i in range(len(points),10000):
                        sample_nps[i].hide(CamMask.AllOn)"""
            
            
                

            
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_vehicle.expert_takeover = True
    except Exception as e:
        raise e
    finally:
        env.close()