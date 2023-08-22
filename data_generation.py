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
import pygame
from metadrive import MetaDriveEnv
import cv2
import os
import re
import math
from panda3d.core import CollisionNode, CollisionBox,Point3,CollisionTraverser,CollisionHandlerQueue
from panda3d.bullet import BulletConvexHullShape
from metadrive.constants import HELP_MESSAGE
from utils_testing import sample_bbox
from panda3d.core import NodePath
def generate(config, max = 100):
    count = 0
    env = MetaDriveEnv(config)
    try:
        o, _ = env.reset()
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        agents = env.engine.agents
        agent = list(agents.values())[0] #if single-agent setting
        agent_id = list(agents.values())[0].id
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
            if i % 20== 0: #note: the "1st" object in objects is the agent
                count += 1
                objects = env.engine.get_objects()
                objects_of_interest = []
                for id, object in objects.items():
                    if id != agent_id:
                        relative_displacement = agent.convert_to_local_coordinates(object.position,agent.position)
                        relative_distance = np.sqrt(relative_displacement[0]**2 + relative_displacement[1]**2)
                        if relative_distance <= 50:
                            objects_of_interest.append(object)
                scene_dict = {}
                SAME_COLOR = (0,0,0)
                scene_dict["agent"] = dict(
                    id = agent.id,
                    color = SAME_COLOR,    
                    heading = agent.heading ,      
                    lane = agent.lane_index,                          
                    speed =  agent.speed,
                    pos = agent.position,
                    bbox = agent.bounding_box
                )
                object_descriptions = [
                    dict(
                        id = object.id,
                        color = SAME_COLOR,    
                        heading = object.heading ,      
                        lane = object.lane_index,                          
                        speed =  object.speed,
                        pos = object.position,
                        bbox = object.bounding_box
                    )
                    for object in objects_of_interest
                ]
                scene_dict["nodes"] = object_descriptions
                observation = {}
                observation['lidar'] = env.vehicle.lidar.perceive(env.vehicle)
                camera = env.vehicle.image_sensors["rgb_camera"]
                observation['camera'] = camera.get_pixels_array(env.vehicle, False)
                print("Generated the %d th data point" %(count))
                if count > max:
                    print("Reached Maximum Limit")
                    return count
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_vehicle.expert_takeover = True        
    except Exception as e:
        raise e
    finally:
        env.close()
    return count


def vehicle_type(type_str):
    vehicle_type = {
        "SVehicle": "s",
        "MVehicle": "m",
        "LVehicle": "l",
        "XLVehicle": "xl",
        "DefaultVehicle": "default",
        "StaticDefaultVehicle": "static_default",
        "VaryingDynamicsVehicle": "varying_dynamics"
    }
    for t in vehicle_type.keys():
        if re.search(t,type_str) is not None:
            return vehicle_type[t]
    return "f"





if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        manual_control=True,
        traffic_density=0.2,
        num_scenarios=100,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        #need_inverse_traffic = True,
        # image_on_cuda = True,
        # debug=True,
        # debug_static_world=True,
        map=4,  # seven block
        start_seed=random.randint(0, 1000),
        vehicle_config = {"image_source":"rgb_camera", "rgb_camera":(1920,1080)}
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
    parser.add_argument("--num_instance", type = int, default = 100)
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(dict(image_observation=True))
    env = MetaDriveEnv(config)
    try:
        o, _ = env.reset()
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("The observation is an numpy array with shape: ", o.shape)
        count = 1
        path = '{}'.format(env.current_seed)
        folder = os.path.join(os.getcwd(), path)
        os.mkdir(folder)
        print(folder)
        print("Folder has name %s" %(env.current_seed))
        for i in range(1, 9000000000):
            if count > args.num_instance:
                env.reset()
                break
            o, r, tm, tc, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
            if i % 30== 0: #note: the "1st" object in objects is the agent
                agents = env.engine.agents
                agent = list(agents.values())[0] #if single-agent setting
                agent_id = list(agents.values())[0].id
                objects = env.engine.get_objects()
                objects_of_interest = []
                for id, object in objects.items():
                    if id != agent_id:
                        relative_displacement = agent.convert_to_local_coordinates(object.position,agent.position)
                        relative_distance = np.sqrt(relative_displacement[0]**2 + relative_displacement[1]**2)
                        if relative_distance <= 50:
                            objects_of_interest.append(object)
                if len(objects_of_interest)>0:
                    identifier = "{}_{}".format(env.current_seed,env.episode_step)
                    instance_folder = os.path.join(folder,identifier)
                    try:
                        os.mkdir(instance_folder)
                    except:
                        print("Error in making instance folder")
                    scene_dict = {}
                    SAME_COLOR = (0,0,0)
                    scene_dict["agent"] = dict(
                        id = agent.id,
                        color = SAME_COLOR,    
                        heading = agent.heading ,      
                        lane = agent.lane_index,                          
                        speed =  agent.speed,
                        pos = agent.position,
                        bbox = [tuple(point) for point in agent.bounding_box],
                        type = vehicle_type(str(type(agent))),
                        height = agent.HEIGHT,
                        road_type = agent.navigation.current_road.block_ID()
                    )
                    observation = {}
                    observation['lidar'],observable = env.vehicle.lidar.perceive(env.vehicle)
                    observable_id = [car.id for car in list(observable)]
                    final_objects = [object for object in objects_of_interest if object.id in observable_id]
                    







                    object_descriptions = [
                        dict(
                            id = object.id,
                            color = SAME_COLOR,    
                            heading = object.heading ,      
                            lane = object.lane_index,                          
                            speed =  object.speed,
                            pos = object.position,
                            bbox = [tuple(point) for point in object.bounding_box],
                            type = vehicle_type(str(type(object))),
                            height = object.HEIGHT,
                            road_type = object.navigation.current_road.block_ID()
                        )
                        for object in objects_of_interest
                    ]
                    scene_dict["vehicles"] = object_descriptions
                    rgb_cam = env.vehicle.get_camera(env.vehicle.config["image_source"])
                    for object in final_objects:
                        if rgb_cam.get_cam().node().isInView(object.origin.getPos(rgb_cam.get_cam())):
                                print("RGB_Center_Observable:{}".format(object.id))
                        """origin_x, origin_y, origin_z = agent.origin.getPos()
                        z_augmented = [[x,y, origin_z] for x,y in agent.bounding_box]
                        object_sample_box = sample_bbox(z_augmented,agent.height,4,4,4)
                        observable_count = 0
                        total_count = len(object_sample_box)
                        for sample in object_sample_box:
                            sample_np = NodePath("tmp_node")
                            sample_np.reparentTo(env.engine.render)
                            sample_np.setPos(sample[0],sample[1],sample[2])
                            print("rgb_render",rgb_cam.get_cam().getPos(env.engine.render))
                            print("sample_np_render",sample_np.getPos(env.engine.render))
                            print("sample_np to rgb",sample_np.getPos(rgb_cam.get_cam()))
                            if rgb_cam.get_cam().node().isInView(sample_np.getPos(rgb_cam.get_cam())):
                                observable_count += 1
                            sample_np.detach_node()
                        print(object.id, observable_count)"""

                    rgb_cam.save_image(env.vehicle, name= path + "/" +identifier + "/"+ "rgb_{}.png".format(identifier))
                    ret1 = env.render(mode = 'top_down', film_size=(6000, 6000), target_vehicle_heading_up=False, screen_size=(3000,3000),show_agent_name=True)
                    ret1 = pygame.transform.flip(ret1,flip_x = True, flip_y = False)
                    file_path = os.path.join(instance_folder, "top_down_{}.png".format(identifier))
                    pygame.image.save(ret1, file_path)
                    ret2 = env.render(mode = 'rgb_array',screen_size=(1600, 900), film_size=(6000, 6000), target_vehicle_heading_up=True)
                    file_path = os.path.join(instance_folder, "front_{}.png".format(identifier))
                    cv2.imwrite(file_path,cv2.cvtColor(ret2, cv2. COLOR_BGR2RGB))
                    try:
                        file_path = os.path.join(instance_folder,"world_{}.json".format(identifier))
                        with open(file_path, 'w') as f:
                            # write the dictionary to the file in JSON format 
                            json.dump(scene_dict,f)
                        file_path = os.path.join(instance_folder,"lidar_{}.json".format(identifier))
                        with open(file_path, 'w') as f:
                            # write the dictionary to the file in JSON format
                            json.dump(observation,f)
                    except:
                        print("Error in storing JSON file")
                    print("Generated the %d th data point" %(count))
                    count += 1
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_vehicle.expert_takeover = True
    except Exception as e:
        raise e
    finally:
        env.close()