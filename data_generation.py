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
from metadrive.constants import HELP_MESSAGE
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



if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        manual_control=True,
        traffic_density=0.2,
        num_scenarios=100,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        # image_on_cuda = True,
        # debug=True,
        # debug_static_world=True,
        map=4,  # seven block
        start_seed=random.randint(0, 1000),
        vehicle_config = {"image_source":"rgb_camera", "rgb_camera":(960, 540)}
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
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
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
            if i % 40== 0: #note: the "1st" object in objects is the agent
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
                        type = str(type(agent)).split()[-1]
                    )
                    object_descriptions = [
                        dict(
                            id = object.id,
                            color = SAME_COLOR,    
                            heading = object.heading ,      
                            lane = object.lane_index,                          
                            speed =  object.speed,
                            pos = object.position,
                            bbox = [tuple(point) for point in object.bounding_box],
                            type = str(type(object)).split()[-1]
                        )
                        for object in objects_of_interest
                    ]
                    
                    scene_dict["vehicles"] = object_descriptions
                    observation = {}
                    observation['lidar'] = env.vehicle.lidar.perceive(env.vehicle)[0]
                    rgb_cam = env.vehicle.get_camera(env.vehicle.config["image_source"])
                    rgb_cam.save_image(env.vehicle, name="{}.png".format(i))
                    ret1 = env.render(mode = 'top_down', film_size=(6000, 6000), target_vehicle_heading_up=False, screen_size=(2000,2000),show_agent_name=True)
                    ret1 = pygame.transform.flip(ret1,flip_x = True, flip_y = False)
                    pygame.image.save(ret1, "top_down_{}_{}.png".format(env.current_seed, env.episode_step))
                    ret2 = env.render(mode = 'rgb_array',screen_size=(1600, 900), film_size=(6000, 6000), target_vehicle_heading_up=True)
                    cv2.imwrite("front_{}.png".format(env.episode_step),cv2.cvtColor(ret2, cv2. COLOR_BGR2RGB))
                    try:
                        with open("{}.json".format(i), 'w') as f:
                            # write the dictionary to the file in JSON format 
                            json.dump(scene_dict,f)
                        #with open("{}_lidar.json".format(i), 'w') as f:
                            # write the dictionary to the file in JSON format
                        #    json.dump(observation,f)
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