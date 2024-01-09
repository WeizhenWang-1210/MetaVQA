from metadrive.envs.base_env import BaseEnv
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.envs.test_pede_metadrive_env import TestPedeMetaDriveEnv
from vqa.utils import get_visible_object_ids, genearte_annotation, generate_annotations
import argparse
import random
from vqa.dataset_utils import l2_distance
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.engine.engine_utils import get_engine
import cv2
import pygame
import os
import json
import yaml

def logging(env, lidar, rgb, scene_dict,masks,identifier, instance_folder, log_mapping):
    """
    The function responsible for saving the observations/informations to destination.
    """
    try:
        #Save the world annotation in world{}.json
        file_path = os.path.join(instance_folder, "world_{}.json".format(identifier))
        with open(file_path, 'w') as f:
            json.dump(scene_dict,f)
        #Save the lidar observation in lidar{}.json
        file_path = os.path.join(instance_folder,"lidar_{}.json".format(identifier))
        with open(file_path, 'w') as f:
            observation = {}
            observation['lidar'] = lidar.tolist()
            json.dump(observation,f)
        #Save the rgb observation in rgb{}.json
        file_path = os.path.join(instance_folder, "rgb_{}.png".format(identifier))
        rgb = rgb * 255 #Needed as cv2 expect 0-255 for rgb value, but the original observations are clipped
        cv2.imwrite(file_path,rgb)
        #Save the top-down view in top_down{}.json
        top_down = env.render(mode = 'top_down', film_size=(6000, 6000), screen_size=(3000,3000),show_agent_name=True)
        top_down = pygame.transform.flip(top_down, flip_x = True, flip_y = True) #flipped since(for whatever reason) the text for id are mirrored w.r.t. y-axis
        file_path = os.path.join(instance_folder, "top_down_{}.png".format(identifier))
        pygame.image.save(top_down, file_path)
        #Save the rendering in front_{}.png
        main = env.render(mode = 'rgb_array',screen_size=(1600, 900), film_size=(6000, 6000))
        file_path = os.path.join(instance_folder, "front_{}.png".format(identifier))
        cv2.imwrite(file_path,cv2.cvtColor(main, cv2.COLOR_BGR2RGB))
        #Save the instance segmentation mask in mask.{}.png
        file_path = os.path.join(instance_folder, "mask_{}.png".format(identifier))
        masks = masks * 255
        cv2.imwrite(file_path, masks)
        #Save the mapping front ID to color for visualization purpose. in metainformation_{}.json
        file_path = os.path.join(instance_folder, "metainformation_{}.json".format(identifier))
        with open(file_path, 'w') as f:
            # write the dictionary to the file in JSON format 
            json.dump(log_mapping,f)
        return 0
    except Exception as e:
        raise e

def generate_data(env: BaseEnv, num_points: int, sample_frequency:int, max_iterations:int, 
                  observation_config:dict, IO_config: dict, seed:int, temporal_generation: bool):
    '''
    Initiate a data recording session with specified parameters. Works with any BaseEnv. Specify the data-saving folder in
    IO_config.
    '''
    try:  
        o, _ = env.reset(seed=seed)
        """
        Create the folder in which you will save your data. The structure would be of:
            root/{episode}_{step}/[name]_{episode}_{step}.extensions
            name := front|lidar|mask|metainformation|rgb|top_down|world
            front: The image captured by the main_camera during rendering
            lidar: The lidar observation in episode at specific step
            mask: The instance segmentation mask according to the ego's rgb observation. This is used for illustration
                    verification of the correctness of the generated ground truth
            top_down: The top-down image in episode at specific step
            wolrd: The annotation of this step in the episode. Utilized to create the scene graphs for dataset generation
        """
        folder = os.path.join(IO_config["batch_folder"])
        os.makedirs(folder, exist_ok = True)
        print("This session is stored in folder: {}".format(folder))
        env.vehicle.expert_takeover = True
        counter = step = 1
        W, H = observation_config["resolution"]
        engine = get_engine()
        #Create image sensors according to the resolutions.
        camera = RGBCamera(W,H,engine)
        instance_camera =  InstanceCamera(W,H,engine)
        #The main data recording loop.
        while counter <= num_points and step <= max_iterations:
            o, r, tm, tc, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
            if IO_config["log"] and step % sample_frequency == 0:
                #Retrieve rgb observations and instance segmentation mask
                masks = instance_camera.perceive(env.vehicle)
                rgb = camera.perceive(env.vehicle)
                #Retrieve mapping from a color to the object it represents. This is used simulate z-buffering. (0,0,0)
                #is reserved for special purpose, and no objects will take this color.
                mapping = engine.c_id
                filter = lambda r,g,b,c : not(r==1 and g==1 and b==1) and not(r==0 and g==0 and b==0) and (c > 32)
                visible_ids, log_mapping = get_visible_object_ids(masks,mapping,filter)
                #Record only if there are observable objects.
                if temporal_generation:
                    identifier = "{}_{}".format(env.current_seed, env.episode_step)
                    instance_folder = os.path.join(folder, identifier)
                    try:
                        os.makedirs(instance_folder, exist_ok=True)
                    except Exception as e:
                        raise e
                    valid_objects = engine.get_objects(lambda x: l2_distance(x, env.vehicle)<=50)
                    log_mapping = {id: log_mapping[id] for id in visible_ids}
                    visible_mask = [True if x in visible_ids else False for x in valid_objects.keys()]
                    objects_annotations = generate_annotations(list(valid_objects.values()), env, visible_mask)
                    ego_annotation = genearte_annotation(env.vehicle, env)
                    scene_dict = dict(
                        ego=ego_annotation,
                        objects=objects_annotations
                    )
                    # send all observations/informations to logging function for the actual I/O part.
                    return_code = logging(env, o, rgb, scene_dict, masks, identifier, instance_folder, log_mapping)
                    if return_code == 0:
                        print("Generated the %d th data point" % (counter))
                        counter += 1
                else:
                    if len(visible_ids) > 0:
                        identifier = "{}_{}".format(env.current_seed,env.episode_step)
                        instance_folder = os.path.join(folder, identifier)
                        try:
                            os.makedirs(instance_folder, exist_ok=True)
                        except Exception as e:
                            raise e
                        #Retrieve all observable objeccts within 50 meter w.r.t. the ego
                        visible_objects = engine.get_objects(lambda x: x.id in visible_ids and l2_distance(x, env.vehicle)<=50)
                        #The next line is needed as the previous log_mapping dictionary contains mapping for objects with distance greater than 50m
                        #away from ego.
                        log_mapping = {id:log_mapping[id] for id in visible_objects.keys()}
                        visible_mask = [True for _ in len(visible_objects.keys())]
                        objects_annotations = generate_annotations(list(visible_objects.values()),env, visible_mask)
                        ego_annotation = genearte_annotation(env.vehicle,env)
                        scene_dict = dict(
                            ego = ego_annotation,
                            objects = objects_annotations
                        )
                        #send all observations/informations to logging function for the actual I/O part.
                        return_code = logging(env, o, rgb, scene_dict,masks,identifier,instance_folder, log_mapping)
                        if return_code == 0:
                            print("Generated the %d th data point" %(counter))
                            counter += 1
            step += 1
    except Exception as e:
        raise e
    finally:
        env.close()

def main():
    #Setup the config
    try:
        with open('vqa/configs/scene_generation_config.yaml', 'r') as f:
        #with open("D:\\research\\metavqa-merge\\MetaVQA\\vqa\\configs\\scene_generation_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise e
    #Setup the gymnasium environment
    scene_config = dict(
        use_render=True,
        manual_control=True,
        traffic_density=config["map_setting"]["traffic_density"],
        num_scenarios=config["map_setting"]["num_scenarios"],
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        need_inverse_traffic = config["map_setting"]["inverse_traffic"],
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=True, show_navi_mark=True),
        map=config["map_setting"]["map_size"] if config["map_setting"]["PG"] else config["map_setting"]["map_sequence"],  
        start_seed=config["map_setting"]["start_seed"],
        debug = True
    )
    env = MetaDriveEnv(scene_config)
    #Call the ACTUAL data recording questions
    generate_data(env, config["num_samples"],config["sample_frequency"],config["max_iterations"], 
                  dict(resolution=(1920,1080)),
                  dict(batch_folder = config["storage_path"], log = True),config["map_setting"]["start_seed"],
                  temporal_generation=config["temporal_generation"])

if __name__ == "__main__":
    main()
    