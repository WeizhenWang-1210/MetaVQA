from metadrive.envs.base_env import BaseEnv
# rom metadrive.envs.real_data_envs.waymo_env import WaymoEnv
# from metadrive.envs.test_pede_metadrive_env import TestPedeMetaDriveEnv
from vqa.utils import get_visible_object_ids, genearte_annotation, generate_annotations
import argparse
import random
import numpy as np
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
from vqa.vqa_observation import MyObservation


def logging(env, lidar, rgb, scene_dict, masks, identifier, instance_folder, log_mapping):
    """
    The function responsible for saving the observations/informations to destination.
    """
    try:
        # Save the world annotation in world{}.json
        file_path = os.path.join(instance_folder, "world_{}.json".format(identifier))
        with open(file_path, 'w') as f:
            json.dump(scene_dict, f, indent=4)
        # Save the lidar observation in lidar{}.json
        file_path = os.path.join(instance_folder, "lidar_{}.json".format(identifier))
        with open(file_path, 'w') as f:
            observation = {}
            observation['lidar'] = lidar.tolist()
            json.dump(observation, f, indent=4)
        # Save the rgb observation in rgb{}.json
        file_path = os.path.join(instance_folder, "rgb_{}.png".format(identifier))
        rgb = rgb * 255  # Needed as cv2 expect 0-255 for rgb value, but the original observations are clipped
        cv2.imwrite(file_path, rgb)
        # Save the top-down view in top_down{}.json
        top_down = env.render(mode='top_down', film_size=(6000, 6000), screen_size=(3000, 3000), show_agent_name=True,
                              window=False)
        top_down = np.fliplr(
            np.flipud(top_down))  # flipped since(for whatever reason) the text for id are mirrored w.r.t. y-axis
        file_path = os.path.join(instance_folder, "top_down_{}.png".format(identifier))
        cv2.imwrite(file_path, top_down)
        # Save the rendering in front_{}.png
        main = env.main_camera_perceive(env.agent)
        file_path = os.path.join(instance_folder, "front_{}.png".format(identifier))
        cv2.imwrite(file_path, cv2.cvtColor(main, cv2.COLOR_BGR2RGB))
        # Save the instance segmentation mask in mask.{}.png
        file_path = os.path.join(instance_folder, "mask_{}.png".format(identifier))
        masks = masks * 255
        cv2.imwrite(file_path, masks)
        # Save the mapping front ID to color for visualization purpose. in metainformation_{}.json
        file_path = os.path.join(instance_folder, "metainformation_{}.json".format(identifier))
        with open(file_path, 'w') as f:
            # write the dictionary to the file in JSON format
            json.dump(log_mapping, f, indent=4)
        return 0
    except Exception as e:
        raise e


def saving(env, lidar, rgb, scene_dict, masks, log_mapping, debug=False):
    """
    This function will save all information regarding one frame into a single dict, and it can be saved into the buffer to log later.
    """
    result = dict()
    result["world"] = scene_dict
    result["lidar"] = lidar
    result["rgb"] = rgb * 255
    if debug:
        top_down = env.render(mode='top_down', film_size=(6000, 6000), screen_size=(1920, 1080), window=False,
                              draw_contour=True, screen_record=False, show_agent_name=True)
        result["top_down"] = np.fliplr(np.flipud(top_down))
        result["front"] = env.main_camera.perceive(False)  # cv2.cvtColor(main, cv2.COLOR_BGR2RGB)
    result["mask"] = masks * 255
    result["log_mapping"] = log_mapping
    return result


def episode_logging(buffer, root, IO):
    env_seed, env_start, env_end = IO
    episode_folder = os.path.join(root, "{}_{}_{}".format(env_seed, env_start + 1, env_end))
    # create the episode folder, which will contain each frame folder
    try:
        os.makedirs(episode_folder, exist_ok=True)
    except Exception as e:
        raise e

    def log_data(data, instance_folder):
        try:
            # Save the world annotation in world{}.json
            file_path = os.path.join(instance_folder, "world_{}.json".format(identifier))
            with open(file_path, 'w') as f:
                json.dump(data["world"], f, indent=4)
            # Save the lidar observation in lidar{}.json
            file_path = os.path.join(instance_folder, "lidar_{}.json".format(identifier))
            with open(file_path, 'w') as f:
                observation = {}
                observation['lidar'] = data["lidar"]
                json.dump(observation, f, indent=4)
            # Save the rgb observation in rgb{}.json
            file_path = os.path.join(instance_folder, "rgb_{}.png".format(identifier))
            cv2.imwrite(file_path, data["rgb"])
            if "top_down" in data.keys():
                # Save the top-down view in top_down{}.json
                file_path = os.path.join(instance_folder, "top_down_{}.png".format(identifier))
                cv2.imwrite(file_path, data["top_down"])
            if "front" in data.keys():
                # Save the rendering in front_{}.png
                file_path = os.path.join(instance_folder, "front_{}.png".format(identifier))
                cv2.imwrite(file_path, data["front"])
            # Save the instance segmentation mask in mask.{}.png
            file_path = os.path.join(instance_folder, "mask_{}.png".format(identifier))
            cv2.imwrite(file_path, data["mask"])
            # Save the mapping front ID to color for visualization purpose. in metainformation_{}.json
            file_path = os.path.join(instance_folder, "metainformation_{}.json".format(identifier))
            with open(file_path, 'w') as f:
                # write the dictionary to the file in JSON format
                json.dump(data["log_mapping"], f, indent=4)
            return 0
        except Exception as e:
            raise e

    for identifier, data in buffer.items():
        # create the frame folder
        instance_folder = os.path.join(episode_folder, identifier)
        try:
            os.makedirs(instance_folder, exist_ok=True)
        except Exception as e:
            raise e
        log_data(data, instance_folder)
    return 0


def run_episode(env, engine, sample_frequency, episode_length, camera, instance_camera, lidar):
    print("I'm in the episode!")
    total_steps = 0
    buffer = dict()
    env_id = env.current_seed
    env_start = env.episode_step


    while total_steps < episode_length:
        o, r, tm, tc, info = env.step([0, 0])
        env.render(
            text={
                "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
            }
        )
        if total_steps % sample_frequency == 0:

            cloud_points, _ = lidar.perceive(
                env.agent,
                physics_world=env.engine.physics_world.dynamic_world,
                num_lasers=env.agent.config["lidar"]["num_lasers"],
                distance=env.agent.config["lidar"]["distance"],
                show=False,
            )
            masks = instance_camera.perceive(clip=True, new_parent_node=env.agent.origin, position=(0., 0.8, 1.5),
                                             hpr=[0, 0, 0])
            rgb = camera.perceive(clip=True, new_parent_node=env.agent.origin, position=(0., 0.8, 1.5), hpr=[0, 0, 0])
            # Retrieve mapping from a color to the object it represents. This is used simulate z-buffering. (0,0,0)
            # is reserved for special purpose, and no objects will take this color.
            mapping = engine.c_id
            # to be consider observable, the object must not be black/white(reserved) and must have at least 32 pixels observable
            filter = lambda r, g, b, c: not (r == 1 and g == 1 and b == 1) and not (r == 0 and g == 0 and b == 0) and (
                        c > 32)
            visible_ids, log_mapping = get_visible_object_ids(masks, mapping, filter)
            # Record only if there are observable objects.
            identifier = "{}_{}".format(env.current_seed, env.episode_step)
            valid_objects = engine.get_objects(
                lambda x: l2_distance(x, env.vehicle) <= 50)  # get all objectes within 50m of the ego
            log_mapping = {id: log_mapping[id] for id in visible_ids}
            visible_mask = [True if x in visible_ids else False for x in valid_objects.keys()]
            objects_annotations = generate_annotations(list(valid_objects.values()), env, visible_mask)
            ego_annotation = genearte_annotation(env.vehicle, env)
            scene_dict = dict(
                ego=ego_annotation,
                objects=objects_annotations
            )
            # send all observations/informations to logging function for the actual I/O part.
            buffer[identifier] = saving(env=env,
                                        lidar=cloud_points,
                                        rgb=rgb,
                                        scene_dict=scene_dict,
                                        masks=masks,
                                        log_mapping=log_mapping,
                                        debug=True)
        total_steps += 1
        if (tm or tc) and info["arrive_dest"]:
            env.reset(env.current_seed + 1)
            env.current_track_vehicle.expert_takeover = True
            break
    env_end = env.episode_step
    print("exist episode")
    return total_steps, buffer, (env_id, env_start, env_end)


def generate_data(env: BaseEnv, num_points: int, sample_frequency: int, max_iterations: int,
                  IO_config: dict, seed: int, temporal_generation: bool, episode_length: int,
                  skip_length: int):
    '''
    Initiate a data recording session with specified parameters. Works with any BaseEnv. Specify the data-saving folder in
    IO_config.
    '''
    try:
        o, _ = env.reset(seed=seed)
        """
        Create the folder in which you will save your data. The structure would be of:
            root/{episode}_{begin_step}_{end_step}/{step}/[name]_{episode}_{step}.extensions
            name := front|lidar|mask|metainformation|rgb|top_down|world
            front: The image captured by the main_camera during rendering
            lidar: The lidar observation in episode at specific step
            mask: The instance segmentation mask according to the ego's rgb observation. This is used for illustration
                    verification of the correctness of the generated ground truth
            top_down: The top-down image in episode at specific step
            wolrd: The annotation of this step in the episode. Utilized to create the scene graphs for dataset generation
        """
        folder = os.path.join(IO_config["batch_folder"])
        os.makedirs(folder, exist_ok=True)
        print("This session is stored in folder: {}".format(folder))
        env.vehicle.expert_takeover = True
        counter = step = 1
        episode_counter = 0
        engine = get_engine()
        # Create image sensors according to the resolutions.
        camera = env.engine.get_sensor("rgb")
        instance_camera= env.engine.get_sensor("instance")
        lidar = env.engine.get_sensor("lidar")
        # The main data recording loop.
        while counter <= num_points and step <= max_iterations:
            # Skip steps logic: Skip the specified number of steps at the beginning of each episode
            if step % (skip_length + episode_length) < skip_length:
                env.step([0, 0])
                step += 1
                continue
            step_ran, records, IO = run_episode(env=env,
                                                engine=engine,
                                                sample_frequency=sample_frequency,
                                                episode_length=episode_length,
                                                camera=camera,
                                                instance_camera=instance_camera,
                                                lidar=lidar)
            step += step_ran
            counter += len(records)
            ret_code = episode_logging(records, folder, IO)
            records.clear()  # recycling memory
            if ret_code == 0:
                print("Successfully created episode {}".format(episode_counter))
                episode_counter += 1
    except Exception as e:
        raise e
    finally:
        env.close()


def main():
    # Setup the config
    cwd = os.getcwd()
    full_path = os.path.join(cwd, "vqa", "configs", "scene_generation_config.yaml")
    try:
        with open(full_path, 'r') as f:
            # with open("D:\\research\\metavqa-merge\\MetaVQA\\vqa\\configs\\scene_generation_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise e
    # Setup the gymnasium environment
    scene_config = dict(
        use_render=False,  # True,
        manual_control=True,
        traffic_density=config["map_setting"]["traffic_density"],
        num_scenarios=config["map_setting"]["num_scenarios"],
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        need_inverse_traffic=config["map_setting"]["inverse_traffic"],
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=False, show_navi_mark=False),
        map=config["map_setting"]["map_size"] if config["map_setting"]["PG"] else config["map_setting"]["map_sequence"],
        start_seed=config["map_setting"]["start_seed"],
        debug=False,
        sensors=dict(
            rgb=(RGBCamera, 960,640),
            instance=(InstanceCamera, 960,640)
        ),
        # image_observation = False,
        # agent_observation = MyObservation
    )

    env = MetaDriveEnv(scene_config)

    env.reset()
    # env.engine.force_fps.disable()
    env.agent.expert_takeover = True
    # Call the ACTUAL data recording questions
    for i in range(100):
        o, r, d, _, _ = env.step([0, 0])

        #print(cloud_points)
        #ret1 = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, 0.8, 1.5], [0, 0, 0])
        """ret1 = o["instance"][...,-1]
        """
        #cv2.imshow("image", ret1)
        #ret2 = env.engine.get_sensor("instance").perceive(False, env.agent.origin, [0, 0.8, 1.5], [0, 0, 0])
        # ret3 = env.engine.get_sensor("main_camera").perceive(False, env.agent.origin, [0,0.8,1.5], [0,0,0])
        #cv2.imshow("image_2", ret2)
        # ret3

        #cv2.waitKey(1)
    #env.close()
    generate_data(env, config["num_samples"],config["sample_frequency"],config["max_iterations"],
                  dict(batch_folder = config["storage_path"], log = True),config["map_setting"]["start_seed"],
                  temporal_generation=config["temporal_generation"],episode_length=config["episode_length"],skip_length=config["skip_length"])


if __name__ == "__main__":
    main()
