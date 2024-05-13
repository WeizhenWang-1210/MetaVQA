from metadrive.envs.base_env import BaseEnv
from vqa.utils import get_visible_object_ids, genearte_annotation, generate_annotations
import argparse
import numpy as np
from vqa.dataset_utils import l2_distance
from metadrive import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioDiverseEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.engine.engine_utils import get_engine
import pickle
from collections import defaultdict
import cv2
import os
import json
import yaml


def multiview_saving(env, lidar, rgb_dict, scene_dict, log_mapping, debug=False):
    def preprocess_topdown(image):
        # flip topdown  then change channel.
        image = np.fliplr(np.flipud(top_down_debug))
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        rgb_image = np.dstack((r, g, b))
        return rgb_image

    """This function will save all information regarding one frame into a single dict, and it can be saved into the 
    buffer to log later."""
    result = dict()
    result["world"] = scene_dict
    result["lidar"] = lidar
    result["rgb"] = dict()
    result["mask"] = dict()
    result["log_mapping"] = log_mapping
    for perspective in rgb_dict:
        result["rgb"][perspective] = rgb_dict[perspective]["rgb"] * 255
        result["mask"][perspective] = rgb_dict[perspective]["mask"] * 255
    if debug:
        # With name and history.
        top_down_debug = env.render(mode='top_down', film_size=(6000, 6000), screen_size=(1920, 1080), window=False,
                                    draw_contour=True, screen_record=False, show_agent_name=True)
        result["top_down_debug"] = preprocess_topdown(top_down_debug)
        # ego head up.
        top_down_ego = env.render(mode='top_down', film_size=(6000, 6000), screen_size=(1920, 1080), window=False,
                                  draw_contour=True, screen_record=False, show_agent_name=False,
                                  target_agent_heading_up=True)
        result["top_down_ego"] = preprocess_topdown(top_down_ego)
        # no history
        top_down_static = env.render(mode='top_down', film_size=(6000, 6000), screen_size=(1920, 1080), window=False,
                                     draw_contour=True, screen_record=False, show_agent_name=False, num_stack=1)
        result["top_down_static"] = preprocess_topdown(top_down_static)
        # without name, no ego head up, with history.
        top_down = env.render(mode='top_down', film_size=(6000, 6000), screen_size=(1920, 1080), window=False,
                              draw_contour=True, screen_record=False, show_agent_name=False)
        result["top_down"] = preprocess_topdown(top_down)
        #front view image for debugging purpose.
        result["front"] = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, -6, 2],
                                                                [0, -0.5, 0])
    return result


def episode_logging(buffer, root, IO):
    def log_frame(data, frame_folder):
        observations=dict()
        for key in data.keys():
            if key in ["top_down", "top_down_debug", "top_down_static", "top_down_ego", "front"]:
                # Save the top-down observations in {top_down*}_{identifier}
                file_path = os.path.join(frame_folder, f"{key}_{identifier}.png")
                cv2.imwrite(file_path, data[key])
            elif key == "world":
                # Save the world annotation in world_{}.json
                file_path = os.path.join(frame_folder, f"{key}_{identifier}.json")
                json.dump(data["world"], open(file_path, 'w'), indent=2)
            elif key in ["rgb", "mask"]:
                # Save the rgb|instance observation in {rgb|lidar}_{perspective}_{id}.png
                files = defaultdict(lambda:"")
                for perspective in data[key].keys():
                    file_path = os.path.join(frame_folder, f"{key}_{perspective}_{identifier}.png")
                    cv2.imwrite(file_path, data[key][perspective])
                    files[perspective] = f"{key}_{perspective}_{identifier}.png"
                observations[key] = files
            elif key == "lidar":
                # Save the lidar observation in lidar{id}.json
                file_path = os.path.join(frame_folder, f"lidar_{identifier}.pkl")
                pickle.dump(data["lidar"], open(file_path, 'w'))
                observations[key] = f"lidar_{identifier}.pkl"
            elif key == "log_mapping":
                # Save the mapping front ID to color for visualization purpose. in metainformation_{}.json
                file_path = os.path.join(frame_folder, "id2c_{}.json".format(identifier))
                json.dump(data["log_mapping"], open(file_path, 'w'), indent=2)
            else:
                raise Exception(f"{key} is not in the annotation!")
        file_path = os.path.join(frame_folder, f"observations_{identifier}.json")
        json.dump(observations, open(file_path, "w"), indent=2)
        return 0
    env_seed, env_start, env_end = IO
    episode_folder = os.path.join(root, "{}_{}_{}".format(env_seed, env_start + 1, env_end))
    # env_start+1 since it is one step before the first frame.
    # create the episode folder, which will contain each frame folder
    try:
        os.makedirs(episode_folder, exist_ok=True)
        for identifier, data in buffer.items():
            # create the frame folder
            instance_folder = os.path.join(episode_folder, identifier)
            os.makedirs(instance_folder, exist_ok=True)
            log_frame(data, instance_folder)
    except Exception as e:
        raise e
    return 0


def annotate_episode(env, engine, sample_frequency, episode_length, camera, instance_camera, lidar, job_range=None):
    """
    Record an episode of observations. Note that multiple episodes can be extracted from one seed. Moreover,
    by setting episode_length to 1, you get single-frame observations.
    """
    print("I'm in the episode! Starting at env{}, step{}".format(env.current_seed, env.episode_step))
    total_steps = 0  # record how many steps have actually taken place in the current episode
    buffer = dict()  # Store all frame annotations for the current episode.
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
            identifier = "{}_{}".format(env.current_seed, env.episode_step)
            positions = [(0., 0.0, 1.5), (0., 0., 1.5), (0., 0., 1.5), (0., 0, 1.5), (0., 0., 1.5),
                         (0., 0., 1.5)]
            hprs = [[0, 0, 0], [45, 0, 0], [135, 0, 0], [180, 0, 0], [225, 0, 0], [315, 0, 0]]
            names = ["front", "leftf", "leftb", "back", "rightb", "rightf"]
            rgb_annotations = {}
            for position, hpr, name in zip(positions, hprs, names):
                mask = instance_camera.perceive(to_float=True, new_parent_node=env.agent.origin, position=position,
                                                hpr=hpr)
                rgb = camera.perceive(to_float=True, new_parent_node=env.agent.origin, position=position, hpr=hpr)
                rgb_annotations[name] = dict(
                    mask=mask,
                    rgb=rgb
                )
            # Retrieve mapping from a color to the object it represents. This is used simulate z-buffering. (0,0,0)
            # is reserved for special purpose, and no objects will take this color.
            mapping = engine.c_id
            visible_ids_set = set()
            # to be considered as observable, the object must not be black/white(reserved) and must have at least 240
            # in any of the 960*540 resolution camera
            filter = lambda r, g, b, c: not (r == 1 and g == 1 and b == 1) and not (
                    r == 0 and g == 0 and b == 0) and (
                                                c > 240)
            Log_Mapping = dict()
            for perspective in rgb_annotations.keys():
                visible_ids, log_mapping = get_visible_object_ids(rgb_annotations[perspective]["mask"], mapping, filter)
                visible_ids_set.update(visible_ids)
                rgb_annotations[perspective]["visible_ids"] = visible_ids
                Log_Mapping.update(log_mapping)

            # Record only if there are observable objects.
            # get all objectes within 50m of the ego(except agent)
            valid_objects = engine.get_objects(
                lambda x: l2_distance(x,
                                      env.agent) <= 50 and x.id != env.agent.id and not isinstance(x,
                                                                                                   BaseTrafficLight))
            observing_camera = []
            for obj_id in valid_objects.keys():
                final = []
                for perspective in rgb_annotations.keys():
                    if obj_id in rgb_annotations[perspective]["visible_ids"]:
                        final.append(perspective)
                observing_camera.append(final)
            # We will determine visibility for all valid_objects set.
            visible_mask = [True if x in visible_ids_set else False for x in valid_objects.keys()]
            # we will annotate all objects within 50 meters with respective to ego.
            # TODO unify annotation nomenclature with Chenda.
            objects_annotations = generate_annotations(list(valid_objects.values()), env, visible_mask,
                                                       observing_camera)
            ego_annotation = genearte_annotation(env.agent, env)
            scene_annotation = dict(
                ego=ego_annotation,
                objects=objects_annotations
            )
            # send all observations/information to saving function for deferred I/O(when the episode is completed)
            buffer[identifier] = multiview_saving(env=env,
                                                  lidar=cloud_points,
                                                  rgb_dict=rgb_annotations,
                                                  scene_dict=scene_annotation,
                                                  log_mapping=Log_Mapping,
                                                  debug=True)
        total_steps += 1
        if (tm or tc) and info["arrive_dest"]:
            '''
            Don't go into the save environment twice.
            '''
            if job_range is not None and env.current_seed + 1 >= job_range[-1]:
                break
            env.reset(env.current_seed + 1)
            env.current_track_agent.expert_takeover = True
            break
    env_end = env.episode_step
    print("exist episode")
    return total_steps, buffer, (env_id, env_start, env_end)  # this end is tail-inclusive


def generate_data(env: BaseEnv, num_points: int, sample_frequency: int, max_iterations: int,
                  IO_config: dict, episode_length: int,
                  skip_length: int, job_range=None):
    """
    Initiate a data recording session with specified parameters. Works with any BaseEnv. Specify the data-saving folder in
    IO_config.
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

    try:
        # o, _ = env.reset(seed)

        folder = os.path.join(IO_config["batch_folder"])
        os.makedirs(folder, exist_ok=True)
        print("This session is stored in folder: {}".format(folder))
        env.agent.expert_takeover = True
        counter = step = 1
        episode_counter = 0
        engine = get_engine()
        # Fetch image sensors
        camera = env.engine.get_sensor("rgb")
        instance_camera = env.engine.get_sensor("instance")
        lidar = env.engine.get_sensor("lidar")
        # The main data recording loop.
        while counter <= num_points and step <= max_iterations:
            if job_range is not None:
                # We've exhausted all the environments for process.
                if env.current_seed not in range(job_range[0], job_range[-1]):
                    break
            # Skip steps logic: Skip the specified number of steps at the beginning of each episode
            if step % (skip_length + episode_length) < skip_length:
                env.step([0, 0])
                step += 1
                continue
            step_ran, records, IO = annotate_episode(env=env,
                                                     engine=engine,
                                                     sample_frequency=sample_frequency,
                                                     episode_length=episode_length,
                                                     camera=camera,
                                                     instance_camera=instance_camera,
                                                     lidar=lidar,
                                                     job_range=job_range)
            step += step_ran
            counter += len(records)
            ret_code = episode_logging(records, folder, IO)
            records.clear()  # recycling memory
            if ret_code == 0:
                print("Successfully created episode {}:{}".format(episode_counter, IO))
                episode_counter += 1
    except Exception as e:
        raise e
    finally:
        env.close()


def main():
    # TODO use deluxe rendering.

    # Set up the config
    cwd = os.getcwd()
    full_path = os.path.join(cwd, "vqa", "configs", "scene_generation_config.yaml")
    try:
        # If your path is not correct, run this file with root folder based at metavqa instead of vqa.
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise e

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", default=False, help="Rendering in headless mode", action="store_true")
    parser.add_argument("--scenarios", default=False, help="Use ScenarioNet environment", action="store_true")
    parser.add_argument("--data_directory", type=str, default=None, help="the paths that stores the ScenarioNet data")
    args = parser.parse_args()
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))
    if args.headless:
        use_render = False
    else:
        use_render = True
    # Setup the gymnasium environment
    if args.scenarios:
        from metadrive.engine.asset_loader import AssetLoader
        asset_path = AssetLoader.asset_path
        use_waymo = True
        from metadrive.policy.replay_policy import ReplayEgoCarPolicy
        # Load the dicrectory.
        if args.data_directory:
            from metadrive.scenario import utils as sd_utils
            scenario_summary, scenario_ids, scenario_files = sd_utils.read_dataset_summary(args.data_directory)
            num_scenarios = len(scenario_summary.keys())
        else:
            num_scenarios = 3 if use_waymo else 10
        env_config = {
            "sequential_seed": True,
            "reactive_traffic": True,
            "use_render": use_render,
            "data_directory": args.data_directory if args.data_directory is not None else AssetLoader.file_path(
                asset_path, "waymo" if use_waymo else "nuscenes", unix_style=False
            ),
            "num_scenarios": num_scenarios,
            "agent_policy": ReplayEgoCarPolicy,
            "sensors": dict(
                rgb=(RGBCamera, 960, 540),
                instance=(InstanceCamera, 960, 540),
            ),
            "height_scale": 1
        }
        print("Finished reading")
        env = ScenarioDiverseEnv(env_config)
        env.reset(seed=0)
    else:
        env_config = dict(
            use_render=use_render,
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
            map=config["map_setting"]["map_size"] if config["map_setting"]["PG"] else config["map_setting"][
                "map_sequence"],
            start_seed=config["map_setting"]["start_seed"],
            debug=False,
            sensors=dict(
                rgb=(RGBCamera, 960, 540,),
                instance=(InstanceCamera, 960, 540)
            ),
            height_scale=1
        )
        env = MetaDriveEnv(env_config)
        env.reset()
        env.agent.expert_takeover = True

    generate_data(env=env, num_points=config["num_samples"], sample_frequency=config["sample_frequency"],
                  max_iterations=config["max_iterations"],
                  IO_config=dict(batch_folder=config["storage_path"], log=True),
                  episode_length=config["episode_length"], skip_length=config["skip_length"])


if __name__ == "__main__":
    main()
