from metadrive.envs.base_env import BaseEnv
from vqa.annotation_utils import get_visible_object_ids, genearte_annotation, generate_annotations
import argparse
import numpy as np
from vqa.dataset_utils import l2_distance
from metadrive import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioDiverseEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.engine.engine_utils import get_engine
from vqa.configs.NAMESPACE import OBS_HEIGHT,OBS_WIDTH, MIN_OBSERVABLE_PIXEL, MAX_DETECT_DISTANCE
import pickle
from collections import defaultdict
import cv2
import os
import json
import yaml


def try_load_observations(observations_path):
    import json
    from PIL import Image
    base = os.path.dirname(observations_path)
    obseravations = json.load(open(observations_path, "r"))
    for modality in obseravations.keys():
        if modality == "lidar":
            try_read_pickle(os.path.join(base, obseravations[modality]))
        else:
            # rgb then
            for perspective in obseravations[modality]:
                Image.open(os.path.join(base, obseravations[modality][perspective]))
                print(modality, perspective)


def try_read_pickle(filepath):
    content = pickle.load(open(filepath, 'rb'))
    print(content)


def postprocess_annotation(env, lidar, rgb_dict, scene_dict, log_mapping, debug=False):
    def preprocess_topdown(image):
        # flip topdown  then change channel.
        image = np.fliplr(np.flipud(image))
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
    result["depth"] = dict()
    result["semantic"] = dict()
    result["log_mapping"] = log_mapping
    for perspective in rgb_dict.keys():
        result["rgb"][perspective] = rgb_dict[perspective]["rgb"] * 255
        result["mask"][perspective] = rgb_dict[perspective]["mask"] * 255
        result["depth"][perspective] = rgb_dict[perspective]["depth"] * 255
        result["semantic"][perspective] = rgb_dict[perspective]["semantic"] * 255
    if debug:
        # With name and history.
        top_down = env.render(mode='top_down', film_size=(6000, 6000), screen_size=(1920, 1080), window=False,
                              draw_contour=True, screen_record=False, history_smooth=1, num_stack=10, show_agent_name =True)
        result["top_down"] = preprocess_topdown(top_down)
        # front view image for debugging purpose.
        result["front"] = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, -6, 2],
                                                                [0, -0.5, 0])
    return result


def save_episode(buffer, root, IO):
    def log_frame(data, frame_folder):
        observations = dict()
        for key in data.keys():
            if key in ["top_down", "front"]:
                # Save the top-down observations in {top_down*}_{identifier}
                file_path = os.path.join(frame_folder, f"{key}_{identifier}.png")
                cv2.imwrite(file_path, data[key])
            elif key == "world":
                # Save the world annotation in world_{}.json
                file_path = os.path.join(frame_folder, f"{key}_{identifier}.json")
                json.dump(data["world"], open(file_path, 'w'), indent=2)
            elif key in ["rgb", "mask", "depth", "semantic"]:
                # Save the rgb|instance|depth|semantic observation in {rgb|lidar}_{perspective}_{id}.png
                files = defaultdict(lambda: "")
                for perspective in data[key].keys():
                    file_path = os.path.join(frame_folder, f"{key}_{perspective}_{identifier}.png")
                    cv2.imwrite(file_path, data[key][perspective])
                    files[perspective] = f"{key}_{perspective}_{identifier}.png"
                observations[key] = files
            elif key == "lidar":
                # Save the lidar observation in lidar{id}.json
                file_path = os.path.join(frame_folder, f"lidar_{identifier}.pkl")
                pickle.dump(data["lidar"], open(file_path, 'wb'))
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
    env_id = ".".join(env.engine.data_manager.current_scenario_file_name.split(".")[:-1])  #env.current_seed
    print(env_id)
    env_start = env.episode_step
    depth_cam, semantic_cam = engine.sensors["depth"], engine.sensors["semantic"]
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
            positions = [(0., 0.0, 1.5)] #[(0., 0.0, 1.5), (0., 0., 1.5), (0., 0., 1.5), (0., 0, 1.5), (0., 0., 1.5),(0., 0., 1.5)]
            hprs = [[0, 0, 0]]           #[[0, 0, 0], [45, 0, 0], [135, 0, 0], [180, 0, 0], [225, 0, 0], [315, 0, 0]]
            perspectives = ["front"]     #["front", "leftf", "leftb", "back", "rightb", "rightf"]
            rgb_annotations = {}
            for position, hpr, perspective in zip(positions, hprs, perspectives):
                mask = instance_camera.perceive(to_float=True, new_parent_node=env.agent.origin, position=position,
                                                hpr=hpr)
                rgb = camera.perceive(to_float=True, new_parent_node=env.agent.origin, position=position, hpr=hpr)
                #need to invoke twice to flush the buffer.
                depth = depth_cam.perceive(to_float=True, new_parent_node=env.agent.origin, position=position, hpr=hpr)
                depth = depth_cam.perceive(to_float=True, new_parent_node=env.agent.origin, position=position, hpr=hpr)
                semantic = semantic_cam.perceive(to_float=True, new_parent_node=env.agent.origin, position=position, hpr=hpr)
                rgb_annotations[perspective] = dict(
                    mask=mask,
                    rgb=rgb,
                    depth=depth,
                    semantic=semantic
                )
            # Retrieve mapping from a color to the object it represents. This is used simulate z-buffering. (0,0,0)
            # is reserved for special purpose, and no objects will take this color.
            mapping = engine.c_id
            visible_ids_set = set()
            # to be considered as observable, the object must not be black/white(reserved) and must have at least 960
            # in any of the 1920*1080 resolution camera
            filter = lambda r, g, b, c: not (r == 1 and g == 1 and b == 1) and not (
                    r == 0 and g == 0 and b == 0) and (
                                                c > MIN_OBSERVABLE_PIXEL)
            Log_Mapping = dict()
            for perspective in rgb_annotations.keys():
                visible_ids, log_mapping = get_visible_object_ids(rgb_annotations[perspective]["mask"], mapping, filter)
                visible_ids_set.update(visible_ids)
                rgb_annotations[perspective]["visible_ids"] = visible_ids
                Log_Mapping.update(log_mapping)

            # Record only if there are observable objects.
            # get all objectes within 100m of the ego(except agent)
            valid_objects = engine.get_objects(
                lambda x: l2_distance(x, env.agent) <= MAX_DETECT_DISTANCE and x.id != env.agent.id and not isinstance(x, BaseTrafficLight))
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
                objects=objects_annotations,
                world=env.engine.data_manager.current_scenario_file_name if isinstance(env, ScenarioDiverseEnv) else env.current_seed,
                dataset_summary=env.config["data_directory"] if isinstance(env, ScenarioDiverseEnv) else "PG"
            )
            # send all observations/information to saving function for deferred I/O(when the episode is completed)
            buffer[identifier] = postprocess_annotation(env=env,
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
            break
    env_end = env.episode_step
    print(f"exist episode {env.current_seed}")
    return total_steps, buffer, (env_id, env_start, env_end)  # this end is tail-inclusive


def generate_episodes(env: BaseEnv, num_points: int, sample_frequency: int, max_iterations: int,
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
    print(job_range)
    try:
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
        if job_range is not None:
            terminate = False
            seed = job_range[0]
            while seed < job_range[1] and not terminate:
                env.reset(seed)
                print(f"Reset to seed {seed}")
                flag = True
                while flag: #while we have not exhaust the current scenario.
                    step_ran, records, IO = annotate_episode(
                        env=env, engine=engine, sample_frequency=sample_frequency,
                        episode_length=episode_length, camera=camera, instance_camera=instance_camera,
                        lidar=lidar, job_range=job_range)
                    ret_code = save_episode(records, folder, IO)
                    records.clear()
                    counter += len(records)
                    if ret_code == 0:
                        print("Successfully created episode {}:{}".format(episode_counter, IO))
                        episode_counter += 1
                    if counter > num_points:
                        terminate = True
                        break
                    #skipping some steps.
                    for _ in range(skip_length):
                        o, r, tm, tc, info = env.step([0, 0])
                        if tm or tc and info["arrive_dest"]:
                            print(f"Finished scenario {seed}.")
                            flag = False
                            break
                seed += 1
        else:
            '''Need rework'''
            while counter <= num_points and step <= max_iterations:
                if job_range is not None:
                    # We've exhausted all the environments for process.
                    if env.current_seed not in range(job_range[0], job_range[-1]):
                        break
                # Skip steps logic: Skip the specified number of steps at the beginning of each episode
                if step % (skip_length + episode_length) < skip_length:
                    o, r, tm, tc, info = env.step([0, 0])
                    if tm or tc and info["arrive_dest"]:
                        env.reset(env.current_seed+1)
                        continue

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
                ret_code = save_episode(records, folder, IO)
                records.clear()  # recycling memory
                if ret_code == 0:
                    print("Successfully created episode {}:{}".format(episode_counter, IO))
                    episode_counter += 1
    except Exception as e:
        raise e
    finally:
        env.close()


def annotate_scenarios():
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
                rgb=(RGBCamera, OBS_WIDTH, OBS_HEIGHT),
                instance=(InstanceCamera, OBS_WIDTH, OBS_HEIGHT),
                semantic=(SemanticCamera, OBS_WIDTH, OBS_HEIGHT),
                depth=(DepthCamera, OBS_WIDTH, OBS_HEIGHT)
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
                rgb=(RGBCamera, OBS_WIDTH, OBS_HEIGHT),
                instance=(InstanceCamera, OBS_WIDTH, OBS_HEIGHT),
                semantic=(SemanticCamera, OBS_WIDTH, OBS_HEIGHT),
                depth=(DepthCamera, OBS_WIDTH, OBS_HEIGHT)
            ),
            height_scale=1
        )
        env = MetaDriveEnv(env_config)
        env.reset()
        env.agent.expert_takeover = True

    generate_episodes(env=env, num_points=config["num_samples"], sample_frequency=config["sample_frequency"],
                      max_iterations=config["max_iterations"],
                      IO_config=dict(batch_folder=config["storage_path"], log=True),
                      episode_length=config["episode_length"], skip_length=config["skip_length"], job_range=[2,5])


if __name__ == "__main__":
    annotate_scenarios()
    #try_load_observations("multiview_final/80_30_30/80_30/observations_80_30.json")
