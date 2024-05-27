PATH = "/bigdata/zhouyunsong/zhouys/Datasets/scenarionet/data_nusc_multiview_trainval.json"
NUSCENES_PATH= "/bigdata/datasets/scenarionet/nuscenes/trainval"



from metadrive.envs.scenario_env import ScenarioDiverseEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.scenario import utils as sd_utils
from vqa.dataset_utils import l2_distance
from vqa.episodes_generation import get_visible_object_ids, generate_annotations,genearte_annotation, postprocess_annotation
import json
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
import re
from collections import defaultdict
import os
import cv2


PAIRED_OBSERVATION = json.load(open(PATH,"r"))
from PIL import Image
import numpy as np
import pickle

PERSPECTIVE_MAPPING = {
    'CAM_FRONT':"front",
    'CAM_FRONT_RIGHT':"rightf",
    'CAM_FRONT_LEFT':"leftf",
    'CAM_BACK':"back",
    'CAM_BACK_LEFT':"leftb",
    'CAM_BACK_RIGHT':"rightb"
}


def save_episode_raw(buffer, raw_buffer,  root, IO):
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
            elif key in ["rgb", "mask"]:
                # Save the rgb|instance observation in {rgb|lidar}_{perspective}_{id}.png
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

    def log_real(data, frame_folder):
        for perspective in data.keys():
            file_path = os.path.join(frame_folder, f"real_{perspective}_{identifier}.png")
            data[perspective].save(file_path)

    env_seed, env_start, env_end = IO
    episode_folder = os.path.join(root, "{}_{}_{}".format(env_seed, env_start + 1, env_end))
    try:
        os.makedirs(episode_folder, exist_ok=True)
        for identifier, data in buffer.items():
            # create the frame folder
            instance_folder = os.path.join(episode_folder, identifier)
            os.makedirs(instance_folder, exist_ok=True)
            log_frame(data, instance_folder)

        for identifier, data in raw_buffer.items():
            instance_folder = os.path.join(episode_folder, identifier)
            os.makedirs(instance_folder, exist_ok = True)
            log_real(data, instance_folder)

    except Exception as e:
        raise e
    return 0






def annotate_episode_with_raw(env, engine, sample_frequency, episode_length, camera, instance_camera, lidar, scene_id):
    """
        Record an episode of observations. Note that multiple episodes can be extracted from one seed. Moreover,
        by setting episode_length to 1, you get single-frame observations.
        """
    print("I'm in the episode! Starting at env{}, step{}".format(env.current_seed, env.episode_step))
    total_steps = 0  # record how many steps have actually taken place in the current episode
    buffer = dict()  # Store all frame annotations for the current episode.
    env_id = env.current_seed
    env_start = env.episode_step

    raw_buffer = dict()
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
            hprs = [[0, 0, 0], [55, 0, 0], [110, 0, 0], [180, 0, 0], [-110, 0, 0], [-55, 0, 0]]
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
                objects=objects_annotations,
                world=env.engine.data_manager.current_scenario_file_name if isinstance(env,
                                                                                       ScenarioDiverseEnv) else env.current_seed,
                dataset_summary=env.config["data_directory"] if isinstance(env, ScenarioDiverseEnv) else "PG"
            )
            # send all observations/information to saving function for deferred I/O(when the episode is completed)
            buffer[identifier] = postprocess_annotation(env=env,
                                                        lidar=cloud_points,
                                                        rgb_dict=rgb_annotations,
                                                        scene_dict=scene_annotation,
                                                        log_mapping=Log_Mapping,
                                                        debug=True)
            if total_steps % 5 == 0:
                #its a key frame
                img_path_dict = PAIRED_OBSERVATION[scene_id]['img_path']
                key_frame = total_steps//5
                collection = {}
                for perspective, paths in img_path_dict.items():
                    path = paths[key_frame]
                    rgb = Image.open(path)
                    collection[PERSPECTIVE_MAPPING[perspective]] = rgb
                raw_buffer[identifier] = collection
        total_steps += 1
        if (tm or tc) and info["arrive_dest"]:
            '''
            Don't go into the save environment twice.
            '''
            break
    env_end = env.episode_step
    print(f"exist episode {env.current_seed}")
    return total_steps, buffer, raw_buffer, (env_id, env_start, env_end)  # this end is tail-inclusive



def paired_logging():
    root_folder = "./real"
    scenario_summary, scenario_ids, scenario_files = sd_utils.read_dataset_summary(NUSCENES_PATH)
    num_scenarios = len(scenario_summary)
    print(f"We have {num_scenarios} in total")
    headless = True
    env = ScenarioDiverseEnv(
        {
            "sequential_seed": True,
            "reactive_traffic": True,
            "use_render": not headless,
            "data_directory": NUSCENES_PATH,
            "num_scenarios": num_scenarios,
            "agent_policy": ReplayEgoCarPolicy,
            "sensors": dict(
                rgb=(RGBCamera, 800, 600),
                instance=(InstanceCamera, 800, 600)
            ),
            "height_scale": 1
        }
    )
    env.reset()
    camera = env.engine.get_sensor("rgb")
    instance_camera = env.engine.get_sensor("instance")
    lidar = env.engine.get_sensor("lidar")
    pattern = r'scene-\d+'

    for seed in range(2):
        env.reset(seed)
        print(env.engine.data_manager.current_scenario_file_name)
        id = re.findall(pattern, env.engine.data_manager.current_scenario_file_name)[0]
        step_ran, buffer, raw_buffer, IO = \
            annotate_episode_with_raw(env, env.engine, 1, 25, camera,instance_camera,lidar, id)
        ret_code = save_episode_raw(buffer, raw_buffer, root_folder, IO)
        for _ in range(skip_length):
            o, r, tm, tc, info = env.step([0, 0])
            if tm or tc and info["arrive_dest"]:
                print(f"Finished scenario {seed}.")
                flag = False
                break

        print(ret_code)




if __name__ == "__main__":
    paired_logging()



