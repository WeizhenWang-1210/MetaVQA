from metadrive.envs.scenario_env import ScenarioDiverseEnv
from metadrive.scenario import utils as sd_utils
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from collections import deque
import cv2
import os
import json

from vqa.utils import get_visible_object_ids, genearte_annotation, generate_annotations
from vqa.dataset_utils import l2_distance
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.component.sensors.instance_camera import InstanceCamera
from vqa.episodes_generation import postprocess_annotation


def find_collision_step(env):
    step = -1
    for i in range(10000):
        o, r, tm, tc, info = env.step([0,0])
        if len(env.agent.crashed_objects)>0:
            step = env.engine.episode_step
            break
        if (tm or tc) and info["arrive_dest"]:
            break
    return step

class Buffer:
    def __init__(self, size=3, shape=(640, 960, 3)):
        self.dq = deque(maxlen=size)
        self.size = size
        self.shape = shape

    def insert(self, stuff):
        if len(self.dq) < self.size:
            self.dq.append(stuff)
        else:
            garbage = self.dq.popleft()
            del garbage
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

    def export(self, episode_folder):
        def log_data_mult(data, instance_folder):
            try:
                # Save the world annotation in world{}.json
                file_path = os.path.join(instance_folder, "world_{}.json".format(identifier))
                with open(file_path, 'w') as f:
                    json.dump(data["world"], f, indent=2)
                # Save the lidar observation in lidar{id}.json
                file_path = os.path.join(instance_folder, "lidar_{}.json".format(identifier))
                with open(file_path, 'w') as f:
                    observation = {}
                    observation['lidar'] = data["lidar"]
                    json.dump(observation, f, indent=2)
                for perspective in data["rgb"].keys():
                    # Save the rgb observation in rgb_{perspective}_{id}.png
                    file_path = os.path.join(instance_folder, "rgb_{}_{}.png".format(perspective, identifier))
                    cv2.imwrite(file_path, data["rgb"][perspective])
                    # Save the instance segmentation mask in mask_{perspective}_{id}.png
                    file_path = os.path.join(instance_folder, "mask_{}_{}.png".format(perspective, identifier))
                    cv2.imwrite(file_path, data["mask"][perspective])
                if "top_down" in data.keys():
                    # Save the top-down view in top_down{id}.json
                    file_path = os.path.join(instance_folder, "top_down_{}.png".format(identifier))
                    cv2.imwrite(file_path, data["top_down"])
                if "front" in data.keys():
                    # Save the rendering in front_{id}.png
                    file_path = os.path.join(instance_folder, "front_{}.png".format(identifier))
                    cv2.imwrite(file_path, data["front"])
                # Save the mapping front ID to color for visualization purpose. in metainformation_{}.json
                file_path = os.path.join(instance_folder, "metainformation_{}.json".format(identifier))
                with open(file_path, 'w') as f:
                    # write the dictionary to the file in JSON format
                    json.dump(data["log_mapping"], f, indent=2)
                return 0
            except Exception as e:
                raise e
        os.makedirs(episode_folder, exist_ok=True)
        for item in self.dq:
            identifier, frame_summary = item
            frame_folder = os.path.join(episode_folder, identifier)
            os.makedirs(frame_folder, exist_ok=True)
            log_data_mult(frame_summary, frame_folder)



def record_frame(env, lidar, camera, instance_camera):
    engine = env.engine
    positions = [(0., 0.0, 1.5), (0., 0., 1.5), (0., 0., 1.5), (0., 0, 1.5), (0., 0., 1.5),
                 (0., 0., 1.5)]
    hprs = [[0, 0, 0], [45, 0, 0], [135, 0, 0], [180, 0, 0], [225, 0, 0], [315, 0, 0]]
    names = ["front", "leftf", "leftb", "back", "rightb", "rightf"]
    cloud_points, _ = lidar.perceive(
        env.agent,
        physics_world=env.engine.physics_world.dynamic_world,
        num_lasers=env.agent.config["lidar"]["num_lasers"],
        distance=env.agent.config["lidar"]["distance"],
        show=False,
    )
    rgb_dict = {}
    for position, hpr, name in zip(positions, hprs, names):
        mask = instance_camera.perceive(to_float=True, new_parent_node=env.agent.origin, position=position,
                                        hpr=hpr)
        rgb = camera.perceive(to_float=True, new_parent_node=env.agent.origin, position=position, hpr=hpr)
        rgb_dict[name] = dict(
            mask=mask,
            rgb=rgb
        )
    mapping = engine.c_id
    visible_ids_set = set()
    filter = lambda r, g, b, c: not (r == 1 and g == 1 and b == 1) and not (
            r == 0 and g == 0 and b == 0) and (c > 240)
    Log_Mapping = dict()
    for perspective in rgb_dict.keys():
        visible_ids, log_mapping = get_visible_object_ids(rgb_dict[perspective]["mask"], mapping, filter)
        visible_ids_set.update(visible_ids)
        rgb_dict[perspective]["visible_ids"] = visible_ids
        Log_Mapping.update(log_mapping)
    valid_objects = engine.get_objects(
        lambda x: l2_distance(x, env.agent) <= 50 and
                  x.id != env.agent.id and
                  not isinstance(x, BaseTrafficLight))
    observing_camera = []
    for obj_id in valid_objects.keys():
        final = []
        for perspective in rgb_dict.keys():
            if obj_id in rgb_dict[perspective]["visible_ids"]:
                final.append(perspective)
        observing_camera.append(final)
    visible_mask = [True if x in visible_ids_set else False for x in valid_objects.keys()]
    objects_annotations = generate_annotations(list(valid_objects.values()), env, visible_mask,
                                               observing_camera)
    ego_annotation = genearte_annotation(env.agent, env)
    scene_dict = dict(
        ego=ego_annotation,
        objects=objects_annotations
    )
    final_summary = postprocess_annotation(
        env = env, lidar = cloud_points, rgb_dict = rgb_dict,
        scene_dict = scene_dict, log_mapping= Log_Mapping, debug=True

    )
    return final_summary

def record_accident(env, buffer, countdown, session_folder):
    engine = env.engine
    camera = engine.get_sensor("rgb")
    instance_camera = engine.get_sensor("instance")
    lidar = engine.get_sensor("lidar")
    while countdown > 0:
        o, r, tm, tc, info = env.step([0, 0])
        identifier = f"{env.current_seed}_{engine.episode_step}"
        scene_summary = record_frame(env, lidar, camera, instance_camera)
        buffer.insert((identifier,scene_summary))
        if tm or tc:
            break
        countdown -= 1
    buffer_size = len(buffer.dq)
    final_step = engine.episode_step
    initial_step = final_step - buffer_size + 1
    folder = os.path.join(session_folder, "{}_{}_{}".format(env.current_seed, initial_step, final_step))
    buffer.export(folder)
    return tm, tc

def generate_safe_data(env, seeds, folder):
    env.reset()
    os.makedirs(folder, exist_ok=True)
    print("This session is saved in folder {}".format(folder))
    env.agent.expert_takeover = True
    offset = 30
    history = 25
    future = 5
    annotation_buffer = Buffer(history + future)
    engine = env.engine
    camera = engine.get_sensor("rgb")
    instance_camera = engine.get_sensor("instance")
    lidar = engine.get_sensor("lidar")
    inception = False
    for seed in seeds:
        env.reset(seed)
        collision_step = find_collision_step(env)
        if collision_step < 0:
            #there can be no collision in cated scenario
            print("No collision in this seed {}".format(env.current_seed))
            env.reset()
            inception = False
            annotation_buffer.flush()
            continue
        env.reset(seed)
        for _ in range(1, collision_step - offset):
            env.step([0, 0])
        for i in range(10000):
            o, t, tm, tc, info = env.step([0, 0])
            if not inception:
                identifier = f"{env.current_seed}_{engine.episode_step}"
                scene_summary = record_frame(env, lidar, camera, instance_camera)
                annotation_buffer.insert((identifier, scene_summary))
                if len(env.agent.crashed_objects) > 0:
                    print("Collision happened at step {}".format(env.engine.episode_step))
                    inception = True
                    tm, tc = record_accident(env, annotation_buffer, future, folder)
            if tm or tc:
                break
        inception = False
        annotation_buffer.flush()

if __name__ == "__main__":
    scenario_summary, _, _ = sd_utils.read_dataset_summary("E:/Bolei/cat")
    env = ScenarioDiverseEnv(
        {
            "sequential_seed": True,
            "reactive_traffic": True,
            "use_render": True,
            "data_directory": "E:/Bolei/cat",
            "num_scenarios": len(scenario_summary),
            "agent_policy": ReplayEgoCarPolicy,
            "sensors": dict(
                rgb=(RGBCamera, 960, 720),
                instance=(InstanceCamera, 960,720)
            ),
            "height_scale": 1

        }
    )
    seeds = [i for i in range(len(scenario_summary))]
    generate_safe_data(env, seeds, "test_collision")






