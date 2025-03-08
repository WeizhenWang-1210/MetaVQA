import json
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import InterventionPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from som.closed_loop_evaluations import observe, in_forbidden_area
from som.embodied_utils import ACTION
from collections import defaultdict
import os
from PIL import Image

def select_pkls(traj):
    scenes = list(traj["crash"].keys())
    good_scenes = set(scenes)
    for key,value in traj["crash"].items():
        if value == True:
            good_scenes.remove(key)
    for key,value in traj["off"].items():
        if value == True:
            good_scenes.remove(key)
    for key,value in traj["completion"].items():
        if value < 0.9:
            good_scenes.remove(key)
    return list(good_scenes)

def replay():
    data_directory = "/data_weizhen/scenarios"
    total_scenarios = 120
    env = ScenarioEnv(
        {
            "sequential_seed": True,
            "use_render": False,
            "data_directory": data_directory,
            "num_scenarios": total_scenarios,
            "agent_policy": InterventionPolicy,
            "sensors": dict(
                rgb=(RGBCamera, 960, 540),
            ),
            "height_scale": 1,
            "vehicle_config": dict(vehicle_model="static_default")
        }
    )
    traj = json.load(open("/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/traj.json"))
    scenes = list(traj["gt"].keys())
    freq = 5
    for seed in range(120):
        env.reset(seed)
        if env.engine.data_manager.current_scenario_file_name != "sd_nuscenes_v1.0-trainval_scene-0502.pkl":
            continue
        else:
            print(seed)
            exit()
        if env.engine.data_manager.current_scenario_file_name not in scenes:
            continue
        print(f"{env.engine.data_manager.current_scenario_file_name}")
        run = True
        while run:
            if env.engine.episode_step >= len(traj["act"][env.engine.data_manager.current_scenario_file_name]):
                run = False
                break
            front = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, -15, 3], [0, -0.8, 0])
            os.makedirs(f"/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/replay/{seed}", exist_ok=True)
            if env.engine.episode_step % freq == 0:
                Image.fromarray(front[:, :, ::-1]).save(
                    f"/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/replay/{seed}/{str(env.engine.episode_step)}.png")
            action_code = traj["act"][env.engine.data_manager.current_scenario_file_name][env.engine.episode_step]
            o, r, tm, tc, info = env.step(ACTION.get_control(action_code))
            if len(env.agent.crashed_objects) > 0:
                print(f"Collision! {seed}")
                run = False
            if in_forbidden_area(env.agent):
                print(f"Off! {seed}")
                run = False


if __name__ == "__main__":
    traj = json.load(open("/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/traj.json"))
    selected_worlds = select_pkls(traj)
    total_qa = json.load(open("/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/qa.json"))
    new_qa = dict()
    local_id = 0
    for qid, value in total_qa.items():
        if value["world"][0] in selected_worlds:
            new_qa[local_id] = value
            local_id += 1
    json.dump(
        new_qa,
        open("/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/filtered.json","w"),
        indent=2
    )
