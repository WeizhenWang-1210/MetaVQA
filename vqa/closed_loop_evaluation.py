from metadrive.envs.scenario_env import ScenarioDiverseEnv
from metadrive.scenario import utils as sd_utils
#from vqa.online_eval import load_model, eval_model
#from vqa.closed_loop_collision import Buffer, vector_transform #,load_and_process_images
#import torch
from metadrive.scenario.parse_object_state import parse_object_state
from collections import defaultdict
ACTION_STATISTICS = defaultdict(lambda: 0)
device = "cuda"
RECORD_BUFFER=defaultdict(lambda:defaultdict(lambda:dict()))
RECORD_FOLDER = "C:/school/closed_loops"
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.replay_policy import InterventionPolicy

def capture(env):
    position = (0., 0.0, 1.5)
    hpr = [0, 0, 0]
    obs = env.engine.get_sensor("rgb").perceive(to_float=False, new_parent_node=env.agent.origin, position=position, hpr=hpr)
    front = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, -15, 3], [0, -0.8, 0])
    RECORD_BUFFER[env.current_seed][env.engine.episode_step] = dict(action=None, front=front, obs=obs, state=[env.agent.position, env.agent.heading, env.agent.speed])




def convert_action(action, intervened):
    if intervened:
        ACTION_MAPPING = {
            "left": [0.5, 0.8],
            "right": [-0.5, 0.8],
            "stop": [0, -0.2],
            "none": [0, 0.3]
        }
    else:
        ACTION_MAPPING = {
            "left": [0.5, 1],
            "right": [-0.5, 1],
            "stop": [0, -0.1],
            "none": [0, 0]
        }
    if action in ACTION_MAPPING.keys():
        return ACTION_MAPPING[action]
    else:
        return [0, 0]
import random
def random_action(intervened):
    if intervened:
        ACTION_MAPPING = {
            "left": [0.5, 0.8],
            "right": [-0.5, 0.8],
            "stop": [0, -0.2],
            "none": [0, 0.3]
        }
        action = random.choice(list(ACTION_MAPPING.values()))
    else:
        ACTION_MAPPING = {
            "left": [0.5, 1],
            "right": [-0.5, 1],
            "stop": [0, -0.1],
            "none": [0, 0]
        }
        action = random.choice(list(ACTION_MAPPING.values()))
        if not(action[0]==0 and action[1]==0):
            intervened = True
    return action, intervened


"""
{   
    env_0={
        step0={
            action=None,
            obs=None,
        }
    }
}

"""
"""def preprocess_observation(destination, buffer, vis_processors, text_processors):
    im, displacement, now_world_position = buffer.read_tensor(vis_processors)  #5*6*3*364*364
    now_heading = buffer.dq[-1][2]
    destination = vector_transform(now_world_position, now_heading, destination)
    destination = int(destination[0]), int(destination[1])
    question = text_processors(
        "You are the driver, what is the safest action to do? Choose from one option from: (A) left ;(B) right ;(C) stop ;(D) none. For example, if you want to turn left, answer \"A\" ")
    im = torch.unsqueeze(im, 0)
    question = question
    return {
        "vfeats": im,
        "questions": [question],
        "answers": ["something"]
    }"""


def record_frame(env: ScenarioDiverseEnv, record=False):
    engine = env.engine
    camera = engine.get_sensor("rgb")
    positions = [(0., 0.0, 1.5)]#[(0., 0.0, 1.5), (0., 0., 1.5), (0., 0., 1.5), (0., 0, 1.5), (0., 0., 1.5),(0., 0., 1.5)]
    hprs = [[0, 0, 0]] #[[0, 0, 0], [45, 0, 0], [135, 0, 0], [180, 0, 0], [225, 0, 0], [315, 0, 0]]
    names = ["front"] #["front", "leftf", "leftb", "back", "rightb", "rightf"]
    rgb_dict = {}
    for position, hpr, name in zip(positions, hprs, names):
        rgb = camera.perceive(to_float=False, new_parent_node=env.agent.origin, position=position, hpr=hpr)
        #if record:
            #capture(env)
        rgb_dict[name] = rgb
    return rgb_dict, env.agent.position, env.agent.heading


def get_trajectory_info(engine):
    # Directly get trajectory from data manager
    trajectory_data = engine.data_manager.current_scenario["tracks"]
    sdc_track_index = str(engine.data_manager.current_scenario["metadata"]["sdc_id"])
    ret = []
    for i in range(len(trajectory_data[sdc_track_index]["state"]["position"])):
        ret.append(parse_object_state(
            trajectory_data[sdc_track_index],
            i,
        ))
    return ret


"""def generation_action(destination, buffer, model, vis_processors, text_processors, intervened):
    observation_dict = preprocess_observation(destination, buffer, vis_processors, text_processors)
    #print(observation_dict["questions"])
    answer = eval_model(model, observation_dict, vis_processors, text_processors)
    answer = answer[0].lower()
    ACTION_STATISTICS[answer] += 1
    #print(answer)
    intervention = convert_action(answer, intervened)
    if not (intervention[0] == 0 and intervention[1] == 0):
        return intervention, True
    else:
        return intervention, False"""

import os
from PIL import Image
import json
def closed_loop(env: ScenarioDiverseEnv, seeds, record=True):
    #model, vis_processors, text_processors = load_model(False)
    #model.to(device)
    num_src_collisions = 0
    num_collisions = 0
    total_rewards = 0
    command_frequency = 5
    observation_frequency = 5
    for seed in seeds:
        env.reset(seed)
        traj = get_trajectory_info(env.engine)
        max_step = len(traj)
        destination = traj[max_step - 1]['position']
        print(f"Rolling out seed {env.engine.current_seed}")
        roll_out = True
        while roll_out:
            o, r, tm, tc, info = env.step([0, 0])
            if len(env.agent.crashed_objects) > 0:
                print(f"{seed} contains collision.")
                num_src_collisions += 1
                roll_out = False
            if tm or tc:
                roll_out = False
        env.reset(seed)
        print(f"Evaluating seed {env.engine.current_seed}")
        run = True
        intervened = False
        episodic_reward = 0
        #buffer = Buffer(size=5)
        step = 0
        intervention = [0, 0]
        while run:
            capture(env)
            index = max(int(env.engine.episode_step), 0)
            if index >= max_step:
                run = False
            if step % observation_frequency == 0:
                print("perceive")
                #rgb_observation, current_position, current_heading = record_frame(env, record)
                #buffer.insert((rgb_observation, current_position, current_heading))
            if step % command_frequency == 0:
                print("command")
                #intervention, intervened = generation_action(destination, buffer, model, vis_processors,
                #                                             text_processors, intervened)
                intervention,intervened = random_action(intervened)
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["action"] = intervention
                print(intervention, intervened)

            o, r, tm, tc, info = env.step(intervention)
            episodic_reward = info["episode_reward"]
            if len(env.agent.crashed_objects) > 0:
                if record:
                    capture(env)
                print("VLM still collided.")
                num_collisions += 1
                run = False
            if (tm or tc) and not info["out_of_road"]:
                run = False
            step += 1
        total_rewards += episodic_reward
        if record:
            for episode_id, frames in RECORD_BUFFER.items():
                action_buffer={}
                folder_path = os.path.join(RECORD_FOLDER, str(episode_id))
                os.makedirs(folder_path, exist_ok=True)
                for frame_id, frame in frames.items():
                    obs_path = os.path.join(folder_path, f"obs_{frame_id}.jpg")
                    front_path = os.path.join(folder_path, f"front_{frame_id}.jpg")
                    Image.fromarray(frame["obs"][:, :, ::-1]).save(obs_path)
                    Image.fromarray(frame["front"][:, :, ::-1]).save(front_path)
                    action_buffer[frame_id] = dict(action=frame["action"],state=frame["state"])
                json.dump(action_buffer, open(os.path.join(folder_path, "action_buffer.json"), "w"))
            RECORD_BUFFER.clear()
        print(f"Finished seed {env.engine.current_seed}")
        print(f"episodic_reward: {episodic_reward}")
    return num_collisions, num_src_collisions, total_rewards





def main():
    use_render = False
    traffic = "C:\school\Bolei\cat\cat"
    scenario_summary, scenario_ids, scenario_files = sd_utils.read_dataset_summary(traffic)
    num_scenarios = len(scenario_summary)
    env_config = {
        "sequential_seed": True,
        "use_render": use_render,
        "data_directory": traffic,
        "num_scenarios": num_scenarios,
        "agent_policy": InterventionPolicy,
        "sensors": dict(
            rgb=(RGBCamera, 1920, 1080)
        ),
        "height_scale": 1
    }
    env = ScenarioDiverseEnv(env_config)
    total_collision, total_src_collision, total_rewards = closed_loop(env, list(range(num_scenarios)))
    summary = dict(
        src=traffic,
        num_scenarios=num_scenarios,
        total_collision=total_collision,
        total_src_collision=total_src_collision,
        total_rewards=total_rewards,
        action_statistics=ACTION_STATISTICS
    )
    import json
    json.dump(summary, open("./online_eval_baseline.json", "w"), indent=2)


if __name__ == "__main__":
    main()
