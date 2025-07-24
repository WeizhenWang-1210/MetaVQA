import json
import os
import random

import numpy as np
from PIL import Image

from closed_loop.configs import INTERVENED, NON_INTERVENED
from closed_loop.utils.log_utils import capture_som
from closed_loop.configs import ACTION_STATISTICS, RECORD_BUFFER
from closed_loop.navigation import get_trajectory, dest_navigation_signal
from closed_loop.closed_loop_utils import computeADE, absoluteFDE
from metadrive.envs.scenario_env import ScenarioEnv


def always_stop(intervened):
    ACTION_MAPPING = INTERVENED
    ACTION_STATISTICS["d"] += 1
    return ACTION_MAPPING["d"], True


def always_straight(intervened):
    ACTION_MAPPING = INTERVENED
    ACTION_STATISTICS["e"] += 1
    return ACTION_MAPPING["e"], True


def random_action(intervened):
    if intervened:
        ACTION_MAPPING = INTERVENED
        action = random.choice(list(ACTION_MAPPING.keys()))
        ACTION_STATISTICS[action] += 1
        return ACTION_MAPPING[action], True
    else:
        ACTION_MAPPING = NON_INTERVENED
        action = random.choice(list(ACTION_MAPPING.keys()))
        ACTION_STATISTICS[action] += 1
        if not (ACTION_MAPPING[action][0] == 0 and ACTION_MAPPING[action][1] == 0):
            intervened = True
        return ACTION_MAPPING[action], intervened


def closed_loop_baselines(env: ScenarioEnv, seeds, actor: callable, record_folder=None):
    record = record_folder is not None
    total_out_of_road = total_completions = total_rewards = num_collisions = num_src_collisions = 0
    command_frequency = 5
    collided_scenarios = set()
    offroad_scenarios = set()
    ADEs, FDEs = [], []
    for seed in seeds:
        env.reset(seed)
        original_trajectory = get_trajectory(env)
        max_step = original_trajectory.shape[0]
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
        step = episodic_completion = episodic_reward = 0
        intervention = [0, 0]
        trajectory = []
        offroad = collide = 0
        while run:
            index = max(int(env.engine.episode_step), 0)
            if index >= max_step:
                print("Time horizon ended")
                run = False
                break
            trajectory.append(env.agent.position)
            if step % command_frequency == 0:
                print("Perceive & Act")
                dir, dist = dest_navigation_signal(env.engine.data_manager.current_scenario, timestamp=env.episode_step, env=env)
                obs, front, id2label = capture_som(env)
                intervention, intervened = actor(intervened)
                print(intervention, intervened)
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["action"] = intervention
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["navigation"] = (dir, dist)
            assert not (intervention[0] == 0 and intervention[1] == 0)
            o, r, tm, tc, info = env.step(intervention)
            episodic_reward, episodic_completion = info["episode_reward"], info["route_completion"]
            if len(env.agent.crashed_objects) > 0:
                print("VLM still collided.")
                collided_scenarios.add(env.engine.data_manager.current_scenario_file_name)
                collide = 1
            if info["out_of_road"]:
                print("VLM wander off road")
                offroad = 1
                offroad_scenarios.add(env.engine.data_manager.current_scenario_file_name)
                run = False
            if tm or tc:
                run = False
            step += 1
        num_collisions += collide
        total_out_of_road += offroad
        total_rewards += episodic_reward
        total_completions += episodic_completion
        ADE, FDE = computeADE(original_trajectory, np.array(trajectory)), absoluteFDE(original_trajectory, np.array(trajectory))
        ADEs.append(ADE)
        FDEs.append(FDE)
        if record:
            for episode_id, frames in RECORD_BUFFER.items():
                action_buffer = {}
                folder_path = os.path.join(record_folder, str(episode_id))
                os.makedirs(folder_path, exist_ok=True)
                for frame_id, frame in frames.items():
                    obs_path = os.path.join(folder_path, f"obs_{frame_id}.jpg")
                    front_path = os.path.join(folder_path, f"front_{frame_id}.jpg")
                    Image.fromarray(frame["obs"][:, :, ::-1]).save(obs_path)
                    Image.fromarray(frame["front"][:, :, ::-1]).save(front_path)
                    action_buffer[frame_id] = dict(
                        scene=env.engine.data_manager.current_scenario_file_name, collide=collide, offroad=offroad,
                        ADE=ADE, FDE=FDE,
                        action=frame["action"], state=frame["state"],
                        navigation=frame["navigation"],
                    )
                json.dump(action_buffer, open(os.path.join(folder_path, "action_buffer.json"), "w"))
            RECORD_BUFFER.clear()
        print(f"Finished seed {env.engine.current_seed}")
        print(f"episodic_reward: {episodic_reward}")
        print(f"episodic_completion:{episodic_completion}")
        print(f"ADE:{ADE}; FDE{FDE}")
    return num_collisions, num_src_collisions, total_rewards, total_completions, ADEs, FDEs, total_out_of_road, list(collided_scenarios), list(offroad_scenarios)
