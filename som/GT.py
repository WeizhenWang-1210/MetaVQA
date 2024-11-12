import argparse

import PIL
import os
from metadrive.envs.scenario_env import ScenarioDiverseEnv
from metadrive.scenario import utils as sd_utils
import sys
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.replay_policy import InterventionPolicy
from metadrive.component.sensors.instance_camera import InstanceCamera
import json
import tqdm
from som.closed_loop_evaluations import observe_som
import numpy as np
from som.closed_loop_utils import computeFDE, ACTION, classify_speed
from som.navigation import dynamic_get_navigation_signal, get_navigation_signal, TurnAction
import math
from som.qa_utils import split_list


def visualize_action(action_squence, duration):
    return "->".join([ACTION.get_action(action) for action in action_squence[::duration]])


def job(rank, data_directory, total_scenarios, seeds, save_directory, traj_name, qa_name, imfolders):
    traj_path = os.path.join(save_directory, f"{rank}_{traj_name}")
    qa_path = os.path.join(save_directory, f"{rank}_{qa_name}")
    qas = dict()
    qid = 0
    try:
        env = ScenarioDiverseEnv(
            {
                "sequential_seed": True,
                "use_render": False,
                "data_directory": data_directory,
                "num_scenarios": total_scenarios,
                "agent_policy": InterventionPolicy,
                "sensors": dict(
                    rgb=(RGBCamera, 1600, 900),
                    instance=(InstanceCamera, 1600, 900)
                ),
                "height_scale": 1
            }
        )
        actions = [ACTION.TURN_LEFT, ACTION.TURN_RIGHT, ACTION.SLOW_DOWN, ACTION.BRAKE, ACTION.KEEP_STRAIGHT]
        duration = 5
        gt_trajectories = dict()
        optimal_trajectories = dict()
        optimal_sequences = dict()
        for seed in tqdm.tqdm(seeds, desc="Working on scenes", unit="scene"):
            o, _ = env.reset(seed)
            if env.engine.data_manager.current_scenario_file_name.find("adv") != -1:
                print(f"Seed {seed}: {env.engine.data_manager.current_scenario_file_name} contains collsion, skip it")
                continue
            optimal_sequence = []
            scenario = env.engine.data_manager.current_scenario
            ego_id = scenario["metadata"]["sdc_id"]
            ego_track = scenario["tracks"][ego_id]
            ego_traj = ego_track["state"]["position"][..., :2]
            total_step = ego_traj.shape[0]
            for start_step in tqdm.tqdm(range(0, total_step, duration), desc=f"Scene {seed} step simulations",
                                        unit="step"):
                print(f"Optimal Sequence till {start_step} for env{seed}: ",
                      visualize_action(optimal_sequence, duration))
                # Generate gt action at each timestamp
                # Extract G.T. trajectory
                if start_step + duration < total_step:
                    gt_trajectory = ego_traj[start_step:start_step + duration]
                else:
                    gt_trajectory = ego_traj[start_step:]
                trajectories = []
                # Roll out individual actions
                observed = False
                for action in tqdm.tqdm(actions, desc=f"Rolling out action for {seed} {start_step}", unit="choice"):
                    env.reset(seed)
                    trajectory = []
                    # Go to the desired step, following already generated optimal action.
                    for t in range(start_step):
                        env.step(ACTION.get_control(optimal_sequence[t]))

                    # Now we've followed the optimal action,
                    if not observed:
                        obs_array, id2labels = observe_som(env)
                        navigation = dynamic_get_navigation_signal(scenario, env.episode_step, env)
                        observed = True
                    # Start the intervention for duration
                    for t in range(duration):
                        o, r, tm, tc, info = env.step(ACTION.get_control(action))
                        if tm or tc:
                            break
                        trajectory.append(env.agent.position)
                    # Collect generated traj v.s. gt. traj
                    if len(trajectory) > gt_trajectory.shape[0]:
                        trajectory = trajectory[:gt_trajectory.shape[0]]
                    trajectories.append(trajectory)
                # The optimal trajectory is the action with the least FDE from the original trajectory.
                FDEs = [computeFDE(gt_trajectory, np.array(trajectory)) for trajectory in trajectories]
                optimal_action = FDEs.index(min(FDEs))
                speed_class = classify_speed(env.agent.speed)
                criteria = {
                    "slow": "(0-10 mph)",
                    "moderate": "(10-30 mph)",
                    "fast": "(30-50 mph)",
                    "very fast": "(50+ mph)",
                }
                desc = criteria[speed_class]
                explanation = dict(
                    TURN_LEFT="if chosen, your steering wheel will be turned left while your speed will remain relatively constant",
                    TURN_RIGHT="if chosen, your steering wheel will be turned right while your speed will remain relatively constant",
                    SLOW_DOWN="if chosen, your brake will be gently pressed to reduce speed while your steering wheel will be unturned",
                    BRAKE="if chosen, your brake will be pressed hard in order to fully stop while your steering wheel will be unturned",
                    KEEP_STRAIGHT="if chosen, your steering wheel will be unturned while your speed will remain relatively constant"
                )
                question = (
                    f"You are driving on the road with {speed_class} speed{desc}, and your current navigation command is \"{navigation}\". "
                    f"Carefully examine the image as your front observation, and choose the safest action to execute for the next 0.5 seconds from the following options:\n"
                    f"(A) TURN_LEFT, {explanation[ACTION.get_action(ACTION.TURN_LEFT)]}.\n"
                    f"(B) TURN_RIGHT, {explanation[ACTION.get_action(ACTION.TURN_RIGHT)]}.\n"
                    f"(C) SLOW_DOWN, {explanation[ACTION.get_action(ACTION.SLOW_DOWN)]}.\n"
                    f"(D) BRAKE, {explanation[ACTION.get_action(ACTION.BRAKE)]}.\n"
                    f"(E) KEEP_STRAIGHT, {explanation[ACTION.get_action(ACTION.KEEP_STRAIGHT)]}.\n"
                    f"Answer in a single capitalized character chosen from [\"A\", \"B\", \"C\", \"D\", \"E\"].")
                answer = chr(ord('A') + optimal_action)
                optimal_sequence += [optimal_action] * duration
                option2answer = dict(
                    A="TURN_LEFT", B="TURN_RIGHT", C="SLOW_DOWN", D="BRAKE", E="KEEP_STRAIGHT"
                )
                print(question)
                print(answer)
                print(option2answer)
                observation_name = f"{seed}_{start_step}.png"
                obs_path = os.path.join(imfolders, observation_name)
                obs_array = obs_array[:, :, ::-1]
                PIL.Image.fromarray(obs_array).save(obs_path)
                record = dict(
                    question=question, answer=answer, explanation="", type="embodied_action", obs=[obs_path],
                    objects=list(id2labels.keys()),
                    options=option2answer, domain="sim", world=[env.engine.data_manager.current_scenario_file_name],
                )
                qas[qid] = record
                qid += 1

            # finally, collect the original trajectory and the optimal trajectory
            o, _ = env.reset(seed)
            gt = []
            run = True
            while run:
                gt.append(env.agent.position)
                o, r, tm, tc, info = env.step([0, 0])
                if tm or tc:
                    run = False
            optimal_trajectory = []
            o, _ = env.reset(seed)
            for action in optimal_sequence:
                optimal_trajectory.append(env.agent.position)
                o, r, tm, tc, info = env.step(ACTION.get_control(action))
                if tm or tc:
                    run = False
            gt_trajectories[env.engine.data_manager.current_scenario_file_name] = gt
            optimal_trajectories[env.engine.data_manager.current_scenario_file_name] = optimal_trajectory
            optimal_sequences[env.engine.data_manager.current_scenario_file_name] = optimal_sequence
    except Exception as e:
        raise e
    finally:
        json.dump(
            dict(gt=gt_trajectories, opt=optimal_trajectories, act=optimal_sequences),
            open(traj_path, "w"),
            indent=2
        )
        json.dump(qas, open(qa_path, "w"), indent=1)
        env.close()

import multiprocessing as mp
if __name__ == "__main__":
    num_scenarios = 120
    qa_name = "qa.json"
    traj_name = "traj.json"
    data_dir= "/data_weizhen/scenarios"
    save_dir= "/data_weizhen/metavqa_cvpr/gts/"
    im_dir = "/data_weizhen/metavqa_cvpr/gts/obs"
    num_proc = 30
    seeds = split_list(list(range(num_scenarios)), num_proc)
    processes=[]
    for proc_id in range(num_proc):
        print(f"Sending job {proc_id} on scenarios {seeds[proc_id]}")
        p = mp.Process(
            target=job,
            args=(
                proc_id, data_dir, num_scenarios, seeds[proc_id], save_dir, traj_name, qa_name, im_dir
            )
        )
        print(f"Successfully sent {proc_id} on scenarios {seeds[proc_id]}")
        processes.append(p)
        p.start()
    for p in processes:
        p.join()