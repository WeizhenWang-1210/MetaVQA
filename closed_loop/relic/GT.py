import PIL
import os
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.replay_policy import InterventionPolicy
from metadrive.component.sensors.instance_camera import InstanceCamera
import json
import tqdm
from som.closed_loop_evaluations import observe_som, in_forbidden_area
import numpy as np
from som.closed_loop_utils import computeFDE, ACTION, classify_speed
from som.navigation import dynamic_get_navigation_signal
from vqa.vqagen.qa_utils import split_list
import traceback


def visualize_action(action_squence, duration):
    return "->".join([ACTION.get_action(action) for action in action_squence[::duration]])


def job(rank, data_directory, total_scenarios, seeds, save_directory, traj_name, qa_name, imfolders, duration):
    traj_path = os.path.join(save_directory, f"{rank}_{traj_name}")
    qa_path = os.path.join(save_directory, f"{rank}_{qa_name}")
    qas = dict()
    qid = 0
    actions = [
        ACTION.TURN_LEFT, ACTION.TURN_RIGHT, ACTION.SLOW_DOWN, ACTION.BRAKE,
               ACTION.KEEP_STRAIGHT, ACTION.SPEED_UP, ACTION.BIG_LEFT, ACTION.BIG_RIGHT
               ]
    gt_trajectories = dict()
    optimal_trajectories = dict()
    optimal_sequences = dict()
    crash_flags = dict()
    off_flags = dict()
    completions = dict()
    try:
        env = ScenarioEnv(
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
                "height_scale": 1,
                "vehicle_config": dict(vehicle_model="static_default")
            }
        )
        for seed in tqdm.tqdm(seeds, desc="Working on scenes", unit="scene"):
            o, _ = env.reset(seed)
            if env.engine.data_manager.current_scenario_file_name.find("adv") != -1:
                print(f"Seed {seed}: {env.engine.data_manager.current_scenario_file_name} contains collsion, skip it")
                continue
            # Ground Truth trajectory to termination
            gt = []
            run = True
            total_step = 0
            while run:
                gt.append(env.agent.position)
                o, r, tm, tc, info = env.step([0, 0])
                total_step += 1
                if tm or tc:
                    run = False
            gt_trajectories[env.engine.data_manager.current_scenario_file_name] = gt
            #
            optimal_sequence = []
            scenario = env.engine.data_manager.current_scenario
            ego_id = scenario["metadata"]["sdc_id"]
            ego_track = scenario["tracks"][ego_id]
            ego_traj = ego_track["state"]["position"][..., :2]

            optimal_off_flag, optimal_crash_flag = False, False
            for start_step in tqdm.tqdm(range(0, total_step, duration),
                                        desc=f"Proc-{rank} scene-{seed} step simulations",
                                        unit="step"):
                print(f"Optimal Sequence till {start_step} for env{seed}: ",
                      visualize_action(optimal_sequence, duration))
                # Generate gt action at each timestamp
                # Extract G.T. trajectory
                if start_step + duration < total_step:
                    gt_trajectory = ego_traj[start_step:start_step + duration]
                else:
                    gt_trajectory = ego_traj[start_step:]
                if gt_trajectory.shape[0] <= 0:
                    print(f"Finished at {env.engine.episode_step} for seed {seed}")
                    break
                trajectories = []
                # Roll out individual actions
                observed = False
                dangerous_actions = []

                for action in tqdm.tqdm(actions,
                                        desc=f"Proc-{rank}: Rolling out action for scene-{seed} starting at {start_step}",
                                        unit="action"):
                    print(f"Reset {seed} for {ACTION.get_action(action)}")
                    env.reset(seed)
                    trajectory = []
                    # Go to the desired step, following already generated optimal action.
                    for t in range(start_step):
                        env.step(ACTION.get_control(optimal_sequence[t]))
                        if len(env.agent.crashed_objects) > 0:
                            optimal_crash_flag = True
                            print(f"Crash in rollout! Seed {seed} at {env.engine.episode_step}")
                        if in_forbidden_area(env.agent):
                            optimal_off_flag = True
                            print(f"Out-of-road in rollout! Seed {seed} at {env.engine.episode_step}")
                    # Now we've followed the optimal action,
                    if not observed:
                        obs_array, id2labels = observe_som(env)
                        navigation = dynamic_get_navigation_signal(scenario, env.episode_step, env)
                        observed = True
                    # Start the intervention for duration
                    for t in range(duration):
                        o, r, tm, tc, info = env.step(ACTION.get_control(action))
                        if len(env.agent.crashed_objects) > 0 or in_forbidden_area(env.agent):
                            print(
                                f"Dangerous to do {ACTION.get_action(action)} at {env.engine.episode_step} for seed {seed}")
                            dangerous_actions.append(action)
                            #break
                        #assert len(env.agent.crashed_objects) <= 0, "Crash in predict!"
                        trajectory.append(env.agent.position)
                    # Collect generated traj v.s. gt. traj
                    trajectory = trajectory[:gt_trajectory.shape[0]]
                    trajectories.append(trajectory)
                #assert all([len(trajectory) == gt_trajectory.shape[0] for trajectory in trajectories]), f"{str([len(trajectory) for trajectory in trajectories])}, {str(gt_trajectory.shape[0])}"
                # The optimal trajectory is the action with the least FDE from the original trajectory.
                minFDE = 100000000000000
                optimal_action = None
                print(f"Dangerous actions:{str(dangerous_actions)}")
                for action, trajectory in zip(actions, trajectories):
                    if action not in dangerous_actions:
                        FDE = computeFDE(gt_trajectory, np.array(trajectory))
                        if FDE < minFDE:
                            minFDE = FDE
                            optimal_action = action
                if optimal_action is None:
                    #all action leads to collision,
                    print(f"All actions are dangerous at {env.engine.episode_step} for seed {seed}. Terminate")
                    break
                speed_class = classify_speed(env.agent.speed)
                criteria = {
                    "slow": "(0-10 mph)",
                    "moderate": "(10-30 mph)",
                    "fast": "(30-50 mph)",
                    "very fast": "(50+ mph)",
                }
                desc = criteria[speed_class]
                explanation = dict(
                    TURN_LEFT="if chosen, your steering wheel will be turned slightly left while your speed will remain relatively constant",
                    TURN_RIGHT="if chosen, your steering wheel will be turned slightly right while your speed will remain relatively constant",
                    SLOW_DOWN="if chosen, your brake will be gently pressed to reduce speed while your steering wheel will be unturned",
                    BRAKE="if chosen, your brake will be pressed hard in order to fully stop while your steering wheel will be unturned",
                    KEEP_STRAIGHT="if chosen, your steering wheel will be unturned while your speed will remain relatively constant",
                    SPEED_UP="if chosen, your steering wheel will be unturned while your speed will steadily increase",
                    BIG_LEFT="if chosen, your steering wheel will be turned significantly left while your speed will remain relatively constant",
                    BIG_RIGHT="if chosen, your steering wheel will be turned significantly right while your speed will remain relatively constant"
                )
                question = (
                    f"You are the driver of a vehicle on the road, following given navigation commands. All possible navigations are as the follwing:\n"
                    f"(1) \"forward\", your immediate destination is in front of you.\n"
                    f"(2) \"go left\", your immediate destination is to your left-front.\n"
                    f"(3) \"go right\", your immediate destination is to your right-front.\n"
                    f"Currently, you are driving with {speed_class} speed{desc}, and your navigation command is \"{navigation}\". The image is observed in front of you. "
                    f"Carefully examine the image, and choose the safest action to execute for the next 0.5 seconds from the following options:\n"
                    f"(A) TURN_LEFT, {explanation[ACTION.get_action(ACTION.TURN_LEFT)]}.\n"
                    f"(B) TURN_RIGHT, {explanation[ACTION.get_action(ACTION.TURN_RIGHT)]}.\n"
                    f"(C) SLOW_DOWN, {explanation[ACTION.get_action(ACTION.SLOW_DOWN)]}.\n"
                    f"(D) BRAKE, {explanation[ACTION.get_action(ACTION.BRAKE)]}.\n"
                    f"(E) KEEP_STRAIGHT, {explanation[ACTION.get_action(ACTION.KEEP_STRAIGHT)]}.\n"
                    f"(F) SPEED_UP, {explanation[ACTION.get_action(ACTION.SPEED_UP)]}.\n"
                    f"(G) BIG_LEFT, {explanation[ACTION.get_action(ACTION.BIG_LEFT)]}.\n"
                    f"(H) BIG_RIGHT, {explanation[ACTION.get_action(ACTION.BIG_RIGHT)]}.\n"
                    f"You will attempt to follow the navigation command, but you are allowed to choose other actions to avoid potential hazards. "
                    f"Answer in a single capitalized character chosen from [\"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\"].")
                answer = chr(ord('A') + optimal_action)
                optimal_sequence += [optimal_action] * duration
                option2answer = dict(
                    A="TURN_LEFT", B="TURN_RIGHT", C="SLOW_DOWN", D="BRAKE", E="KEEP_STRAIGHT", F="SPEED_UP",
                    G="BIG_LEFT", H="BIG_RIGHT"
                )
                #print(question)
                #print(answer)
                #print(option2answer)
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
            # finally, collect the optimal trajectory
            optimal_trajectory = []
            o, _ = env.reset(seed)
            completion = 0
            for action in optimal_sequence:
                optimal_trajectory.append(env.agent.position)
                o, r, tm, tc, info = env.step(ACTION.get_control(action))
                completion = info["route_completion"]
                if len(env.agent.crashed_objects) > 0:
                    optimal_crash_flag = True
                if in_forbidden_area(env.agent):
                    optimal_off_flag = True
            optimal_trajectories[env.engine.data_manager.current_scenario_file_name] = optimal_trajectory
            optimal_sequences[env.engine.data_manager.current_scenario_file_name] = optimal_sequence
            crash_flags[env.engine.data_manager.current_scenario_file_name] = optimal_crash_flag
            off_flags[env.engine.data_manager.current_scenario_file_name] = optimal_off_flag
            completions[env.engine.data_manager.current_scenario_file_name] = float(completion)

    except Exception as e:
        print("Something Wrong! save partial results")
        print(f"Encountered issue at {env.engine.episode_step},{env.current_seed}")
        print(e)
        var = traceback.format_exc()
        debug_path = os.path.join(
            save_directory,
            f"{proc_id}_debug.json"
        )
        json.dump(
            {"proc_id": proc_id, "seed": env.seed, "end_frame": str(env.engine.episode_step),
             "end_file": env.engine.data_manager.current_scenario_file_name, "error": str(e), "trace": str(var)},
            open(debug_path, "w"),
            indent=2
        )
        raise e

    finally:
        json.dump(
            dict(gt=gt_trajectories, opt=optimal_trajectories, act=optimal_sequences,
                 crash=crash_flags, off=off_flags, completion=completions),
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
    data_dir = "/data_weizhen/scenarios"
    save_dir = "/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts"
    im_dir = "/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/obs"
    num_proc = 16
    duration= 5
    seeds = split_list(list(range(0, num_scenarios)), num_proc)
    processes = []
    for proc_id in range(num_proc):
        print(f"Sending job {proc_id} on scenarios {seeds[proc_id]}")
        p = mp.Process(
            target=job,
            args=(
                proc_id, data_dir, num_scenarios, seeds[proc_id], save_dir, traj_name, qa_name, im_dir, duration
            )
        )
        print(f"Successfully sent {proc_id} on scenarios {seeds[proc_id]}")
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
