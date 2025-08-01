import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

from closed_loop.baselines import closed_loop_baselines, random_action, always_stop, always_straight
from closed_loop.closed_loop_utils import absoluteFDE, computeADE
from closed_loop.configs import ACTION_STATISTICS, RECORD_BUFFER, ACTION2OPTION, convert_action
from closed_loop.navigation import get_trajectory, dest_navigation_signal
from closed_loop.utils.log_utils import capture_som
from closed_loop.utils.prompt_utils import prepare_prompt_dest
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import InterventionPolicy
from vqa.eval.parse_responses import parse_response
from closed_loop.models import inference, load_model





def generation_action(model, processor, tokenizer, prompt, obs, intervened):
    # Here, the image is reshaped to have RGB channeling as the last dimension
    answer = inference(model, processor, tokenizer, prompt, obs[:, :, ::-1])
    answer = parse_response(answer, ACTION2OPTION)
    answer = answer.lower()
    ACTION_STATISTICS[answer] += 1
    intervention = convert_action(answer, intervened)
    print(answer, intervention)
    if not (intervention[0] == 0 and intervention[1] == 0):
        return intervention, True
    else:
        return intervention, False


def closed_loop(env: ScenarioEnv, seeds, model_path, record_folder=None):
    record = record_folder is not None
    model, processor, tokenizer = load_model(model_path=model_path)
    model.to("cuda")
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
        # The main loop for the closed-loop evaluation
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
                prompt = prepare_prompt_dest(env.agent.speed, dir, dist)
                obs, _ , _ = capture_som(env)
                intervention, intervened = generation_action(model, processor, tokenizer, prompt, obs, intervened)
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["action"] = intervention
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["navigation"] = (dir, dist)
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["prompt"] = prompt
            assert not (intervention[0] == 0 and intervention[1] == 0)
            o, r, tm, tc, info = env.step(intervention)
            episodic_reward, episodic_completion = info["episode_reward"], info["route_completion"]
            if len(env.agent.crashed_objects) > 0:
                print("VLM collided.")
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
                        prompt=frame["prompt"], action=frame["action"], state=frame["state"],
                        navigation=frame["navigation"],
                    )
                json.dump(action_buffer, open(os.path.join(folder_path, "action_buffer.json"), "w"))
            RECORD_BUFFER.clear()
        print(f"Finished seed {env.engine.current_seed}")
        print(f"episodic_reward: {episodic_reward}")
        print(f"episodic_completion:{episodic_completion}")
        print(f"ADE:{ADE}; FDE{FDE}")
    return num_collisions, num_src_collisions, total_rewards, \
        total_completions, ADEs, FDEs, total_out_of_road, list(collided_scenarios), list(offroad_scenarios)



def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Set if don't use render")
    parser.add_argument("--num_scenarios", type=int, default=120, help="How many scenarios(from the start) to use")
    parser.add_argument("--data_directory", type=str, default=f"{cwd}/assets/scenarios",
                        help="Path to the test scenarios")
    parser.add_argument("--model_path", type=str, default="/home/chenda/ckpt/internvl_demo_merge",
                        help="Path to the model ckpt. Compatible with transformers.AutoModel")
    parser.add_argument("--prompt_schema", type=str, default="direct", help="Whether to use CoT prompting or not")
    parser.add_argument("--record_path", type=str, default=None,
                        help="Directory to store the visualizations of VLM's decision. If None, no visualization will be saved")
    parser.add_argument("--result_path", type=str, default="eval.json",
                        help="Path to the file storing experiment statistics")
    args = parser.parse_args()
    print("Running with the following parameters")
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))
    use_render = False if args.headless else True
    traffic = args.data_directory
    num_scenarios = args.num_scenarios
    env_config = {
        "sequential_seed": True,
        "use_render": use_render,
        "data_directory": traffic,
        "num_scenarios": num_scenarios,
        "agent_policy": InterventionPolicy,
        "sensors": dict(
            rgb=(RGBCamera, 1600, 900),
            instance=(InstanceCamera, 1600, 900)
        ),
        "vehicle_config": dict(vehicle_model="static_default"),
        "height_scale": 0.1
    }
    env = ScenarioEnv(env_config)
    if args.model_path in ['random', 'always_stop', 'always_straight']:
        if args.model_path == "random":
            actor = random_action
        elif args.model_path == 'always_stop':
            actor = always_stop
        else:
            actor = always_straight
        total_collision, total_src_collision, total_rewards, total_completions, ADEs, FDEs, total_out_of_road, collided_scenarios, offroad_scenarios = \
            closed_loop_baselines(env, list(range(args.num_scenarios)), actor=actor, record_folder=args.record_path)
    else:
        total_collision, total_src_collision, total_rewards, total_completions, ADEs, FDEs, total_out_of_road, collided_scenarios, offroad_scenarios = \
            closed_loop(env, list(range(args.num_scenarios)), model_path=args.model_path, record_folder=args.record_path)

    summary = dict(
        model=args.model_path,
        src=traffic,
        num_scenarios=num_scenarios,
        total_collision=total_collision,
        total_out_of_road=total_out_of_road,
        total_src_collision=total_src_collision,
        total_rewards=total_rewards,
        total_completions=total_completions,
        avgADE=sum(ADEs) / len(ADEs),
        avgFDE=sum(FDEs) / len(FDEs),
        minADE=min(ADEs),
        minFDE=min(FDEs),
        ADEs=ADEs,
        FDEs=FDEs,
        action_statistics=ACTION_STATISTICS,
        collided_scenarios=collided_scenarios,
        offroad_scenarios=offroad_scenarios,
    )
    json.dump(summary, open(args.result_path, "w"), indent=2)


if __name__ == "__main__":
    main()
