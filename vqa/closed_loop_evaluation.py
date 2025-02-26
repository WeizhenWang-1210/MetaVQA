from metadrive.envs.scenario_env import ScenarioDiverseEnv
from metadrive.scenario import utils as sd_utils
from vqa.online_eval import load_model, eval_model
from vqa.closed_loop_collision import Buffer, load_and_process_images, vector_transform
import torch
from metadrive.scenario.parse_object_state import parse_object_state

device = "cuda"
from collections import defaultdict

ACTION_STATISTICS = defaultdict(lambda: 0)


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


def preprocess_observation(destination, buffer, vis_processors, text_processors):
    im, displacement, now_world_position = buffer.read_tensor(vis_processors)  #5*6*3*364*364
    now_heading = buffer.dq[-1][2]
    destination = vector_transform(now_world_position, now_heading, destination)
    #displacement = round(displacement[0], 0), round(displacement[1], 0)
    #destination = round(destination[0], 0), round(destination[1], 0)
    destination = int(destination[0]), int(destination[1])
    #print(destination)
    question = text_processors(
        "You are the driver, what is the safest action to do? Choose from one option from: (A) left ;(B) right ;(C) stop ;(D) none. For example, if you want to turn left, answer \"A\" ")
    im = torch.unsqueeze(im, 0)
    question = question
    return {
        "vfeats": im,
        "questions": [question],
        "answers": ["something"]
    }


def record_frame(env: ScenarioDiverseEnv):
    engine = env.engine
    camera = engine.get_sensor("rgb")
    positions = [(0., 0.0, 1.5), (0., 0., 1.5), (0., 0., 1.5), (0., 0, 1.5), (0., 0., 1.5),
                 (0., 0., 1.5)]
    hprs = [[0, 0, 0], [45, 0, 0], [135, 0, 0], [180, 0, 0], [225, 0, 0], [315, 0, 0]]
    names = ["front", "leftf", "leftb", "back", "rightb", "rightf"]
    rgb_dict = {}
    for position, hpr, name in zip(positions, hprs, names):
        rgb = camera.perceive(to_float=False, new_parent_node=env.agent.origin, position=position, hpr=hpr)
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


def generation_action(destination, buffer, model, vis_processors, text_processors, intervened):
    observation_dict = preprocess_observation(destination, buffer, vis_processors, text_processors)
    #print(observation_dict["questions"])
    answer = eval_model(model, observation_dict, vis_processors, text_processors)
    #answer = answer[0].lower()
    answer = "stop"
    ACTION_STATISTICS[answer] += 1
    #print(answer)
    intervention = convert_action(answer, intervened)
    if not (intervention[0] == 0 and intervention[1] == 0):
        return intervention, True
    else:
        return intervention, False


def closed_loop(env: ScenarioDiverseEnv, seeds):
    model, vis_processors, text_processors = load_model(False)
    model.to(device)
    num_src_collisions = 0
    num_collisions = 0
    total_rewards = 0
    command_frequency = 8
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
        buffer = Buffer(size=5)
        step = 1
        intervention = [0, 0]
        observation_frequency = 4
        while run:
            index = max(int(env.engine.episode_step), 0)
            if index >= max_step:
                run = False
            if step % observation_frequency == 0:
                rgb_observation, current_position, current_heading = record_frame(env)
                buffer.insert((rgb_observation, current_position, current_heading))
            if step % command_frequency == 0:
                intervention, intervened = generation_action(destination, buffer, model, vis_processors,
                                                             text_processors, intervened)
            o, r, tm, tc, info = env.step(intervention)
            episodic_reward = info["episode_reward"]
            if len(env.agent.crashed_objects) > 0:
                print("VLM still collided.")
                num_collisions += 1
                run = False
            if (tm or tc) and not info["out_of_road"]:
                run = False
            step += 1
        total_rewards += episodic_reward
        print(f"Finished seed {env.engine.current_seed}")
        print(f"episodic_reward: {episodic_reward}")

    return num_collisions, num_src_collisions, total_rewards


from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.replay_policy import InterventionPolicy
from metadrive.envs.scenario_env import ScenarioEnv

def main():
    use_render = False
    traffic = "/bigdata/yuxin/cat_reconstructed/test/subdir_2"
    scenario_summary, scenario_ids, scenario_files = sd_utils.read_dataset_summary(traffic)
    num_scenarios = len(scenario_summary)
    env_config = {
        "sequential_seed": True,
        "use_render": use_render,
        "data_directory": traffic,
        "num_scenarios": num_scenarios,
        "agent_policy": InterventionPolicy,
        "sensors": dict(
            rgb=(RGBCamera, 1920, 1080),
        ),
        "height_scale": 1
    }
    env = ScenarioEnv(env_config)
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
    #json.dump(summary, open("./online_eval_baseline.json", "w"), indent=2)
    json.dump(summary, open("./online_stop_baseline.json", "w"), indent=2)


if __name__ == "__main__":
    main()
