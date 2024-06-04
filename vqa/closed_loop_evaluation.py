from metadrive.envs.scenario_env import ScenarioDiverseEnv
import argparse
from metadrive.scenario import utils as sd_utils
from vqa.online_eval import load_model, eval_model
from vqa.closed_loop_collision import Buffer, load_and_process_images
device = "cuda"


def convert_action(action, intervened):
    if intervened:
        ACTION_MAPPING = {
            "left": [1, 0.3],
            "right": [-1, 0.3],
            "stop": [0, -0.1],
            "none": [0, 0.3]
        }
    else:
        ACTION_MAPPING = {
            "left": [1, 0.3],
            "right": [-1, 0.3],
            "stop": [0, -0.1],
            "none": [0, 0]
        }
    if action in ACTION_MAPPING.keys():
        return ACTION_MAPPING[action]
    else:
        return [0, 0]


def preprocess_observation(buffer, vis_processors, text_processors):
    im = buffer.read_tensor(vis_processors) #5*6*3*364*364
    question = text_processors("You are the driver, what is the safest action to do? Anwser in left|right|stop|none.")
    im = torch.squeeze(im,0)
    question = [question]
    return {
        "vfeats": im,
        "questions": question
    }


def record_frame(env):
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
    return rgb_dict



def generation_action(buffer, model, vis_processors, text_processors, intervened):
    observation_dict = preprocess_observation(buffer, vis_processors, text_processors)
    answer = eval_model(model, observation_dict, vis_processors, text_processors)
    answer = answer.tolower()
    intervention = convert_action(answer, intervened)
    if not (intervention[0] == 0 and intervention[1] == 0):
        return intervention, True
    else:
        return intervention, False


def closed_loop(env: ScenarioDiverseEnv, seeds):
    model, vis_processors, text_processors = load_model()
    num_collisions = 0
    total_rewards = 0
    return
    for seed in seeds:
        env.reset(seed)
        print(f"Evaluating seed {env.engine.current_seed}")
        run = True
        episodic_reward = 0
        intervened = False
        buffer = Buffer
        while run:
            rgb_observation = record_frame(env)
            buffer.insert(rgb_observation)
            #intervention, intervened = generation_action(buffer, model, vis_processors, text_processors, intervened)
            intervention = [0,0]
            o, r, tm, tc, info = env.step(intervention)
            episodic_reward += info["reward"]
            if len(env.agent.collided_objects) > 0:
                num_collisions += 1
                run = False
            if tm or tc:
                if info["out_of_road"]:
                    continue
                run = False
        total_rewards += episodic_reward
        print(f"Finished seed {env.engine.current_seed}")
        print(f"episodic_reward: {episodic_reward}")
    return num_collisions, total_rewards


from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.replay_policy import InterventionPolicy

def main():
    use_render = False
    traffic = "/bigdata/yuxin/cat_reconstructed/test/subdir_2"
    scenario_summary, scenario_ids, scenario_files = sd_utils.read_dataset_summary(traffic)
    num_scenarios = len(scenario_summary)
    env_config = {
        "sequential_seed": True,
        "use_render": use_render,
        "data_directory": traffic,
        "num_scenarios": len(scenario_summary),
        "agent_policy": InterventionPolicy,
        "sensors": dict(
            rgb=(RGBCamera, 960, 540),
        ),
        "height_scale": 1
    }
    env = ScenarioDiverseEnv(env_config)
    total_collision, total_rewards = closed_loop(env, scenario_ids)
    #return total_collision, total_collision / num_scenarios, total_rewards, total_rewards / num_scenarios


if __name__ == "__main__":
    main()
