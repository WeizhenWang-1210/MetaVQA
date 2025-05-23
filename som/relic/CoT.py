import argparse
from metadrive.envs.scenario_env import ScenarioDiverseEnv
from metadrive.scenario import utils as sd_utils
import sys
from collections import defaultdict
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.replay_policy import InterventionPolicy
from som.closed_loop_utils import computeADE, computeFDE
from vqa.configs.NAMESPACE import MIN_OBSERVABLE_PIXEL, MAX_DETECT_DISTANCE
from vqa.dataset_utils import l2_distance
from vqa.annotation_utils import get_visible_object_ids
import numpy as np
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from som.navigation import get_trajectory, dynamic_get_navigation_signal
from som.parse_responses import parse_response_safe
import cv2
import os
from PIL import Image
from som.closed_loop_utils import classify_speed
import torch
from som.closed_loop_utils import CoT_prompts
from som.verify_close import described
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import json

INTERNVL = os.getenv("INTERNVL", False)
INTERNVLZEROSHOT = os.getenv("INTERNVLZEROSHOT", False)

print(INTERNVL, INTERNVLZEROSHOT)


def in_forbidden_area(agent):
    forbidden_places = ["CROSSWALK", "GROUND"]
    if len(set(forbidden_places).intersection(agent.contact_results)) > 0:
        return True
    else:
        return False


ACTION_STATISTICS = defaultdict(lambda: 0)
device = "cuda"
RECORD_BUFFER = defaultdict(lambda: defaultdict(lambda: dict()))
RECORD_FOLDER = "/home/weizhen/closed_loops"
MODELPATHS = (
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "Qwen/Qwen2-VL-7B-Instruct"
)
MODELPATH = None
sys.path.append('/home/chenda/lmms-finetune/chenda_scripts/')
sys.path.append('/home/chenda/internvl/internvl_chat/chenda_scripts/')
from inference_with_onevisn_finetuned import load_model, inference
from zero_shot import load_internvl, inference_internvl, inference_internvl_zeroshot, split_model, \
    batch_inference_internvl
from som.masking import find_center, put_text, put_rectangle


def observe(env):
    #obs = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, -15, 3], [0, -0.8, 0])
    position = (0., -2, 1.4)
    hpr = [0, 0, 0]
    obs = env.engine.get_sensor("rgb").perceive(to_float=False, new_parent_node=env.agent.origin, position=position,
                                                hpr=hpr)
    return obs


def observe_som(env, font_scale=1, bounding_box=True, background_color=(0, 0, 0)):
    def find_areas(img, colors, mode="RGB"):
        """
        Find the areas occupied by each color in <colors> in <img>. Default color indexing is "RGB"
        :param img: (H, W, C) numpy array
        :param colors: List of colors converted to 0-255 scale
        :param mode: if in anything other than "RGB", will swap channels here
        :return: areas(int) in list and corresponding boolean bitmasks(np.array) for each color.
        """
        flattened = img.reshape(-1, img.shape[2])
        unique_colors, counts = np.unique(flattened, axis=0, return_counts=True)
        unique_colors, counts = unique_colors.tolist(), counts.tolist()
        if mode == "BGR":
            unique_colors = [(r, g, b) for (b, g, r) in unique_colors]
        unique_colors = [(round(r, 5), round(g, 5), round(b, 5)) for r, g, b in unique_colors]
        color_mapping = {
            color: count for color, count in zip(unique_colors, counts)
        }
        results = []
        masks = []
        for color in colors:
            if color in color_mapping.keys():
                results.append(color_mapping[color])
                bitmask = np.zeros((img.shape[0], img.shape[1]))
                mask = np.all(img == color, axis=-1)
                bitmask = np.logical_or(bitmask, mask)
                masks.append(bitmask)
            else:
                results.append(0)
        return results, masks

    position = (0., -2, 1.4)
    hpr = [0, 0, 0]
    obs = env.engine.get_sensor("rgb").perceive(to_float=True, new_parent_node=env.agent.origin, position=position,
                                                hpr=hpr)
    instance = env.engine.get_sensor("instance").perceive(to_float=True, new_parent_node=env.agent.origin,
                                                          position=position, hpr=hpr)
    instance_5 = np.round(instance, 5)
    base_img = np.copy(obs)
    color2id = env.engine.c_id
    filter = lambda r, g, b, c: not (r == 1 and g == 1 and b == 1) and not (r == 0 and g == 0 and b == 0) and (
            c > MIN_OBSERVABLE_PIXEL)
    visible_object_ids, log_mapping = get_visible_object_ids(instance, color2id, filter)
    valid_objects = env.engine.get_objects(
        lambda x: l2_distance(x, env.agent) <= MAX_DETECT_DISTANCE and x.id != env.agent.id and not isinstance(x,
                                                                                                               BaseTrafficLight))
    query_ids = [object_id for object_id in valid_objects.keys() if object_id in visible_object_ids]
    colors = [log_mapping[query_id] for query_id in query_ids]
    colors = [(round(b, 5), round(g, 5), round(r, 5)) for r, g, b in colors]
    areas, binary_masks = find_areas(instance_5, colors)
    tuples = [(query_id, color, area, binary_mask)
              for query_id, color, area, binary_mask
              in zip(query_ids, colors, areas, binary_masks)]
    area_ascending = sorted(tuples, key=lambda x: x[2])
    center_list, contour_list = [], []
    text_boxes = np.zeros_like(instance)
    id2l = {
        query_id: idx for idx, query_id in enumerate(query_ids)
    }
    for i in range(len(area_ascending)):
        query_id, color, area, binary_mask = area_ascending[i]
        legal_mask = binary_mask
        for j in range(i):
            legal_mask = np.logical_and(legal_mask, ~area_ascending[j][3])
            occupied_text = np.all(text_boxes == [1., 1., 1.], axis=-1)
            legal_mask = np.logical_and(legal_mask, ~occupied_text)
        colored_mask = base_img.copy()
        center, contours = find_center(legal_mask)
        center_list.append(center)
        contour_list.append(contours)
        text_boxes = put_rectangle(text_boxes, str(id2l[query_id]), center, [1., 1., 1.], font_scale)
    #Put boxes/shapes/contours
    for i in range(len(area_ascending)):
        query_id, color, area, binary_mask = area_ascending[i]
        if bounding_box:
            contours, _ = cv2.findContours((binary_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            tlxs = []
            tlys = []
            brxs = []
            brys = []
            for contour in contours:
                # Get the tight bounding rectangle around the contour
                x, y, w, h = cv2.boundingRect(contour)
                tlxs.append(x)
                tlys.append(y)
                brxs.append(x + w)
                brys.append(y + h)
                # Draw the bounding box on the original mask or another image
                # For example, create a copy of the mask to visualize
            cv2.rectangle(base_img, (min(tlxs), min(tlys)), (max(brxs), max(brys)), color, 2)  # Draw green bounding box
        else:
            cv2.drawContours(base_img, contour_list[i], -1, color, 2)
    #Put the text
    for i in range(len(area_ascending)):
        query_id, color, area, binary_mask = area_ascending[i]
        put_text(base_img, str(id2l[query_id]), center_list[i], color=color, font_scale=font_scale,
                 background_color=background_color)
    return (base_img * 255).astype(np.uint8), id2l


def capture(env):
    obs = observe(env)
    front = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, -15, 3], [0, -0.8, 0])
    RECORD_BUFFER[env.current_seed][env.engine.episode_step] = dict(action=None, front=front, obs=obs,
                                                                    navigation=dynamic_get_navigation_signal(
                                                                        env.engine.data_manager.current_scenario,
                                                                        timestamp=env.episode_step, env=env),
                                                                    state=[env.agent.position, env.agent.heading,
                                                                           env.agent.speed])
    return obs, front


def capture_som(env):
    obs, id2label = observe_som(env)
    front = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, -15, 3], [0, -0.8, 0])
    RECORD_BUFFER[env.current_seed][env.engine.episode_step] = dict(action=None, front=front, obs=obs,
                                                                    navigation=dynamic_get_navigation_signal(
                                                                        env.engine.data_manager.current_scenario,
                                                                        timestamp=env.episode_step, env=env),
                                                                    state=[env.agent.position, env.agent.heading,
                                                                           env.agent.speed])
    return obs, front, id2label


INTERVENED = {
    "a": [0.5, 0.8],  #turn_left
    "b": [-0.5, 0.8],  #turn_right
    "c": [0, -0.135],  #slow_down
    "d": [0, -0.26],  #brake_now
    "e": [0, 0.15]  #keep_straight
}
NON_INTERVENED = {
    "a": [0.5, 0.8],  #turn_left
    "b": [-0.5, 0.8],  #turn_right
    "c": [0, -0.135],  #slow_down
    "d": [0, -0.26],  #brake_now
    "e": [0, 0.15]  #keep_straight
}

ACTION2OPTION = {
    "TURN_LEFT": "A", "TURN_RIGHT": "B", "SLOW_DOWN": "C", "BRAKE": "D", "KEEP_STRAIGHT": "E"
}


def convert_action(action, intervened):
    if intervened:
        ACTION_MAPPING = INTERVENED
    else:
        ACTION_MAPPING = NON_INTERVENED
    if action in ACTION_MAPPING.keys():
        return ACTION_MAPPING[action]
    else:
        return ACTION_MAPPING["e"]


def generation_action(model, processor, tokenizer, prompt, obs, intervened):
    if INTERNVL:
        if INTERNVLZEROSHOT:
            answer = inference_internvl_zeroshot(model, processor, tokenizer, prompt, obs[:, :, ::-1])
        else:
            answer = inference_internvl(model, processor, tokenizer, prompt, obs[:, :, ::-1])
    else:
        answer = inference(model, processor, tokenizer, prompt, obs[:, :, ::-1])
    print(prompt, answer)

    answer = parse_response_safe(answer, ACTION2OPTION)
    answer = answer.lower()
    ACTION_STATISTICS[answer] += 1
    intervention = convert_action(answer, intervened)
    print(answer, intervention)
    if not (intervention[0] == 0 and intervention[1] == 0):
        return intervention, True
    else:
        return intervention, False


def closed_loop(env: ScenarioDiverseEnv, seeds, model_path, record_folder=None):
    record = record_folder is not None
    if INTERNVL:
        print("Using Internvl")
        model, processor, tokenizer = load_internvl(model_path=model_path)
    else:
        model, processor, tokenizer = load_model(model_path=model_path)
    model.to("cuda")
    total_completions = total_rewards = num_collisions = num_src_collisions = 0
    command_frequency = 20
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
        while run:
            index = max(int(env.engine.episode_step), 0)
            if index >= max_step:
                print("Time horizon ended")
                run = False
                break
            trajectory.append(env.agent.position)
            navigation = dynamic_get_navigation_signal(env.engine.data_manager.current_scenario,
                                                       timestamp=env.episode_step, env=env)
            speed_class = classify_speed(env.agent.speed)
            criteria = {
                "slow": "(0-10 mph)",
                "moderate": "(10-30 mph)",
                "fast": "(30-50 mph)",
                "very fast": "(50+ mph)",
            }
            desc = criteria[speed_class]
            prompt = (
                f"Based on the image as the front observation, choose the safest from the following actions to execute for the next 2.0 seconds: "
                f"(A) TURN_LEFT; (B) TURN_RIGHT; (C) SLOW_DOWN; (D) BRAKE; (E) KEEP_STRAIGHT. "
                f"Answer in a single capitalized character chosen from [\"A\", \"B\", \"C\", \"D\", \"E\"].")
            obs, front, id2label = capture_som(env)
            if step % command_frequency == 0:
                print("Perceive & Act")

                qas = CoT_prompts(env, label2id={val: key for key, val in id2label.items()})
                keys = list(qas.keys())
                questions = [value["question"] for value in qas.values()]
                option_strs = [
                    str(list(record["option2answer"].keys())) for record in qas.values()
                ]
                questions = [
                    f"You are answering a multiple choice question, and you must select answer as a single capitalized letter from {option_str}.\nQuestion:\n{question}"
                    for option_str, question in zip(option_strs, questions)]
                observations = [obs[:, :, -1] for _ in range(len(questions))]  # convert to RGB
                responses = batch_inference_internvl(model, processor, tokenizer, keys, questions, observations,
                                                     batch_size=4)
                responses = {
                    key: parse_response_safe(value, {v: k for k, v in qas[key]["option2answer"].items()}) for key, value in
                    responses.items()
                }
                context = described(qas, responses)
                sys_str = f"You are driving on the road with {speed_class} speed{desc}, and your current navigation command is \"{navigation}\"."
                if context != "":
                    prompt = f"{sys_str}\n{context}\n{prompt}"

                intervention, intervened = generation_action(model, processor, tokenizer, prompt, obs, intervened)
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["action"] = intervention
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["navigation"] = navigation
                print(intervention, intervened)
            o, r, tm, tc, info = env.step(intervention)
            episodic_reward, episodic_completion = info["episode_reward"], info["route_completion"]
            if len(env.agent.crashed_objects) > 0:
                print("VLM still collided.")
                num_collisions += 1
                run = False
            if in_forbidden_area(env.agent):
                print("VLM wandered off road")
                num_collisions += 1
                run = False
            if (tm or tc) and not info["out_of_road"]:
                run = False
            step += 1
        total_rewards += episodic_reward
        total_completions += episodic_completion
        ADE, FDE = computeADE(original_trajectory, np.array(trajectory)), computeFDE(original_trajectory,
                                                                                     np.array(trajectory))
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
                    action_buffer[frame_id] = dict(action=frame["action"], state=frame["state"],
                                                   navigation=frame["navigation"])
                json.dump(action_buffer, open(os.path.join(folder_path, "action_buffer.json"), "w"))
            RECORD_BUFFER.clear()
        print(f"Finished seed {env.engine.current_seed}")
        print(f"episodic_reward: {episodic_reward}")
        print(f"episodic_completion:{episodic_completion}")
        print(f"ADE:{ADE}; FDE{FDE}")
    return num_collisions, num_src_collisions, total_rewards, total_completions, ADEs, FDEs


from metadrive.component.sensors.instance_camera import InstanceCamera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Set if don't use render")
    parser.add_argument("--num_scenarios", type=int, default=5, help="How many scenarios(from the start) to use")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=120)
    parser.add_argument("--data_directory", type=str, default="/home/weizhen/cat",
                        help="Path to the folder storing dataset_summary.pkl")
    parser.add_argument("--model_path", type=str, default="/home/chenda/ckpt/internvl_demo_merge",
                        help="Path to the model ckpt. Can be a name/local folder(if finetuned)")
    parser.add_argument("--prompt_schema", type=str, default="direct", help="Whether to use CoT prompting or not")
    parser.add_argument("--record_path", type=str, default=None,
                        help="Directory to store the visualizations of VLM's decision. If None, then won't store")
    parser.add_argument("--result_path", type=str, default="/home/weizhen/closed_loops/eval.json",
                        help="Path to the file storing experiment statistics")
    args = parser.parse_args()
    print("Running with the following parameters")
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))
    #assert INTERNVL or args.model_path in MODELPATHS, f"No implementation for model {args.model_path}"
    use_render = False if args.headless else True
    traffic = args.data_directory
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
            instance=(InstanceCamera, 1920, 1080)
        ),
        "vehicle_config": dict(vehicle_model="static_default"),
        "height_scale": 1
    }
    env = ScenarioDiverseEnv(env_config)


    jobs = list(range(args.start, args.end))

    total_collision, total_src_collision, total_rewards, total_completions, ADEs, FDEs = \
        closed_loop(env, jobs, model_path=args.model_path,
                    record_folder=args.record_path)
    summary = dict(
        src=traffic,
        num_scenarios=num_scenarios,
        total_collision=total_collision,
        total_src_collision=total_src_collision,
        total_rewards=total_rewards,
        total_completions=total_completions,
        avgADE=sum(ADEs) / len(ADEs),
        avgFDE=sum(FDEs) / len(FDEs),
        minADE=min(ADEs),
        minFDE=min(FDEs),
        ADEs=ADEs,
        FDEs=FDEs,
        action_statistics=ACTION_STATISTICS
    )
    json.dump(summary, open(args.result_path, "w"), indent=2)


if __name__ == "__main__":
    main()
