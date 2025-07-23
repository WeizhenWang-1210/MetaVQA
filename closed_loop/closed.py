import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import InterventionPolicy
from som.closed_loop_utils import absoluteFDE
from som.closed_loop_utils import computeADE
from som.embodied_utils import ACTION, classify_distance
from som.navigation import get_trajectory, dynamic_get_navigation_signal, dest_navigation_signal
from som.parse_responses import parse_response
from vqa.configs.namespace import MIN_OBSERVABLE_PIXEL, MAX_DETECT_DISTANCE
from vqa.scenegen.annotation_utils import get_visible_object_ids
from vqa.vqagen.dataset_utils import l2_distance
from vqa.vqagen.static_question_generation import POSITION2CHOICE

INTERNVL = os.getenv("INTERNVL", False)
INTERNVLZEROSHOT = os.getenv("INTERNVLZEROSHOT", False)

print(INTERNVL, INTERNVLZEROSHOT)


def divide_list(lst, n):
    # Calculate the size of each chunk
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    for i in range(n):
        # Each chunk gets an extra element if there is a remainder
        end = start + chunk_size + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result


def in_forbidden_area(agent):
    """forbidden_places = ["GROUND"]
    if len(set(forbidden_places).intersection(agent.contact_results)) > 0:
        return True
    else:
        return False"""
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
from zero_shot import load_internvl, inference_internvl, inference_internvl_zeroshot
from vqa.vqagen.set_of_marks import find_center, put_text, put_rectangle
from som.closed_loop_utils import classify_speed


def observe(env):
    # obs = env.engine.get_sensor("rgb").perceive(False, env.agent.origin, [0, -15, 3], [0, -0.8, 0])
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
    # Put boxes/shapes/contours
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
    # Put the text
    for i in range(len(area_ascending)):
        query_id, color, area, binary_mask = area_ascending[i]
        put_text(base_img, str(id2l[query_id]), center_list[i], color=color, font_scale=font_scale,
                 bg_color=background_color)
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
    "a": [0.15, 0.8],  # turn_left
    "b": [-0.15, 0.8],  # turn_right
    "c": [0, -0.135],  # slow_down
    "d": [0, -0.26],  # brake_now
    "e": [0, 0.15],  # keep_straight
    "f": [0, 0.3],  # speed_up
    "g": [0.6, 0.2],  # big_left
    "h": [-0.6, 0.2]  # big_right
}
NON_INTERVENED = INTERVENED

ACTION2OPTION = {
    "TURN_LEFT": "A", "TURN_RIGHT": "B", "SLOW_DOWN": "C", "BRAKE": "D", "KEEP_STRAIGHT": "E",
    "SPEED_UP": "F", "BIG_LEFT": "G", "BIG_RIGHT": "H"
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
    print(f"{obs.mean()}, {obs.std()}")
    if INTERNVL:
        if INTERNVLZEROSHOT:
            raise ValueError
            answer = inference_internvl_zeroshot(model, processor, tokenizer, prompt, obs[:, :, ::-1])
        else:
            answer = inference_internvl(model, processor, tokenizer, prompt, obs[:, :, ::-1])
    else:
        answer = inference(model, processor, tokenizer, prompt, obs[:, :, ::-1])
    # print(prompt, answer)
    answer = parse_response(answer, ACTION2OPTION)
    answer = answer.lower()
    ACTION_STATISTICS[answer] += 1
    intervention = convert_action(answer, intervened)
    print(answer, intervention)
    if not (intervention[0] == 0 and intervention[1] == 0):
        return intervention, True
    else:
        return intervention, False


def prepare_prompt(speed, navigation):
    speed_class = classify_speed(speed)
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
    return question


def prepare_prompt_dest(speed, dir, dist):
    sector_string = POSITION2CHOICE[dir]
    distance = classify_distance(dist)
    speed_class = classify_speed(speed)
    criteria = {
        "slow": "(0-10 mph)",
        "moderate": "(10-30 mph)",
        "fast": "(30-50 mph)",
        "very fast": "(50+ mph)",
    }
    dist_creteria = {
        "very close": "(0-2 m)",
        "close": "(2-10 m)",
        "medium": "(10-30 m)",
        "far": "(30+ m)"
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
        f"You are the driver of a vehicle on the road, and your final destination is at {distance} distance{dist_creteria[distance]} to your {sector_string} at this moment. "
        f"Currently, you are driving with {speed_class} speed{desc}. The image is observed in front of you. "
        f"In order to reach your final destination, carefully examine the image. Choose the best action to execute for the next 0.5 seconds from the following options:\n"
        f"(A) TURN_LEFT, {explanation[ACTION.get_action(ACTION.TURN_LEFT)]}.\n"
        f"(B) TURN_RIGHT, {explanation[ACTION.get_action(ACTION.TURN_RIGHT)]}.\n"
        f"(C) SLOW_DOWN, {explanation[ACTION.get_action(ACTION.SLOW_DOWN)]}.\n"
        f"(D) BRAKE, {explanation[ACTION.get_action(ACTION.BRAKE)]}.\n"
        f"(E) KEEP_STRAIGHT, {explanation[ACTION.get_action(ACTION.KEEP_STRAIGHT)]}.\n"
        f"(F) SPEED_UP, {explanation[ACTION.get_action(ACTION.SPEED_UP)]}.\n"
        f"(G) BIG_LEFT, {explanation[ACTION.get_action(ACTION.BIG_LEFT)]}.\n"
        f"(H) BIG_RIGHT, {explanation[ACTION.get_action(ACTION.BIG_RIGHT)]}.\n"
        f"Safety and swiftness to the final destination are both important factors to consider."
        f"Answer in a single capitalized character chosen from [\"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\"].")
    return question


def closed_loop(env: ScenarioEnv, seeds, model_path, record_folder=None):
    record = record_folder is not None
    if INTERNVL:
        print("Using Internvl")
        model, processor, tokenizer = load_internvl(model_path=model_path)
    else:
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
        while run:
            index = max(int(env.engine.episode_step), 0)
            if index >= max_step:
                print("Time horizon ended")
                run = False
                break
            trajectory.append(env.agent.position)
            if step % command_frequency == 0:
                print("Perceive & Act")
                ###Changed heree###Changed heree
                dir, dist = dest_navigation_signal(env.engine.data_manager.current_scenario,
                                                   timestamp=env.episode_step, env=env)
                prompt = prepare_prompt_dest(env.agent.speed, dir, dist)
                ###Changed heree###Changed heree
                obs, front, id2label = capture_som(env)
                # Image.fromarray(obs[:, :, ::-1]).save(
                #    f"{record_folder}/{seed}_{env.engine.episode_step}.png")
                intervention, intervened = generation_action(model, processor, tokenizer, prompt, obs, intervened)
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["action"] = intervention
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["navigation"] = (dir, dist)
                RECORD_BUFFER[env.engine.current_seed][env.engine.episode_step]["prompt"] = prompt
                # print(intervention, intervened)
            assert not (intervention[0] == 0 and intervention[1] == 0)
            o, r, tm, tc, info = env.step(intervention)
            episodic_reward, episodic_completion = info["episode_reward"], info["route_completion"]
            if len(env.agent.crashed_objects) > 0:
                print("VLM collided.")
                collided_scenarios.add(env.engine.data_manager.current_scenario_file_name)
                collide = 1
                # run = False
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
        ADE, FDE = computeADE(original_trajectory, np.array(trajectory)), absoluteFDE(original_trajectory,
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
    return num_collisions, num_src_collisions, total_rewards, total_completions, ADEs, FDEs, total_out_of_road, list(
        collided_scenarios), list(offroad_scenarios)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Set if don't use render")
    parser.add_argument("--num_scenarios", type=int, default=5, help="How many scenarios(from the start) to use")
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
    # assert INTERNVL or args.model_path in MODELPATHS, f"No implementation for model {args.model_path}"
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
        # "out_of_road_done": True,
        "vehicle_config": dict(vehicle_model="static_default"),
        "height_scale": 1
    }
    env = ScenarioEnv(env_config)
    if args.model_path == "random":
        exit()
    elif args.model_path == "always_stop":
        exit()
    elif args.model_path == "always_straight":
        exit()
    else:
        total_collision, total_src_collision, total_rewards, total_completions, ADEs, FDEs, total_out_of_road, collided_scenarios, offroad_scenarios = \
            closed_loop(env, list(range(args.num_scenarios)), model_path=args.model_path,
                        record_folder=args.record_path)
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
