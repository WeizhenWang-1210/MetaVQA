import PIL.Image

from metadrive.envs.scenario_env import ScenarioDiverseEnv
import sys
from collections import defaultdict
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.replay_policy import InterventionPolicy
from som.embodied_utils import ACTION
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from metadrive.component.sensors.instance_camera import InstanceCamera
import json

sys.path.append('/home/chenda/lmms-finetune/chenda_scripts/')
sys.path.append('/home/chenda/internvl/internvl_chat/chenda_scripts/')
from inference_with_onevisn_finetuned import load_model, inference
from zero_shot import load_internvl, inference_internvl, inference_internvl_zeroshot, split_model
import random
from som.closed_loop_evaluations import observe_som
from som.closed_loop_utils import CoT_prompts


def parse(key):
    parsed = key.split("_")
    o_id = parsed[0]
    qtype = parsed[1]
    action = None
    if len(parsed) > 2:
        action = parsed[2]
    return o_id, qtype, action


def described(records, responses):
    """
    Assuming response in A,B,C,D,E ..
    records, responses have the same key, only that responses in uppercase character
    """

    new_dict = defaultdict(lambda: dict())
    selected_ids = []
    for key, value in responses.items():
        oid, qtype, action = parse(key)
        opt2answer = records[key]["option2answer"]
        print(key, value)
        answer = opt2answer[value]
        if qtype == "distance" and answer in ["Very close(0-2m)", "Close(2-10m)", "Medium(10-30m)"]:
            selected_ids.append(oid)

        if action is None:
            new_dict[oid][qtype] = answer
        else:
            new_dict[oid][f"{qtype}_{action}"] = answer
    descs = []
    collide_with_ego = defaultdict(lambda: [])
    for idx in selected_ids:
        if idx == "ego":
            continue
        distance_str = new_dict[idx]["distance"]
        position_str = new_dict[idx]["position"]
        heading_str = ""
        if "heading" in new_dict[idx].keys():
            heading_str = new_dict[idx]["heading"]
        collide_str = ""
        if "collision" in new_dict[idx].keys():
            collide_str = "will collide" if new_dict[idx]["collision"] == "Yes" else "will not collide"

        if heading_str != "" and collide_str != "":
            string = f"\tObject <{idx}> is positioned at our {position_str} sector at {distance_str} distance, heading the {heading_str}. If it proceeds, it {collide_str} into us if we stay still."
        elif heading_str != "" and collide_str == "":
            string = f"\tObject <{idx}> is positioned at our {position_str} sector at {distance_str} distance, heading the {heading_str}."
        elif heading_str == "" and collide_str != "":
            string = f"\tObject <{idx}> is positioned at our {position_str} sector at {distance_str} distance. If it proceeds, it {collide_str} into us if we stay still."
        else:
            string = f"\tObject <{idx}> is positioned at our {position_str} sector at {distance_str} distance."
        descs.append(string)
        for action_idx in range(5):
            ego_collide_str = new_dict[idx][f"ego-collision_{str(action_idx)}"]
            if ego_collide_str == "Yes":
                collide_with_ego[action_idx].append(idx)
    ego_desc = []
    for action_idx in range(5):
        obs = [f"object <{idx}>" for idx in collide_with_ego[action_idx]]
        action_str = ACTION.get_action(action_idx)
        position_str = new_dict["ego"][f"position_{str(action_idx)}"]
        distance_str = new_dict["ego"][f"distance_{str(action_idx)}"]

        s = ", ".join(obs)
        string = f"\tIf we choose {action_str}, we will be end up in our {position_str} sector at {distance_str} distance. If all other objects remain still, we will collide with [{s}]."
        ego_desc.append(string)
    obj_desc = "\n".join(descs)
    ego_desc = "\n".join(ego_desc)
    context_str = f"Here's a summary of the surrounding objects:\n{{\n{obj_desc}\n}}"
    ego_str = f"This is what will happen if we choose each action:\n{{\n{ego_desc}\n}}"
    return f"{context_str}\n{ego_str}"

    #if dist in very close, close,

    #obj a is located in <> sector, heading toward <>, and will not collide into us if we stay still

    #if we go action A, we will be ing <dir, dist>, and we will collide with [x] if it stay stills


if __name__ == "__main__":

    qas = json.load(open("some.json", "r"))
    responses = {key: random.choice(["A","B"]) for key in qas.keys()}
    context_str = described(qas, responses)
    print(context_str)
    exit()

    try:
        env = ScenarioDiverseEnv(
            {
                "sequential_seed": True,
                "use_render": False,
                "data_directory": "/data_weizhen/scenarios",
                "num_scenarios": 120,
                "agent_policy": InterventionPolicy,
                "sensors": dict(
                    rgb=(RGBCamera, 1920, 1080),
                    instance=(InstanceCamera, 1920, 1080)
                ),
            }
        )
        o, _ = env.reset()
        for seed in range(120):
            o, _ = env.reset(seed)
            run = True
            while run:
                o, r, tm, tc, info = env.step([0, 0])
                im_array, id2label = observe_som(env)

                qas = CoT_prompts(env, {val: key for key, val in id2label.items()})

                #responses = {key: "A" for key in qas.keys()}

                #context_str = described(qas, responses)

                json.dump(qas, open("some.json", "w"), indent=1)
                PIL.Image.fromarray(im_array[:, :, ::-1]).save("obs.jpg")
                exit()

                PIL.Image.fromarray(im_array[:, :, ::-1]).save("obs.jpg")
                inference_internvl(model, processor, tokenizer, "What's in this picture", im_array[:, :, ::-1])
                print(im_array.shape)
                if tm or tc:
                    run = False
    finally:
        env.close()
