import copy
import json
import os.path
from collections import defaultdict

import numpy as np


def get(world, target_id):
    for obj in world["objects"]:
        if obj["id"] == target_id:
            return obj
    if world["ego"]["id"] == target_id:
        return world["ego"]
    print("No object of id {} is found! Something is wrong.".format(world))
    return None


def fill_in_label(template_str: str, replacement: dict):
    for placeholder, value in replacement.items():
        template_str = template_str.replace(placeholder, value)
    return template_str


def enumerate_frame_labels(frame_path: str, perspective: str = "front", id2label_path: str = None):
    #assuming object_id : label has been created in  in the episode folder.
    episode_path = os.path.dirname(frame_path)
    identifier = os.path.basename(frame_path)
    if not id2label_path:
        id2label_path = os.path.join(episode_path, "id2label_{}.json".format(perspective))
    world_path = os.path.join(frame_path, "world_{}.json".format(identifier))
    id2label = json.load(open(id2label_path, "r"))
    world = json.load(open(world_path, "r"))
    results = {}
    invalid_id = []
    for obj in world["objects"]:
        if obj["id"] in id2label.keys() and perspective in obj["observing_camera"]:
            results[id2label[obj["id"]]] = obj["id"]
        else:
            invalid_id.append(obj["id"])
    results[-1] = world["ego"]["id"]
    return results, invalid_id


def enumerate_episode_labels(episode_path: str, perspective: str = "front"):
    id2label_path = os.path.join(episode_path, "id2label_{}.json".format(perspective))
    id2label = json.load(open(id2label_path, "r"))
    results = defaultdict(lambda: -1)
    for key, value in id2label.items():
        results[value] = key
    return results

import re
def parse_response(response):
    valid_choices = ["(A)", "(B)", "(C)", "(D)", "(a)", "(b)", "(c)", "(d)", " A", " B", ""]
    for valid_choice in valid_choices:
        if valid_choice in response:
            return valid_choice

    valid_answers = ["A", "B", "C", "D", "a", "b", "c", "d"]
    for valid_answer in valid_answers:
        if valid_answer == response:
            return f"({valid_answer})"
    return " "

def create_black_obs(width=1920, height=1080):
    location = "/bigdata/weizhen/repo/qa_platform/public/black.png"
    from PIL import Image
    black_image = Image.new("RGB", (width, height), (0, 0, 0))
    black_image.save(location)

def replace_obs(qa_records, obs):
    for key,record in qa_records.items():
        qa_records[key]["obs"]=obs
    return qa_records
import random
def random_choice(qa_records):
    options = ["(A)","(B)","(C)","(D)"]
    for key, record in qa_records.items():
        if record["type"] in ["pick_closer"]:
            valid_options = options[:3]
        elif record["type"] in ["predict_crash_ego_still", "predict_crash_ego_dynamic","relative_predict_crash_still","relative_predict_crash_dynamic"]:
            valid_options = options[:2]
        else:
            valid_options = options
        record["final_choice"] = random.choice(valid_options)
    return qa_records

if __name__ == "__main__":
    #old_qa = json.load(open("/bigdata/weizhen/repo/qa_platform/public/grounding.json", "r"))
    #new_qa = replace_obs(old_qa, ["/bigdata/weizhen/repo/qa_platform/public/black.png"])
    #new_qa = random_choice(old_qa)
    #json.dump(
    #    new_qa, open("/bigdata/weizhen/repo/qa_platform/public/ground_result_random_parsed.json", "w"), indent=2
    #)



    #exit()

    response_path = "/bigdata/weizhen/repo/qa_platform/public/ground_result_all_black.json"
    import json

    responses = json.load(open(response_path, "r"))
    for qid in responses.keys():
        choice = parse_response(responses[qid]["model_response"])
        responses[qid]["final_choice"] = choice
    json.dump(
        responses, open("/bigdata/weizhen/repo/qa_platform/public/ground_result_all_black_parsed.json", "w"), indent=2
    )



