import json
import os
import re
from collections import defaultdict

import numpy as np


def replace_substrs(original_string, mapping):
    # Define the pattern to match <n> where n is an integer
    pattern = r'<(\d+)>'

    # Function to use for substitution
    def replacer(match):
        # Extract the number n from <n>
        n = int(match.group(1))
        # Check if n is in the mapping and replace with its mapped value
        return f"<{str(mapping.get(n, match.group(0)))}>"

    # Substitute using re.sub with the replacer function
    replaced_string = re.sub(pattern, replacer, original_string)

    return replaced_string


def create_options(present_values, num_options, answer, namespace, transform=None):
    if len(present_values) < num_options:
        space = set(namespace)
        result = set(present_values)
        diff = space.difference(result)
        choice = np.random.choice(np.array(list(diff)), size=num_options - len(present_values), replace=False)
        result = list(result) + list(choice)
    elif len(present_values) == num_options:
        result = present_values
    else:
        answer = {answer}
        space = set(present_values)
        diff = space.difference(answer)
        choice = np.random.choice(np.array(list(diff)), size=num_options - 1, replace=False)
        result = list(answer) + list(choice)
    if transform:
        if callable(transform):
            result = [transform(o) for o in result]
        elif isinstance(transform, dict):
            result = [transform[o] for o in result]
    #paired_list = list(enumerate(result))
    np.random.shuffle(result)
    #shuffled_list = [element for _, element in paired_list]
    return result  #, {answer: index for index, answer in paired_list}


def create_multiple_choice(options):
    assert len(options) < 26, "no enough alphabetic character"
    result = []
    answer_to_choice = {}
    for idx, option in enumerate(options):
        label = chr(idx + 64 + 1)
        result.append(
            "({}) {}".format(label, option)
        )
        answer_to_choice[option] = label
    return "; ".join(result) + ".", answer_to_choice


def fill_in_label(template_str: str, replacement: dict):
    for placeholder, value in replacement.items():
        template_str = template_str.replace(placeholder, value)
    return template_str

def get_from_world(world, target_id):
    """
    world is the dictionary object you loaded using world_*.json
    """
    for obj in world["objects"]:
        if obj["id"] == target_id:
            return obj
    if world["ego"]["id"] == target_id:
        return world["ego"]
    print("No object of id {} is found! Something is wrong.".format(world))
    return None


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


def angle2sector(degree):
    assert 0<=degree<=360
    if degree < 15 or degree > 345:
        return "f"
    elif 15<degree<75:
        return "rf"
    elif 75<degree<105:
        return "r"
    elif 105<degree<165:
        return "rb"
    elif 165<degree<195:
        return "b"
    elif 195<degree<255:
        return "lb"
    elif 255<degree<285:
        return "l"
    elif 285<degree<345:
        return "lf"
