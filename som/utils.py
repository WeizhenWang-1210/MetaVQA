import json
import os.path
from collections import defaultdict



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
