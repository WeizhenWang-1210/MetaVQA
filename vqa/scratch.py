# question_generator:

# all these questions are counterfactual questions.

# question that requires collision with ego to happen
# [t0, ..... tx, ....tend]

# 1. predict_collision
# Suggested now: tx-5
# Model observe: [t0,... , tx - 5], corresponding to 0.5 second before the collision happened
# Question: Is collision likely to happpen in the immediate future? If so, describe the object
# that we will collide with by specifying its type, color, and bounding box now.


# 3. move around
# Suggested now: tend
# Model observe: tend
# Question: Suppose at tx we are at <dx,dy> instead of <x,y>. Will there be collision?
# Answer: [yes|no]. Describe the thing that will collide into at tend's perspective

# 4. counterfactual trajectory
# Suggested now: tend
# Model observe: [t0, ....., tend]
# Question: Suppose the trajectory [t0, tend] is <x,y> * (tend-t0 + 1). Will we
# avoid the collision?
#

# don't need collision
# Suggested now: tend
# Model observe: [t0,....tend]
# Question: Suppose we stop all together at t_{i}. Will we run into collision?
# NOTE: i need to be fixed for all such question

import os


def find_json_files(root_folder):
    json_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".json") and "qa" in file and "safety" in file and "processed" in file:
                json_files.append(os.path.join(root, file))
    return json_files


from collections import defaultdict


def find_predict_collision_distro(qas):
    distro = defaultdict(lambda: 0)
    for id, record in qas.items():
        if record["question_type"] == "safety|predict_collision":
            if "Yes" in record["answer"]:
                distro["Yes"] += 1
            else:
                distro["No"] += 1
    return distro


def find_distro(qas):
    distro = defaultdict(lambda: defaultdict(lambda: 0))
    for id, record in qas.items():
        # if record["question_type"] == "safety|predict_collision":
        if "Yes" in record["answer"]:
            distro[record["question_type"]]["Yes"] += 1
        else:
            distro[record["question_type"]]["No"] += 1
    return distro


def find_processed_jsons(proceessed_folders):
    json_files = []
    for root_folder in proceessed_folders:
        for root, dirs, files in os.walk(root_folder):
            print(root)
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
    return json_files


def merge_split(json_paths, merged_path, ignore_types=None):
    if ignore_types is None:
        ignore_types = []
    result = {}
    count = 0
    for json_path in json_paths:
        qas = json.load(open(json_path, "r"))
        for info in qas.values():
            continue_flag = True
            for ignore_type in ignore_types:
                if ignore_type in info["question_type"]:
                    continue_flag = False
                    break
            if continue_flag:
                result[count] = info
                count += 1
        print(f"Loaded {count} qas from {json_path}")
    json.dump(result, open(merged_path, "w"), indent=2)


import json

if __name__ == "__main__":
    """train_processed_folders = [
        "training/multi_frame_processed", "training/single_frame_processed", "training/safety_critical_processed",
    ]
    train_processed_folders = [os.path.join("/bigdata/weizhen/metavqa_final/vqa", path) for path in
                               train_processed_folders]
    train_safety_folders = ["training/safety_critical_processed"]
    train_safety_folders = [os.path.join("/bigdata/weizhen/metavqa_final/vqa", path) for path in
                               train_safety_folders]
    train_jsons = find_processed_jsons(train_processed_folders)
    train_safety_jsons = find_processed_jsons(train_safety_folders)
    merge_split(train_safety_jsons,  "/bigdata/weizhen/metavqa_final/vqa/training/train_safety_predict_collision.json",
                ignore_types=["predict_collision"])"""

    """validation_processed_folders = [
        "validation/multi_frame_processed", "validation/single_frame_processed", "validation/safety_critical_processed",
    ]
    validation_processed_folders = [os.path.join("/bigdata/weizhen/metavqa_final/vqa", path) for path in
                               validation_processed_folders]
    validation_jsons = find_processed_jsons((validation_processed_folders))
    merge_split(validation_jsons, "/bigdata/weizhen/metavqa_final/vqa/validation/validation_all.json")"""

    test_processed_folders = [
        "testing/multi_frame_processed", "testing/single_frame_processed", "testing/safety_critical_processed",
    ]
    test_processed_folders = [os.path.join("/bigdata/weizhen/metavqa_final/vqa", path) for path in
                                    test_processed_folders]
    test_jsons = find_processed_jsons((test_processed_folders))
    merge_split(test_jsons, "/bigdata/weizhen/metavqa_final/vqa/testing/testing_all.json")












    #json.load(open("/bigdata/weizhen/metavqa_final/vqa/training/train_all.json","r"))
    #print(list(json.keys())[0])
    # print(len(jsons))

    """questions = dict()
    id = 0
    for file_id, json_path in enumerate(jsons):
        qas = json.load(open(json_path, "r"))
        for key, qa in qas.items():
            questions[id] = qa
            id += 1"""
    # print(find_predict_collision_distro(questions))
    # print(find_distro(questions))
