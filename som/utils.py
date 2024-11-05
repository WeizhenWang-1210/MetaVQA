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
    for key, record in qa_records.items():
        qa_records[key]["obs"] = obs
    return qa_records


import random


def random_choice(qa_records):
    for key, record in qa_records.items():
        valid_options = list(record["options"].keys())
        assert record["type"] == "describe_scenario" or not len(valid_options) <= 0
        if len(valid_options)==0:
            record["final_choice"] = ""
        else:
            record["final_choice"] = random.choice(valid_options)
    return qa_records


def merge_qas(qa_records):
    result = {}
    idx = 0
    for qa_record in qa_records:
        local_idx = 0
        for record in qa_record.values():
            result[idx + local_idx] = record
            local_idx += 1
        idx += local_idx
    return result


def accuracy_analysis(qa_records):
    statistics = dict()
    total_correct = total = 0
    for qid, record in qa_records.items():
        if record["type"] not in statistics.keys():
            statistics[record["type"]] = dict(
                total=0, correct=0
            )
        statistics[record["type"]]["total"] += 1
        statistics[record["type"]]["correct"] += 1 if record["final_choice"].upper() == "({})".format(
            record["answer"]).upper() else 0
    for type, stat in statistics.items():
        stat["accuracy"] = stat["correct"] / stat["total"]
        total += stat["total"]
        total_correct += stat["correct"]
    return statistics, total, total_correct


def analyze_dataset(qa_records):
    def find_sdc_file(frame_path):
        template = os.path.join(frame_path, "world**.json")
        result = glob.glob(template)[0]
        world_dict = json.load(open(result, "r"))
        if "world" in world_dict.keys():
            world = world_dict["world"]
        else:
            #pattern = r"^scene-\d{4}.*"
            episode_path = os.path.basename(os.path.dirname(frame_path))
            #match = re.match(pattern, episode_path)
            #print(match)
            world = episode_path[:10]  #match

        return world

    statistics = dict(
        total=0, question_dist=dict(), answer_dist=defaultdict(lambda: 0), total_frames=0, total_scenarios=0
    )
    frames, scenarios = set(), set()

    for qid, record in qa_records.items():
        frames.update(record["world"])
        scenarios.add(find_sdc_file(record["world"][-1]))
        statistics["total"] += 1
        if record["type"] not in statistics["question_dist"].keys():
            statistics["question_dist"][record["type"]] = dict(
                count=0, answer_dist=defaultdict(lambda: 0)
            )
        statistics["question_dist"][record["type"]]["count"] += 1
        statistics["question_dist"][record["type"]]["answer_dist"][record["answer"]] += 1
        statistics["answer_dist"][record["answer"]] += 1
    statistics["total_frames"] = len(frames)
    statistics["total_scenarios"] = len(scenarios)
    return statistics


def create_split(qa_records, split_path, distributions=(0.8, 0.2)):
    import numpy as np
    data = np.array(list(qa_records.keys()))
    # Example data
    # Step 1: Shuffle the data
    np.random.shuffle(data)
    # Step 2: Define proportions (e.g., 50%, 30%, 20%)
    split_proportions = distributions
    # Step 3: Convert proportions into lengths for each split
    split_indices = np.cumsum([int(len(data) * p) for p in split_proportions])
    # Step 4: Split the shuffled data
    split_data = np.split(data, split_indices[:-1])
    # Print each split
    for i, subset in enumerate(split_data):
        print(f"Subset {i + 1}: {len(subset)} items")
    json.dump(
        {
            "train": list(split_data[0]),
            "val": list(split_data[1])
        },
        open(split_path, "w"),
    )


import glob

import shutil
import os
from concurrent.futures import ThreadPoolExecutor


def copy_file(src, dest):
    try:
        shutil.copy2(src, dest)  # copy2 also preserves metadata like timestamps
        print(f"Copied {src} to {dest}")
    except Exception as e:
        print(f"Error copying {src} to {dest}: {e}")


# Function to perform parallel file copying
def parallel_copy(mappings, num_threads=4):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create a list of tasks for copying files
        tasks = [executor.submit(copy_file, src_path, dest_path) for src_path, dest_path in
                 mappings]
        # Optionally, wait for all tasks to complete and handle results
        for task in tasks:
            task.result()


def export(qa_path, obs_directory, vqa_directory):
    import tqdm
    qa_records = json.load(open(qa_path, "r"))
    obs_old2new = dict()
    token_name = 0
    transfer_tuples = []
    #new_records = copy.deepcopy(qa_records)
    for qid, record in tqdm.tqdm(qa_records.items(), desc="Refactoring obs", unit="vqa"):
        old_ob_paths = record["obs"]
        for old_ob_path in old_ob_paths:
            if old_ob_path not in obs_old2new.keys():
                new_ob_path = os.path.join(obs_directory, f"{token_name}.png")
                #shutil.copy2(old_ob_path, new_ob_path)
                relative_new_ob_path = os.path.relpath(new_ob_path, vqa_directory)
                transfer_tuples.append((old_ob_path, new_ob_path))
                obs_old2new[old_ob_path] = relative_new_ob_path
                token_name += 1
        record["obs"] = [obs_old2new[path] for path in old_ob_paths]
    json.dump(obs_old2new, open(os.path.join(obs_directory, "old2new.json"), "w"))
    json.dump(qa_records, open(os.path.join(vqa_directory, "data.json"), "w"))
    parallel_copy(transfer_tuples)


def split(path, split_path, train_path, val_path):
    import json, os
    #path = "/data_weizhen/metavqa_cvpr/static_medium_export/data.json"
    base_dir = os.path.dirname(path)
    #split_path = "/data_weizhen/metavqa_cvpr/vqa_merged/static_medium_split.json"
    qas = json.load(open(path, "r"))
    split = json.load(open(split_path, "r"))
    print(len(split["train"]))
    print(len(split["val"]))
    #train_path = "/data_weizhen/metavqa_cvpr/static_medium_export/train.json"
    #val_path = "/data_weizhen/metavqa_cvpr/static_medium_export/val.json"
    train_qas, val_qas = dict(), dict()
    local_idx = 0

    def append_prefix(paths, prefix):
        return [os.path.join(prefix, p) for p in paths]

    for idx in split["train"]:
        train_qas[local_idx] = qas[idx]
        train_qas[local_idx]["obs"] = append_prefix(qas[idx]["obs"], base_dir)
        local_idx += 1
    local_idx = 0
    for idx in split["val"]:
        val_qas[local_idx] = qas[idx]
        val_qas[local_idx]["obs"] = append_prefix(qas[idx]["obs"], base_dir)
        local_idx += 1
    #print(train_qas[0]["obs"])
    json.dump(train_qas, open(train_path, "w"), indent=2)
    json.dump(val_qas, open(val_path, "w"), indent=2)


if __name__ == "__main__":
    test_qas = json.load(open("/data_weizhen/metavqa_cvpr/datasets/test/test/test_processed.json"))
    test_qas = random_choice(test_qas)
    json.dump(
        test_qas,
        open("/home/weizhen/experiments/main/random_test_results.json", "w"), indent=2
    )
