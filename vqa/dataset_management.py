import os
import json
import numpy as np


def delete_files_with_prefix(directory, prefix):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")


def merge_ba(folder, destination, base_name):
    folder_content = os.listdir(folder)
    print(folder_content)
    qas = [path for path in folder_content if path.startswith(base_name)]
    print(qas)
    merged_qa = {}
    idx = 0
    for qa in qas:
        try:
            with open(os.path.join(folder, qa), "r") as f:
                data = json.load(f)
        except Exception as e:
            raise e
        for _, content in data.items():
            merged_qa[idx] = content
            idx += 1
    try:
        with open(destination, "w") as f:
            json.dump(merged_qa, f, indent=4)
    except Exception as e:
        raise e


def splitting(qa, path, ps=[0.65, 0.15, 0.2]):
    def generate_categorical_list(n, categories, probabilities):
        return np.random.choice(categories, size=n, p=probabilities).tolist()
    with open(qa, "r") as file:
        data = json.load(file)
    split_file = {
        "src": qa,
        "train": [],
        "val": [],
        "test": []
    }
    num_points = len(data)
    mask = generate_categorical_list(num_points, 3, ps)
    counter = 0
    for key in data.keys():
        if mask[counter] == 0:
            split_file["train"].append(key)
        elif mask[counter] == 1:
            split_file["val"].append(key)
        else:
            split_file["test"].append(key)
        counter += 1
    try:
        with open(path, "w") as f:
            json.dump(split_file, fp=f, indent=4)
    except Exception as e:
        raise e


def dataset_statistics(qa, path):
    from collections import  Counter
    #Which types occur
    #Distribution of counting problems
    #Space distribution of the things referred(ego)
    #Question Types
    question_counter,num_counter,type_counter, pos_storage, bool_counter = Counter(),Counter(),Counter(), [], Counter()
    with open(qa, "r") as file:
        data = json.load(file)
    for data_point in data.values():
        question_counter[data_point["question_type"]] += 1
        for key, value in data_point["type_statistics"].items():
            type_counter[key] += value
        if data_point["question_type"] == "counting":
            num_counter[data_point["answer"][0]] += 1
        if data_point["question_type"] == "localization":
            pos_storage += data_point["answer"]
        if data_point["question_type"] == "count_equal_binary" or data_point["question_type"] == "count_more_binary":
            bool_counter[data_point["answer"]] += 1
    stats = {}
    stats["type_dist"] = type_counter
    stats["question_dist"] = question_counter
    stats["num_dist"] = num_counter
    stats["bool_dist"] = bool_counter
    stats["pos_dist"] = pos_storage
    try:
        with open(path, "w") as f:
            json.dump(stats, fp=f, indent=2)
    except Exception as e:
        raise e

if __name__ == '__main__':
    dataset_statistics("./multiprocess_1/merged.json","./multiprocess_1/merged_stats.json")
    #splitting("./multiprocess_1/merged.json", "./multiprocess_1/spllited_merge.json")
    #merge_ba("./multiprocess_1", "./multiprocess_1/merged.json", "qa")
    #delete_files_with_prefix("./verification", "highlighted")

