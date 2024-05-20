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


def splitting(qa, path, ps=[0.75, 0.15, 0.1]):
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
    from collections import Counter
    # Which types occur
    # Distribution of counting problems
    # Space distribution of the things referred(ego)
    # Question Types
    question_counter, num_counter, type_counter, pos_storage, bool_counter = Counter(), Counter(), Counter(), [], Counter()
    with open(qa, "r") as file:
        data = json.load(file)
    for data_point in data.values():
        question_counter[data_point["question_type"]] += 1
        for key, value in data_point["type_statistics"].items():
            type_counter[key] += value
        if data_point["question_type"] == "counting":
            num_counter[data_point["answer"][0]] += 1
        pos_storage += data_point["pos_statistics"]
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


def visualize_pos(points, vis_path):
    pass


def export_dataset(qa_path, data_directory):
    import shutil
    with open(qa_path, "r") as file:
        qa_pairs = json.load(file)

    export_qas = qa_pairs
    for id, metainfo in qa_pairs.items():
        new_data_path = os.path.join(data_directory, id)
        os.makedirs(new_data_path, exist_ok=True)
        for angle, frames in metainfo["rgb"].items():
            frame_id = 0
            new_frames = []
            for frame in frames:
                folder = os.path.join(new_data_path, "rgb", angle)
                os.makedirs(folder, exist_ok=True)
                new_path = os.path.join(new_data_path, "rgb", angle, f"{frame_id}.png")
                shutil.copy2(frame, new_path)
                frame_id += 1
                new_frames.append(new_path)
            metainfo["rgb"][angle] = new_frames
        new_lidar_path = os.path.join(new_data_path, "lidar.pkl")
        shutil.copy2(metainfo["lidar"], new_lidar_path)
        metainfo["lidar"] = new_lidar_path
    export_path = os.path.join(data_directory, "exported.json")
    with open(export_path, "w") as file:
        json.dump(export_qas, file, indent=2)


def export_dataset_2(qa_path, src_data_directory, target_data_directory):
    import shutil

    with open(qa_path, "r") as file:
        qa_pairs = json.load(file)
    paths_to_keep = set()
    for id, metainfo in qa_pairs.items():
        for angle, frames in metainfo['rgb'].items():
            new_frames = []
            for frame in frames:
                path = "/".join(frame.split("/")[5:])
                paths_to_keep.add(path)
                # print(os.path.join(target_data_directory,path))
                # break
                new_frames.append(os.path.join("./", path))
            qa_pairs[id]["rgb"][angle] = new_frames
        lidar_path = "/".join(metainfo["lidar"].split("/")[5:])
        # print(os.path.join(target_data_directory,lidar_path))
        # exit()
        paths_to_keep.add(lidar_path)
        qa_pairs[id]["lidar"] = os.path.join("./", lidar_path)

    for path in paths_to_keep:
        source_path = os.path.join(src_data_directory, path)
        target_path = os.path.join(target_data_directory, path)
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(source_path, target_path)

    with open(os.path.join(target_data_directory, "exported.json"), "w") as f:
        json.dump(qa_pairs, f, indent=2)


def count_sub_subfolders(directory):
    sub_subfolder_count = 0
    # List all subfolders in the main directory
    for root, dirs, files in os.walk(directory):
        # For each directory in the current root
        dirs = sorted(dirs)
        for dir in dirs[:10]:
            # Get the full path of the current subfolder
            subfolder_path = os.path.join(root, dir)
            # List all entries in the subfolder
            sub_entries = os.listdir(subfolder_path)
            # Count how many of these entries are directories (sub-subfolders)
            sub_subfolder_count += sum(os.path.isdir(os.path.join(subfolder_path, entry)) for entry in sub_entries)
    return sub_subfolder_count


def count_envs(directory):
    seed = set()
    for stuff in os.listdir(directory[:]):
        if os.path.isdir(os.path.join(directory, stuff)):
            env_id = int(stuff.split("_")[0])
            seed.add(env_id)
    return len(seed)



def count_frames(session_folder):
    episodes = os.listdir(session_folder)
    episodes = [os.path.join(session_folder, file) for file in episodes if os.path.isdir(os.path.join(session_folder,file))]
    count = 0
    for episode in episodes:
        count += len(os.listdir(episode))
    return count

def count_proper_episode(session_folder):
    episodes = os.listdir(session_folder)
    episodes = [file for file in episodes if
                os.path.isdir(os.path.join(session_folder, file))]
    valid_count = 0
    valid_episode = set()
    for episode in episodes:
        splitted = episode.split("_")
        if int(splitted[-1])-int(splitted[1])==24:
            valid_count += 1
            valid_episode.add(episode)
    return valid_count, list(valid_episode)

def count_proper_seed(session_folder):
    episodes = os.listdir(session_folder)
    episodes = [file for file in episodes if
                os.path.isdir(os.path.join(session_folder, file))]
    valid_seed = set()
    for episode in episodes:
        splitted = episode.split("_")
        valid_seed.add(splitted[0])
    return len(valid_seed), list(valid_seed)

def store_session_statistics(session_path):
    print("Collecting number of frames")
    num_frames = count_frames(session_path)
    print(f"num_franmes={num_frames}")
    print("Collecting number of 2.5s episodes")
    num_valid_episodes, valid_episodes = count_proper_episode(session_path)
    print(f"num_valid_episodes={num_valid_episodes}")
    print("Collecting number of ScenarioNet Scenarios used")
    num_valid_seeds, valid_seeds = count_proper_seed(session_path)
    print(f"num_valid_seeds={num_valid_seeds}")
    import json
    summary = dict(
        num_frames = num_frames, num_valid_episodes = num_valid_episodes, num_valid_seeds = num_valid_seeds,
        valid_episodes = valid_episodes, valid_seeds = valid_seeds
    )
    print("Summary stored at {}".format(os.path.join(session_path, "session_statistics.json")))
    json.dump(
        summary, open(os.path.join(session_path, "session_statistics.json"), "w")
    )

def display_session_statistics(session_path):
    print("Collecting number of frames")
    num_frames = count_frames(session_path)
    print(f"num_franmes={num_frames}")
    print("Collecting number of 2.5s episodes")
    num_valid_episodes, valid_episodes = count_proper_episode(session_path)
    print(f"num_valid_episodes={num_valid_episodes}")
    print("Collecting number of ScenarioNet Scenarios used")
    num_valid_seeds, valid_seeds = count_proper_seed(session_path)
    print(f"num_valid_seeds={num_valid_seeds}")



if __name__ == '__main__':
    # merge_ba("/bigdata/weizhen/metavqa/100k", "/bigdata/weizhen/metavqa/100k/merged.json", "qa")
    # dataset_statistics("./waymo_sample/merged.json","./waymo_sample/statistics.json")
    # splitting("./export_waymo/converted.json", "./export_waymo/split.json")
    # delete_files_with_prefix("./multiprocess_demo", "highlighted")
    # export_dataset_2("/bigdata/weizhen/metavqa/100k/merged.json","/bigdata/weizhen/metavqa/100k","/bigdata/weizhen/metavqa/100k_export")
    # splitting("verification/static.json", "verification/split.json")
    #print(count_sub_subfolders("/bigdata/weizhen/metavqa_final/scenarios/training/waymo/waymo_train_0"))
    #print(count_frames("/bigdata/weizhen/metavqa_final/scenarios/training/waymo/waymo_train_0"))
    print(count_proper_episode("/bigdata/weizhen/metavqa_final/scenarios/training/waymo/waymo_train_0"))
    #print(count_envs("../100k_export"))
    #store_session_statistics("/bigdata/weizhen/metavqa_final/scenarios/validation/waymo_validation_0")
    #store_session_statistics("/bigdata/weizhen/metavqa_final/scenarios/testing/normal/Waymo_testing_0")