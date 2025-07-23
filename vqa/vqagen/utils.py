import glob
import json
import os
import os.path
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

import tqdm


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


def random_choice(qa_records):
    for key, record in qa_records.items():
        valid_options = list(record["options"].keys())
        assert record["type"] == "describe_scenario" or not len(valid_options) <= 0
        if len(valid_options) == 0:
            record["final_choice"] = ""
        else:
            record["final_choice"] = random.choice(valid_options)
    return qa_records


def merge_qas(qa_records):
    result = {}
    idx = 0
    for qa_record in tqdm.tqdm(qa_records, desc="Merging", unit="file"):
        local_idx = 0
        for record in tqdm.tqdm(qa_record.values(), unit="point"):
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

    for qid, record in tqdm.tqdm(qa_records.items(), desc="Analyzing", unit="point"):
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


def copy_file(src, dest):
    try:
        shutil.copy2(src, dest)  # copy2 also preserves metadata like timestamps
        print(f"Copied {src} to {dest}")
    except Exception as e:
        print(f"Error copying {src} to {dest}: {e}")


# Function to perform parallel file copying
def parallel_copy(mappings, num_threads=16):
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


def export_multiple(qa_paths, obs_directory, vqa_directory):
    import tqdm
    obs_old2new = dict()
    token_name = 0
    transfer_tuples = []
    # new_records = copy.deepcopy(qa_records)
    for qa_path in tqdm.tqdm(qa_paths, desc="Working on QAs", unit="file"):
        name = os.path.basename(qa_path)
        qa_records = json.load(open(qa_path, "r"))
        for qid, record in tqdm.tqdm(qa_records.items(), desc=f"{qa_path}: Refactoring obs", unit="data"):
            old_ob_paths = record["obs"]
            for old_ob_path in old_ob_paths:
                if old_ob_path not in obs_old2new.keys():
                    new_ob_path = os.path.join(obs_directory, f"{token_name}.png")
                    relative_new_ob_path = os.path.relpath(new_ob_path, vqa_directory)
                    transfer_tuples.append((old_ob_path, new_ob_path))
                    obs_old2new[old_ob_path] = relative_new_ob_path
                    token_name += 1
            record["obs"] = [obs_old2new[path] for path in old_ob_paths]
        json.dump(qa_records, open(os.path.join(vqa_directory, f"{name}"), "w"), indent=2)
    json.dump(obs_old2new, open(os.path.join(obs_directory, "old2new.json"), "w"), indent=2)
    parallel_copy(transfer_tuples)


def split(path, split_path, train_path, val_path):
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


def generate_tarinval():
    waymo_sim_paths = glob.glob("/bigdata/weizhen/metavqa_cvpr/vqas/waymo_sim/*_waymo_sim.json")
    waymo_sim_paths = [path for path in waymo_sim_paths if path != "/bigdata/weizhen/metavqa_cvpr/vqas/waymo_sim/31_waymo_sim.json"]
    nusc_real_paths = glob.glob("/bigdata/weizhen/metavqa_cvpr/vqas/nusc_real/*_nusc_real.json")
    nusc_real_paths = [path for path in nusc_real_paths if path != "/bigdata/weizhen/metavqa_cvpr/vqas/nusc_real/31_nusc_real.json"]
    nusc_sim_paths = glob.glob("/bigdata/weizhen/metavqa_cvpr/vqas/nusc_sim/*_nusc_sim.json")
    nusc_sim_paths = [path for path in nusc_sim_paths if path != "/bigdata/weizhen/metavqa_cvpr/vqas/nusc_sim/31_nusc_sim.json"]
    print("Found all")
    assert len(nusc_sim_paths) == len(waymo_sim_paths) == len(nusc_real_paths) == 31
    #waymo_sim
    waymo_sim_qas = [json.load(open(path)) for path in waymo_sim_paths]
    merged_qa = merge_qas(waymo_sim_qas)
    selected_qas = build_subset(merged_qa, size=50000)
    for key in selected_qas.keys():
        selected_qas[key]["domain"] = "sim"
    json.dump(selected_qas, open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/waymo_sim.json", "w"), indent=2)
    #nusc_real
    nusc_real_qas = [json.load(open(path)) for path in nusc_real_paths]
    merged_qa = merge_qas(nusc_real_qas)
    selected_qas = build_subset(merged_qa, size=50000)
    for key in selected_qas.keys():
        selected_qas[key]["domain"] = "real"
    json.dump(selected_qas, open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_real.json", "w"), indent=2)
    #nusc_sim
    nusc_sim_qas = [json.load(open(path)) for path in nusc_sim_paths]
    merged_qa = merge_qas(nusc_sim_qas)
    selected_qas = build_subset(merged_qa, size=50000)
    for key in selected_qas.keys():
        selected_qas[key]["domain"] = "sim"
    json.dump(selected_qas, open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_sim.json", "w"), indent=2)


def analyze_trainval():
    waymo_sim = json.load(open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/waymo_sim.json", "r"))
    json.dump(
        analyze_dataset(waymo_sim), open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/waymo_sim_stats.json", "w"), indent=2
    )
    nusc_sim = json.load(open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_sim.json", "r"))
    json.dump(
        analyze_dataset(nusc_sim),
        open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_sim_stats.json", "w"), indent=2
    )
    nusc_real = json.load(open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_real.json", "r"))
    json.dump(
        analyze_dataset(nusc_real),
        open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_real_stats.json", "w"), indent=2
    )


def build_trainval():
    qa_paths = [
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/waymo_sim.json",
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_sim.json",
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_real.json"
    ]
    qas = [json.load(open(path)) for path in qa_paths]
    merged_qas = merge_qas(qas)
    json.dump(merged_qas, open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/trainval.json", "w"),indent=2)
    json.dump(
        analyze_dataset(merged_qas),
        open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/trainval_stats.json", "w"), indent=2
    )


def build_scaling():
    factors = [0.5,0.25]
    num_points = [int(factor * 50000) for factor in factors]
    qa_paths = [
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/waymo_sim.json",
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_sim.json",
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_real.json"
    ]
    qas = [json.load(open(path)) for path in qa_paths]
    for num_point in num_points:
        subset = [build_subset(qa, num_point) for qa in qas]
        merged = merge_qas(subset)
        analysis = analyze_dataset(merged)
        json.dump(merged, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/scaling/{num_point*3}_trainval.json", "w"),
                  indent=2)
        json.dump(
            analysis,
            open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/scaling/{num_point*3}_trainval_stats.json", "w"), indent=2
        )
        qas = subset


def build_sim2real():
    qa_paths = [
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/waymo_sim.json",
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_sim.json",
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_real.json"
    ]
    qas = [json.load(open(path)) for path in qa_paths]
    sims = [build_subset(qas[0], 25000), build_subset(qas[1], 25000)]
    reals = [qas[2]]
    simreals = [build_subset(qas[0], 12500), build_subset(qas[1], 12500), build_subset(qas[1], 25000)]

    sim, real, simreal  = merge_qas(sims), merge_qas(reals), merge_qas(simreals)
    sim_analysis, real_analysis, simreal_analysis = analyze_dataset(sim), analyze_dataset(real), analyze_dataset(simreal)
    json.dump(sim, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/sim2real/sim.json", "w"), indent=2)
    json.dump(sim_analysis, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/sim2real/sim_stats.json", "w"), indent=2)

    json.dump(real, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/sim2real/real.json", "w"), indent=2)
    json.dump(real_analysis, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/sim2real/real_stats.json", "w"), indent=2)

    json.dump(simreal, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/sim2real/simreal.json", "w"), indent=2)
    json.dump(simreal_analysis, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/sim2real/simreal_stats.json", "w"), indent=2)


def build_subset(qa_record, size):
    selected_keys = random.sample(list(qa_record.keys()), k=size)
    new_idx = 0
    selected_qas = dict()
    for key in selected_keys:
        selected_qas[new_idx] = qa_record[key]
        new_idx += 1
    return selected_qas


def build_diversification():
    qa_paths = [
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/waymo_sim.json",
        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/nusc_sim.json",
    ]
    qas = [json.load(open(path)) for path in qa_paths]
    waymos = [qas[0]]
    nuscs = [qas[1]]
    waymonuscs = [build_subset(qas[0], 25000), build_subset(qas[1], 25000)]
    waymo, nusc, waymonusc = merge_qas(waymos), merge_qas(nuscs), merge_qas(waymonuscs)
    waymo_analysis, nusc_analysis, waymonusc_analysis = analyze_dataset(waymo), analyze_dataset(nusc), analyze_dataset(waymonusc)
    json.dump(waymo, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/diversification/waymo.json", "w"), indent=2)
    json.dump(waymo_analysis, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/diversification/waymo_stats.json", "w"),
              indent=2)

    json.dump(nusc, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/diversification/nusc.json", "w"), indent=2)
    json.dump(nusc_analysis, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/diversification/nusc_stats.json", "w"),
              indent=2)

    json.dump(waymonusc, open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/diversification/waymonusc.json", "w"), indent=2)
    json.dump(waymonusc_analysis,
              open(f"/bigdata/weizhen/metavqa_cvpr/vqas/experiments/diversification/waymonusc_stats.json", "w"), indent=2)


def build_test():
    qa_paths = [
        "/bigdata/weizhen/metavqa_cvpr/vqas/nusc_real/31_nusc_real.json",
        "/bigdata/weizhen/metavqa_cvpr/vqas/nusc_sim/31_nusc_sim.json",
        "/bigdata/weizhen/metavqa_cvpr/vqas/waymo_sim/31_waymo_sim.json"
    ]
    qas = [json.load(open(path)) for path in qa_paths]
    all_qas = [
        build_subset(qas[0], 5000), build_subset(qas[1], 2500), build_subset(qas[2], 2500)
    ]
    sims = [all_qas[1], all_qas[2]]
    for sim in sims:
        for key in sim.keys():
            sim[key]["domain"]="sim"

    reals= [all_qas[0]]
    for real in reals:
        for key in real.keys():
            real[key]["domain"]="real"
    total = merge_qas(all_qas)
    total_analysis = analyze_dataset(total)

    json.dump(total, open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/test/test.json","w"), indent=2)
    json.dump(total_analysis, open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/test/test_stats.json", "w"), indent=2)

    sim = merge_qas(sims)
    sim_analysis = analyze_dataset(sim)
    json.dump(sim, open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/test/test_sim.json", "w"), indent=2)
    json.dump(sim_analysis, open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/test/test_sim_stats.json", "w"),
              indent=2)


    real = merge_qas(reals)
    real_analysis = analyze_dataset(real)
    json.dump(real, open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/test/test_real.json", "w"), indent=2)
    json.dump(real_analysis, open("/bigdata/weizhen/metavqa_cvpr/vqas/experiments/test/test_real_stats.json", "w"),
              indent=2)


def analyze_gt(qa_records):
    def find_identifier(path):
        base = os.path.basename(path)
        splitted = base.split("_")
        episode = splitted[0]
        step = splitted[1].split(".")[0]
        return (int(episode), int(step))

    statistics = dict(
        total=0, question_dist=dict(), answer_dist=defaultdict(lambda: 0), total_frames=0, total_scenarios=0
    )
    frames, scenarios = set(), set()

    for qid, record in qa_records.items():
        frames.add(find_identifier(record["obs"][-1]))
        scenarios.add(record["world"][-1])
        statistics["total"] += 1
        if record["type"] not in statistics["question_dist"].keys():
            statistics["question_dist"][record["type"]] = dict(
                count=0, answer_dist=defaultdict(lambda: 0)
            )
        statistics["question_dist"][record["type"]]["count"] += 1
        statistics["question_dist"][record["type"]]["answer_dist"][record["answer"]] += 1
        statistics["answer_dist"][record["answer"]] += 1
    print(frames)
    statistics["total_frames"] = len(frames)
    statistics["total_scenarios"] = len(scenarios)
    return statistics


def analyze_dataset():
    qa_roots = [
        "/bigdata/weizhen/metavqa_cvpr/vqas/nusc_real",
        "/bigdata/weizhen/metavqa_cvpr/vqas/nusc_sim",
        "/bigdata/weizhen/metavqa_cvpr/vqas/waymo_sim"
    ]
    all_qas = []
    for qa_root in qa_roots:
        basename = os.path.basename(qa_root)
        template = os.path.join(qa_root, f"*_{basename}.json")
        results = glob.glob(template)
        all_qas += results
    all_qas = all_qas
    pprint(all_qas)
    stats = dict(
        type=defaultdict(lambda:0),
        answer_dist=defaultdict(lambda:0),
        domain = defaultdict(lambda:0),
        num_frame=0,
        num_pics=0
    )
    frames = set()
    pics = set()
    for qa_path in all_qas:
        qa = json.load(open(qa_path,"r"))
        if "sim" in qa["0"]["world"][-1]:
            domain_code = "sim"
        elif "real" in qa["0"]["world"][-1]:
            domain_code = "real"
        else:
            raise ValueError
        for qid, record in qa.items():
            frames.add(record["world"][-1])
            pics.add(record["obs"][-1])
            stats["type"][record["type"]] += 1
            stats["answer_dist"][record["answer"]] += 1
            stats["domain"][domain_code] += 1
    stats["num_frame"] += len(frames)
    stats["num_pics"] += len(pics)
    json.dump(
        stats, open("/bigdata/weizhen/metavqa_cvpr/vqas/vqa_compositions.json","w"), indent=2
    )


def misc():
    """
    Old scripts. May be cleaned
    """
    """
    For parsing a generated response json file, you can run see the following as an example.
    """
    #for parsing response    
    response_path = "/bigdata/weizhen/repo/qa_platform/public/data_verification_result.json"
    responses = json.load(open(response_path, "r"))
    for qid in responses.keys():
        choice = parse_response(responses[qid]["model_response"])
        responses[qid]["final_choice"] = choice
    json.dump(
        responses, open("/bigdata/weizhen/repo/qa_platform/public/data_verification_result_parsed.json", "w"), indent=2
    )
    qa = json.load(open("/bigdata/weizhen/repo/qa_platform/public/data_verification_result_parsed.json", "r"))
    stat_by_category, total, total_correct = accuracy_analysis(qa)

    result = dict(
        total_questions = total, total_correct = total_correct, stats = stat_by_category
    )
    answer = []
    for record in qa.values():
        answer.append(record["answer"])
    print(";".join(answer))
    json.dump(result, open("/bigdata/weizhen/repo/qa_platform/public/data_verification_result_parsed_stat.json", "w"),
              indent=2)
    
    record_template = "/home/weizhen/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/*qa.json"
    traj_template = "/home/weizhen/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/*traj.json"
    record_paths = glob.glob(record_template)
    records = [
        json.load(open(record_path)) for record_path in record_paths
    ]
    traj_paths = glob.glob(traj_template)
    trajs = [
        json.load(open(traj_path)) for traj_path in traj_paths
    ]
 
    merged_traj = dict(gt=dict(), opt=dict(), act=dict(), crash=dict(), off=dict(), completion=dict())
    for traj in trajs:
        for key, value in traj.items():
            for scene_id in value.keys():
                print(value[scene_id])
                merged_traj[key][scene_id] = value[scene_id]
    json.dump(
        merged_traj,
        open("/home/weizhen/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/traj.json", "w"),
        indent=2
    )
    merged_qa = merge_qas(records)
    json.dump(
        merged_qa,
        open("/home/weizhen/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/qa.json", "w"),
        indent=2
    )
    stats = analyze_gt(merged_qa)
 
    json.dump(
        stats,
        open("/home/weizhen/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/qa_stats.json", "w"),
        indent=2
    )



if __name__ == "__main__":
    pass
    """
    For reproduction, you can run analyze_dataset() to inspect the dataset statistics.
    """
    # analyze_dataset()

    """
    To create the training sets, run the following methods to merge the generated json files from multiple processes and create a balanced samples of 150,000 VQA tuples in total.
    """
    # generate_tarinval()
    # build_trainval()
    # build_diversification()
    # build_sim2real()
    # build_scaling()

    # Similearly, to create the test set, 
    # build_test()

    """
    Finally, package the datasets into self-cotained folders for moving around
    First, export the training sets/ablations/experiments
    """
    # qas = ["/bigdata/weizhen/metavqa_cvpr/vqas/experiments/trainval/trainval.json",
    #        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/scaling/75000_trainval.json",
    #        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/scaling/37500_trainval.json",
    #        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/sim2real/sim.json",
    #        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/sim2real/real.json",
    #        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/sim2real/simreal.json",
    #        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/diversification/waymo.json",
    #        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/diversification/nusc.json",
    #        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/diversification/waymonusc.json"
    #        ]
    # export_multiple(qas, "/bigdata/weizhen/metavqa_cvpr/exports/experiments/obs", "/bigdata/weizhen/metavqa_cvpr/exports/experiments")

    """
    Second, export the test sets
    """
    # qas = ["/bigdata/weizhen/metavqa_cvpr/vqas/experiments/test/test.json",
    #        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/test/test_sim.json",
    #        "/bigdata/weizhen/metavqa_cvpr/vqas/experiments/test/test_real.json",
    #        ]
    # export_multiple(qas, "/bigdata/weizhen/metavqa_cvpr/exports/test/obs",
    #                 "/bigdata/weizhen/metavqa_cvpr/exports/test")


    """
    For ablation study, you can run the following methods to merge the generated json files from multiple processes and create a balanced samples of 150,000 VQA tuples in total.
    """
    # ablation_files = glob.glob("/bigdata/weizhen/metavqa_cvpr/vqas/general_ablations/*_general.json")
    # ablation_qas = [json.load(open(qa_path, "r")) for qa_path in ablation_files]
    # merged_qa = merge_qas(ablation_qas)    
    # json.dump(merged_qa, open("/bigdata/weizhen/metavqa_cvpr/vqas/general_ablations/general_ablations.json","w"), indent=2)
    # json.dump(
    #     analyze_dataset(merged_qa), open("/bigdata/weizhen/metavqa_cvpr/vqas/general_ablations/grounding_ablations_stats.json", "w"), indent=2
    # )
    # obs_directory = "/bigdata/weizhen/metavqa_cvpr/exports/general_ablations/obs" 
    # vqa_directory = "/bigdata/weizhen/metavqa_cvpr/exports/general_ablations" 
    # export(qa_path="/bigdata/weizhen/metavqa_cvpr/vqas/general_ablations/general_ablations.json", obs_directory=obs_directory,
    #        vqa_directory=vqa_directory)


    
    
    
    





    

