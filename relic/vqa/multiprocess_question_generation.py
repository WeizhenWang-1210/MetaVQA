from relic.vqa.static_question_generation import generate_all_frame, generate_all_frame_nuscene
from relic.vqa.dynamic_question_generation import select_key_frames, extract_observations, generate_dynamic_questions, \
    extract_frames, load_valid_episodes
from vqa.safety_question_generation import generate_safety_questions
from relic.vqa.static_question_generator import QuerySpecifier
from relic.vqa.dynamic_question_generator import DynamicQuerySpecifier
from vqa.scene_graph import TemporalGraph
from relic.vqa.scene_level_functionals import predict_collision
import json
import os
import argparse
import multiprocessing as multp
import yaml
import tqdm

def divide_list_into_n_chunks(lst, n):
    """
    Divides a list into n similarly-sized chunks.

    Parameters:
    lst (list): The list to divide.
    n (int): The number of chunks to divide the list into.

    Returns:
    list of lists: A list containing the chunks.
    """
    # Ensure n is a positive integer
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Calculate the size of each chunk
    total_length = len(lst)
    chunk_size, remainder = divmod(total_length, n)

    chunks = []
    start_index = 0

    for i in range(n):
        # Calculate the end index for the current chunk, adding one if there's a remainder
        end_index = start_index + chunk_size + (1 if i < remainder else 0)
        # Slice the list to create the chunk and add it to the list of chunks
        chunks.append(lst[start_index:end_index])
        # Update the start index for the next chunk
        start_index = end_index

    return chunks


def static_job(proc_id, paths, source, summary_path, verbose=False, multiview=True):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_directory, "question_templates.json")
    try:
        with open(template_path, "r") as f:
            templates = json.load(f)
    except Exception as e:
        raise e
    records = {}
    count = 0
    for path in tqdm.tqdm(paths, desc=f"Process {proc_id}", position=proc_id):
        assert len(QuerySpecifier.CACHE) == 0, f"Non empty cache for {path}"
        folder_name = os.path.dirname(path)
        identifier = os.path.basename(folder_name)
        perspectives = ["front", "leftb", "leftf", "rightb", "rightf", "back"] if multiview else ["front"]
        lidar = os.path.join(folder_name, f"lidar_{identifier}.pkl")
        record, num_data = generate_all_frame(templates["static"], path, 100, 10, count, verbose=verbose,
                                              multiview=multiview)
        for id, info in record.items():
            records[id] = dict(
                question=info["question"],
                answer=info["answer"],
                question_type="|".join(["static", info["question_type"]]),
                answer_form=info["answer_form"],
                type_statistics=info["type_statistics"],
                pos_statistics=info["pos_statistics"],
                color_statistics=info["color_statistics"],
                ids=info["ids"],
                rgb={perspective: [os.path.join(folder_name, f'rgb_{perspective}_{identifier}.png')] for perspective in
                     perspectives},
                lidar=[lidar],
                metadrive_scene=[path],
                multiview=multiview,
                source=source,
            )
        count += num_data
        QuerySpecifier.CACHE.clear()
    try:
        with open(summary_path, "w") as f:
            json.dump(records, f, indent=2),
    except Exception as e:
        raise e





def static_setting(config_path):
    """
    Job for generating static qas on metadrive-only rendering.
    """
    # load config
    config = yaml.safe_load(open(config_path, 'r'))
    print("Running with the following parameters")
    for key, value in config.items():
        print("{}: {}".format(key, value))
    # for each 25-frame-long episode, selct 2 frames.
    all_paths = select_key_frames(config["root_directory"], 2)
    print("Working on {} frames in total, divided among {} processes.".format(len(all_paths), config["num_proc"]))
    chunks = divide_list_into_n_chunks(all_paths, config["num_proc"])
    # send jobs
    processes = []
    for proc_id in range(config["num_proc"]):
        print(f"Sent job {proc_id}")
        p = multp.Process(
            target=static_job,
            args=(
                proc_id,
                chunks[proc_id],
                config["src"],
                os.path.join(config["output_directory"], f"static_qa{proc_id}.json"),
                config["verbose"],
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


def static_job_nuscene(proc_id, paths, source, summary_path, verbose=False, multiview=True):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_directory, "question_templates.json")
    try:
        with open(template_path, "r") as f:
            templates = json.load(f)
    except Exception as e:
        raise e
    records = {}
    count = 0
    for path in tqdm.tqdm(paths, desc=f"Process {proc_id}", position=proc_id):
        assert len(QuerySpecifier.CACHE) == 0, f"Non empty cache for {path}"
        folder_name = os.path.dirname(path)
        identifier = os.path.basename(folder_name)
        perspectives = ["front", "leftb", "leftf", "rightb", "rightf", "back"] if multiview else ["front"]
        lidar = os.path.join(folder_name, f"lidar_{identifier}.pkl")
        record, num_data = generate_all_frame_nuscene(templates["static"], path, 100, 10, count, verbose=verbose,
                                                      multiview=multiview)
        for id, info in record.items():
            records[id] = dict(
                question=info["question"],
                answer=info["answer"],
                question_type="|".join(["static", info["question_type"]]),
                answer_form=info["answer_form"],
                type_statistics=info["type_statistics"],
                pos_statistics=info["pos_statistics"],
                color_statistics=info["color_statistics"],
                ids=info["ids"],
                rgb={perspective: [os.path.join(folder_name, f'rgb_{perspective}_{identifier}.png')] for perspective in
                     perspectives},
                real={
                    perspective: [os.path.join(folder_name, f'real_{perspective}_{identifier}.png')] for perspective in
                    perspectives
                },
                lidar=[lidar],
                metadrive_scene=[path],
                multiview=multiview,
                source=source,
            )
        count += num_data
        QuerySpecifier.CACHE.clear()
    try:
        with open(summary_path, "w") as f:
            json.dump(records, f, indent=2),
    except Exception as e:
        raise e


def static_setting_nuscene(config_path):
    """
    Job for generating static qas with nuScenes observation.
    """
    # load config
    config = yaml.safe_load(open(config_path, 'r'))
    print("Running with the following parameters")
    for key, value in config.items():
        print("{}: {}".format(key, value))
    # for each 25-frame-long episode, selct 2 frames.
    all_paths = select_key_frames(config["root_directory"], 2)
    print("Working on {} frames in total, divided among {} processes.".format(len(all_paths), config["num_proc"]))
    chunks = divide_list_into_n_chunks(all_paths, config["num_proc"])
    # send jobs
    processes = []
    for proc_id in range(config["num_proc"]):
        print(f"Sent job {proc_id}")
        p = multp.Process(
            target=static_job_nuscene,
            args=(
                proc_id,
                chunks[proc_id],
                config["src"],
                os.path.join(config["output_directory"], f"static_qa_nuscene{proc_id}.json"),
                config["verbose"]
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


def dynamic_job(proc_id, episode_folders, source, summary_path, verbose=False, multiview=True):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(current_directory, "question_templates.json")
    templates = json.load(open(templates_path, "r"))
    templates = templates["dynamic"]
    qa_tuples = {}
    idx = 0
    for episode in tqdm.tqdm(episode_folders, desc=f"Process {proc_id}", position=proc_id):
        assert len(DynamicQuerySpecifier.CACHE) == 0, f"Non empty cache for {episode}"
        observations = extract_observations(episode)
        records, num_questions, context = generate_dynamic_questions(
            episode, templates, max_per_type=50, choose=5, attempts_per_type=100, verbose=verbose)  # 50 5 100
        for question_type, record_list in records.items():
            for record in record_list:
                qa_tuples[idx] = dict(
                    question=" ".join([record["question"], context]), answer=record["answer"],
                    question_type="|".join(["dynamic", question_type]), answer_form=record["answer_form"],
                    type_statistics=record["type_statistics"], pos_statistics=record["pos_statistics"],
                    color_statistics=record["color_statistics"], action_statistics=record["action_statistics"],
                    interaction_statistics=record["interaction_statistics"], ids=record["ids"],
                    rgb={
                        perspective: observations[perspective][:record["key_frame"] + 1] for perspective in
                        ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                    },
                    lidar=observations["lidar"][:record["key_frame"] + 1],
                    metadrive_scene=extract_frames(episode),
                    multiview=multiview,
                    source=source,
                )
                idx += 1
        DynamicQuerySpecifier.CACHE.clear()
    json.dump(qa_tuples, open(summary_path, "w"), indent=2)


def dynamic_setting(config_path):
    """
    Job for generating dynamic qas on metadrive-only rendering.
    """
    # load config
    config = yaml.safe_load(open(config_path, 'r'))
    print("Running with the following parameters")
    for key, value in config.items():
        print("{}: {}".format(key, value))
    # find all episodes, annotated under root_directory.
    all_episodes = load_valid_episodes(config["root_directory"])
    print("Find {} valid episodes under session {}".format(len(all_episodes), config["root_directory"]))
    chunks = divide_list_into_n_chunks(all_episodes, config["num_proc"])
    processes = []
    for proc_id in range(config["num_proc"]):
        print(f"Sent job{proc_id}")
        p = multp.Process(
            target=dynamic_job,
            args=(
                proc_id,
                chunks[proc_id],
                config["src"],
                os.path.join(config["output_directory"], f"dynamic_qa{proc_id}.json"),
                config["verbose"]
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


from relic.vqa.dynamic_question_generation import extract_real_observations, generate_dynamic_questions_nuscene


def dynamic_job_nuscene(proc_id, episode_folders, source, summary_path, verbose=False):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(current_directory, "question_templates.json")
    templates = json.load(open(templates_path, "r"))
    templates = templates["dynamic"]
    qa_tuples = {}
    idx = 0
    for episode in tqdm.tqdm(episode_folders, desc=f"Process {proc_id}", position=proc_id):
        assert len(DynamicQuerySpecifier.CACHE) == 0, f"Non empty cache for {episode}"
        observations = extract_observations(episode)
        real_observations = extract_real_observations(episode)
        records, num_questions, context = generate_dynamic_questions_nuscene(
            episode, templates, max_per_type=50, choose=5, attempts_per_type=100, verbose=verbose)
        for question_type, record_list in records.items():
            for record in record_list:
                qa_tuples[idx] = dict(
                    question=" ".join([record["question"], context]), answer=record["answer"],
                    question_type="|".join(["dynamic", question_type]), answer_form=record["answer_form"],
                    type_statistics=record["type_statistics"], pos_statistics=record["pos_statistics"],
                    color_statistics=record["color_statistics"], action_statistics=record["action_statistics"],
                    interaction_statistics=record["interaction_statistics"], ids=record["ids"],
                    rgb={
                        perspective: observations[perspective][:record["key_frame"] + 1] for perspective in
                        ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                    },
                    real={
                        perspective: real_observations[perspective][:4] for perspective in
                        ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                    },
                    lidar=observations["lidar"][:record["key_frame"] + 1],
                    metadrive_scene=extract_frames(episode),
                    multiview=True,
                    source=source,
                )
                idx += 1
        DynamicQuerySpecifier.CACHE.clear()
    json.dump(qa_tuples, open(summary_path, "w"), indent=2)


def dynamic_setting_nuscene(config_path):
    """
    Job for generating dynamic qas on nuscenes observations.
    """
    # load config
    config = yaml.safe_load(open(config_path, 'r'))
    print("Running with the following parameters")
    for key, value in config.items():
        print("{}: {}".format(key, value))
    # find all episodes, annotated under root_directory.
    all_episodes = load_valid_episodes(config["root_directory"])
    print("Find {} valid episodes under session {}".format(len(all_episodes), config["root_directory"]))
    chunks = divide_list_into_n_chunks(all_episodes, config["num_proc"])
    processes = []
    for proc_id in range(config["num_proc"]):
        print(f"Sent job{proc_id}")
        p = multp.Process(
            target=dynamic_job_nuscene,
            args=(
                proc_id,
                chunks[proc_id],
                config["src"],
                os.path.join(config["output_directory"], f"dynamic_qa_nuscene{proc_id}.json"),
                config["verbose"]
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


def safety_job(proc_id, episode_folders, source, summary_path, verbose=False):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(current_directory, "question_templates.json")
    templates = json.load(open(templates_path, "r"))
    templates = templates["safety"]
    qa_tuples = {}
    idx = 0
    for episode in tqdm.tqdm(episode_folders, desc=f"Process {proc_id}", position=proc_id):
        observations = extract_observations(episode)
        frame_files = extract_frames(episode)
        graph = TemporalGraph(frame_files, tolerance=0.5)
        collision_happend, collided_objects, first_impact = predict_collision(graph)
        skip = False
        for collided_object in collided_objects:
            if collided_object not in graph.get_nodes().keys():
                skip = True
                break
        if skip:
            print(f"Skipping {episode} as collided objects is unobservable for 50% of of observation period.")
            continue
        if not collision_happend:
            """
            No collision, no postmortem analysis.
            """
            print(f"No collision for {episode}")
            records, num_questions = generate_safety_questions(
                episode, templates, max_per_type=2, choose=1, attempts_per_type=10, verbose=verbose, only_predict=True)
        else:
            # has collision, see if collision on in prediction period.
            if first_impact >= 20:
                records, num_questions = generate_safety_questions(
                    episode, templates, max_per_type=100, choose=10, attempts_per_type=200, verbose=verbose)
            else:
                if verbose:
                    print(f"Skipping episode {episode} for premature collision")
                continue
        for question_type, record_list in records.items():
            if question_type == "predict_collision":
                for record in record_list:
                    qa_tuples[idx] = dict(
                        question=record["question"], answer=record["answer"],
                        question_type="|".join(["safety", question_type]), answer_form=record["answer_form"],
                        rgb={
                            perspective: observations[perspective][:record["key_frame"] + 1] for perspective in
                            ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                        },
                        lidar=observations["lidar"][:record["key_frame"] + 1],
                        metadrive_scene=extract_frames(episode),
                        multiview=True,
                        source=source
                    )
                    idx += 1
            else:
                for record in record_list:
                    qa_tuples[idx] = dict(
                        question=record["question"], answer=record["answer"],
                        question_type="|".join(["safety", question_type]), answer_form=record["answer_form"],
                        rgb={
                            perspective: observations[perspective][5:] for perspective in
                            ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                        },
                        lidar=observations["lidar"][5:],
                        metadrive_scene=extract_frames(episode),
                        multiview=True,
                        source=source
                    )
                    idx += 1
    json.dump(qa_tuples, open(summary_path, "w"), indent=2)


def safety_setting(config_path):
    """
    Job for generating safety qas on metadrive-only rendering.
    """
    # load config
    config = yaml.safe_load(open(config_path, 'r'))
    print("Running with the following parameters")
    for key, value in config.items():
        print("{}: {}".format(key, value))
    # find all episodes, annotated under root_directory.
    all_episodes = load_valid_episodes(config["root_directory"])
    print("Find {} valid episodes under session {}".format(len(all_episodes), config["root_directory"]))
    chunks = divide_list_into_n_chunks(all_episodes, config["num_proc"])
    processes = []
    for proc_id in range(config["num_proc"]):
        print(f"Sent job{proc_id}")
        p = multp.Process(
            target=safety_job,
            args=(
                proc_id,
                chunks[proc_id],
                config["src"],
                os.path.join(config["output_directory"], f"safety_qa{proc_id}.json"),
                config["verbose"]
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


def safety_job_nuscene(proc_id, episode_folders, source, summary_path, verbose=False):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(current_directory, "question_templates.json")
    templates = json.load(open(templates_path, "r"))
    templates = templates["safety"]
    qa_tuples = {}
    idx = 0
    for episode in tqdm.tqdm(episode_folders, desc=f"Process {proc_id}", position=proc_id):
        observations = extract_observations(episode)
        real_observations = extract_real_observations(episode)
        frame_files = extract_frames(episode)
        graph = TemporalGraph(frame_files, tolerance=0.5)
        collision_happend, collided_objects, first_impact = predict_collision(graph)
        skip = False
        for collided_object in collided_objects:
            if collided_object not in graph.get_nodes().keys():
                skip = True
                break
        if skip:
            print(f"Skipping {episode} as collided objects is unobservable for 50% of of observation period.")
            continue
        if not collision_happend:
            """
            No collision, no postmortem analysis.
            """
            print(f"No collision for {episode}")
            records, num_questions = generate_safety_questions(
                episode, templates, max_per_type=2, choose=1, attempts_per_type=10, verbose=verbose, only_predict=True)
        else:
            # has collision, see if collision on in prediction period.
            if first_impact >= 20:
                records, num_questions = generate_safety_questions(
                    episode, templates, max_per_type=100, choose=10, attempts_per_type=200, verbose=verbose)
            else:
                if verbose:
                    print(f"Skipping episode {episode} for premature collision")
                continue
        for question_type, record_list in records.items():
            if question_type == "predict_collision":
                for record in record_list:
                    qa_tuples[idx] = dict(
                        question=record["question"], answer=record["answer"],
                        question_type="|".join(["safety", question_type]), answer_form=record["answer_form"],
                        rgb={
                            perspective: observations[perspective][:record["key_frame"] + 1] for perspective in
                            ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                        },
                        real={
                            perspective: real_observations[perspective][:4] for perspective in
                            ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                        },
                        lidar=observations["lidar"][:record["key_frame"] + 1],
                        metadrive_scene=extract_frames(episode),
                        multiview=True,
                        source=source
                    )
                    idx += 1
            else:
                for record in record_list:
                    qa_tuples[idx] = dict(
                        question=record["question"], answer=record["answer"],
                        question_type="|".join(["safety", question_type]), answer_form=record["answer_form"],
                        rgb={
                            perspective: observations[perspective][5:] for perspective in
                            ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                        },
                        real={
                            perspective: real_observations[perspective][1:] for perspective in
                            ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                        },
                        lidar=observations["lidar"][5:],
                        metadrive_scene=extract_frames(episode),
                        multiview=True,
                        source=source
                    )
                    idx += 1
    json.dump(qa_tuples, open(summary_path, "w"), indent=2)


def safety_setting_nuscene(config_path):
    """
        Job for generating safety qas on metadrive-only rendering.
        """
    # load config
    config = yaml.safe_load(open(config_path, 'r'))
    print("Running with the following parameters")
    for key, value in config.items():
        print("{}: {}".format(key, value))
    # find all episodes, annotated under root_directory.
    all_episodes = load_valid_episodes(config["root_directory"])
    print("Find {} valid episodes under session {}".format(len(all_episodes), config["root_directory"]))
    chunks = divide_list_into_n_chunks(all_episodes, config["num_proc"])
    processes = []
    for proc_id in range(config["num_proc"]):
        print(f"Sent job{proc_id}")
        p = multp.Process(
            target=safety_job_nuscene,
            args=(
                proc_id,
                chunks[proc_id],
                config["src"],
                os.path.join(config["output_directory"], f"safety_qa_nuscene{proc_id}.json"),
                config["verbose"]
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=["static", "dynamic", "safety", "static_nusc", "dynamic_nusc", "safety_nusc"],
                        help="Choose the type of VQA gen job to run.")
    parser.add_argument("--config", type=str, help="Specify the configuration of this VQA gen session.")
    args = parser.parse_args()
    job_mapping = {
        "static": static_setting,
        "dynamic": dynamic_setting,
        "safety": safety_setting,
        "static_nusc": static_setting_nuscene,
        "dynamic_nusc": dynamic_setting_nuscene,
        "safety_nusc": safety_setting_nuscene
    }
    job_mapping[args.job](args.config)
