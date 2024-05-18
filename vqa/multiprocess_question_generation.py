from vqa.static_question_generation import generate_all_frame
import json
import os
import argparse
import multiprocessing as multp
from vqa.static_question_generator import QuerySpecifier
from vqa.dynamic_question_generator import DynamicQuerySpecifier
from vqa.dynamic_question_generation import find_episodes, extract_observations, generate_dynamic_questions, \
    extract_frames, load_valid_episodes
from vqa.scene_graph import TemporalGraph
from vqa.scene_level_functionals import predict_collision
from vqa.safety_question_generation import generate_safety_questions
import glob


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


def static_job(paths, source, summary_path, verbose=False, multiview=True):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_directory, "question_templates.json")
    try:
        with open(template_path, "r") as f:
            templates = json.load(f)
    except Exception as e:
        raise e
    records = {}
    count = 0
    for path in paths:
        assert len(QuerySpecifier.CACHE) == 0, f"Non empty cache for {path}"
        folder_name = os.path.dirname(path)
        identifier = os.path.basename(folder_name)
        perspectives = ["front", "leftb", "leftf", "rightb", "rightf", "back"] if multiview else ["front"]
        lidar = os.path.join(folder_name, f"lidar_{identifier}.pkl")
        record, num_data = generate_all_frame(templates["generic"], path, 100, 10, count, verbose=verbose,
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

from vqa.dynamic_question_generation import select_key_frames
def static_setting():
    def find_world_json_paths(root_dir, frame_per_episode = 2):
        world_json_paths = []  # List to hold paths to all world_{id}.json files
        for root, dirs, files in os.walk(root_dir):
            # Extract the last part of the current path, which should be the frame folder's name
            frame_folder_name = os.path.basename(root)
            expected_json_filename = f'world_{frame_folder_name}.json'  # Construct expected filename
            if expected_json_filename in files:
                path = os.path.join(root, expected_json_filename)  # Construct full path
                world_json_paths.append(path)
        return world_json_paths
    parser = argparse.ArgumentParser()
    cwd = os.getcwd()
    default_config_path = os.path.join(cwd, "vqa", "configs", "scene_generation_config.yaml")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to generate QA data")
    parser.add_argument("--root_directory", type=str, default=None,
                        help="the paths to the recorded data")
    parser.add_argument("--output_base", type=str, default="./qa",
                        help="directory to the generated QA files, each file will be extended with id ")
    parser.add_argument("--src", type=str, default="PG", help="specify the data source of the driving scenarios")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print("Running with the following parameters")
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))
    all_paths = select_key_frames(args.root_directory,2)
    chunks = divide_list_into_n_chunks(all_paths, args.num_proc)
    processes = []
    for proc_id in range(args.num_proc):
        print(f"Sent job {proc_id}")
        p = multp.Process(
            target=static_job,
            args=(
                chunks[proc_id],
                args.src,
                os.path.join(args.output_base, f"static_qa{proc_id}.json"),
                True if args.verbose else False,
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


def dynamic_job(episode_folders, source, summary_path, verbose=False):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(current_directory, "question_templates.json")
    templates = json.load(open(templates_path, "r"))
    templates = templates["dynamic"]
    qa_tuples = {}
    idx = 0
    for episode in episode_folders:
        assert len(DynamicQuerySpecifier.CACHE) == 0, f"Non empty cache for {episode}"
        observations = extract_observations(episode)
        records, num_questions, context = generate_dynamic_questions(
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
                    lidar=observations["lidar"][:record["key_frame"] + 1],
                    metadrive_scene=extract_frames(episode),
                    multiview=True,
                    source=source,
                )
                idx += 1
        DynamicQuerySpecifier.CACHE.clear()
    json.dump(qa_tuples, open(summary_path, "w"), indent=2)


def dynamic_setting():
    parser = argparse.ArgumentParser()
    cwd = os.getcwd()
    default_config_path = os.path.join(cwd, "vqa", "configs", "scene_generation_config.yaml")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to generate QA data")
    parser.add_argument("--root_directory", type=str, default=None,
                        help="the paths to the recorded data")
    parser.add_argument("--output_base", type=str, default="./temporal_qa",
                        help="directory to the generated QA files, each file will be extended with id ")
    parser.add_argument("--src", type=str, default="PG", help="specify the data source of the driving scenarios")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print("Running with the following parameters")
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))
    all_episodes = load_valid_episodes(args.root_directory)
    print(f"Find {len(all_episodes)} valid episodes under session {args.root_directory}")
    chunks = divide_list_into_n_chunks(all_episodes, args.num_proc)
    processes = []
    for proc_id in range(args.num_proc):
        print(f"Sent job{proc_id}")
        p = multp.Process(
            target=dynamic_job,
            args=(
                chunks[proc_id],
                args.src,
                os.path.join(args.output_base, f"dynamic_qa{proc_id}.json"),
                True if args.verbose else False,
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


def safety_job(episode_folders, source, summary_path, verbose=False):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(current_directory, "question_templates.json")
    templates = json.load(open(templates_path, "r"))
    templates = templates["safety"]
    qa_tuples = {}
    idx = 0
    for episode in episode_folders:
        observations = extract_observations(episode)
        frame_files = extract_frames(episode)
        graph = TemporalGraph(frame_files, tolerance=0.5)
        collision_happend, _, first_impact = predict_collision(graph)
        if not collision_happend:
            """
            No collision, no postmortem analysis.
            """
            records, num_questions = generate_safety_questions(
                episode, templates, max_per_type=2, choose=1, attempts_per_type=10, verbose=True, only_predict=True)
        else:
            #has collision, see if collision on in prediction period.
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


def safety_setting():
    parser = argparse.ArgumentParser()
    cwd = os.getcwd()
    default_config_path = os.path.join(cwd, "vqa", "configs", "scene_generation_config.yaml")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to generate QA data")
    parser.add_argument("--root_directory", type=str, default=None,
                        help="the paths to the recorded data")
    parser.add_argument("--output_base", type=str, default="./safety_qa",
                        help="directory to the generated QA files, each file will be extended with id ")
    parser.add_argument("--src", type=str, default="PG", help="specify the data source of the driving scenarios")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print("Running with the following parameters")
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))

    all_episodes = find_episodes(args.root_directory)
    chunks = divide_list_into_n_chunks(all_episodes, args.num_proc)
    processes = []
    for proc_id in range(args.num_proc):
        p = multp.Process(
            target=safety_job,
            args=(
                chunks[proc_id],
                args.src,
                os.path.join(args.output_base, f"safety_qa{proc_id}.json"),
                True if args.verbose else False,
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


if __name__ == "__main__":
    #static_setting()
    dynamic_setting()
    #safety_setting()
