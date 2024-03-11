from vqa.static_question_generation import generate_all_frame
from vqa.grammar import STATIC_GRAMMAR
import json
import os
import argparse
import multiprocessing as multp
from vqa.question_generator import CACHE
def job(paths, source, summary_path, verbose = False):
    GRAMMAR = STATIC_GRAMMAR
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
        assert len(CACHE) == 0, f"Non empty cache for {path}"
        folder_name = os.path.dirname(path)
        identifider = os.path.basename(folder_name)
        rgb = os.path.join(folder_name, f"rgb_{identifider}.png")
        lidar = os.path.join(folder_name, f"lidar_{identifider}.json")
        record, num_data = generate_all_frame(templates["generic"], path, 100, 2, count, verbose=verbose)
        for id, info in record.items():
            records[id] = dict(
                question=info["question"],
                answer=info["answer"],
                question_type=info["question_type"],
                answer_form=info["answer_form"],
                type_statistics=info["type_statistics"],
                pos_statistics=info["pos_statistics"],
                rgb=dict(
                    front=[rgb],
                    left=[],
                    back=[],
                    right=[]
                ),
                lidar=lidar,
                source=source
            )
        count += num_data
        CACHE.clear()
    try:
        with open(summary_path, "w") as f:
            json.dump(records, f, indent=4),
    except Exception as e:
        raise e


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


if __name__ == "__main__":
    def find_world_json_paths(root_dir):
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
    parser.add_argument("--src", type = str, default="PG", help="specify the data source of the driving scenarios")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print("Running with the following parameters")
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))

    all_paths = find_world_json_paths(args.root_directory)
    chunks = divide_list_into_n_chunks(all_paths, args.num_proc)
    processes = []
    for proc_id in range(args.num_proc):
        p = multp.Process(
            target=job,
            args=(
                chunks[proc_id],
                args.src,
                os.path.join(args.output_base,f"qa{proc_id}.json"),
                True if args.verbose else False,
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


