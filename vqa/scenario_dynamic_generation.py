import cv2
import json
import os
import re
import yaml
from vqa.dynamic_filter import DynamicFilter
def find_valid_car_pairs(scene_info):
    """
    Identifies valid pairs of cars that share at least one episode.

    :param scene_info: A dictionary containing scene information, where keys are car IDs and values are dictionaries
                       with episode names as keys and episode data as values.
    :return: A set of tuples, each containing a pair of car IDs that appear together in the same episodes.
    """
    all_cars = set(scene_info.keys())
    valid_pairs = set()

    for car1 in all_cars:
        for car2 in all_cars:
            if car1 != car2:
                episodes_car1 = set(scene_info[car1].keys())
                episodes_car2 = set(scene_info[car2].keys())
                if episodes_car1.intersection(episodes_car2):
                    valid_pairs.add((car1, car2))
    return valid_pairs
def analyze_actions_for_pair(car1, car2, action_methods):
    """
    Analyzes interactions between a pair of cars using specified dynamic action detection methods.

    :param car1: ID of the first car.
    :param car2: ID of the second car.
    :param action_methods: A list of methods from the DynamicFilter class to apply to the car pair.
    :return: A dictionary where keys are episode identifiers and values are lists of dictionaries, each representing
             an action detected between the car pair in the episode. Each dictionary in the list contains 'pair'
             (a tuple of car IDs) and 'action' (the name of the detected action).
    """
    action_results = {}

    for action_method in action_methods:
        matched_episodes = action_method(car1, car2)
        for episode in matched_episodes:
            if episode not in action_results:
                action_results[episode] = []
            action_name = action_method.__name__
            action_results[episode].append({'pair': (car1, car2), 'action': action_name})

    return action_results
def analyze_car_interactions(scene_folder, sample_frequency:int, episode_length:int, skip_length:int):
    """
    Analyzes car interactions within a scene folder by identifying valid car pairs and applying dynamic action detection
    methods to them.

    :param scene_folder: Path to the folder containing scene data.
    :return: A dictionary where keys are episode identifiers and values are lists of interactions detected in the episode.
             Each interaction is represented as a dictionary containing 'pair' (a tuple of car IDs) and 'action' (the
             name of the detected action).
    """
    dynamic_filter = DynamicFilter(scene_folder,  sample_frequency, episode_length, skip_length)
    action_results = {}
    scene_info = dynamic_filter.scene_info
    valid_pairs = find_valid_car_pairs(scene_info)
    total_pairs = len(valid_pairs)
    pair_count = 0

    # Define the dynamic action methods to apply
    action_methods = [
        dynamic_filter.follow,
        dynamic_filter.pass_by,
        # dynamic_filter.collide_with,
        dynamic_filter.head_toward,
        dynamic_filter.drive_alongside
    ]
    print(f"Starting analysis of {total_pairs} car pairs.")
    # Iterate over each valid pair and analyze actions
    for car1, car2 in valid_pairs:
        pair_results = analyze_actions_for_pair(car1, car2, action_methods)
        # Merge results
        for episode, actions in pair_results.items():
            if episode not in action_results:
                action_results[episode] = []
            action_results[episode].extend(actions)
        pair_count += 1
        if pair_count % 10 == 0:  # Update after every 10 pairs processed
            print(f"Processed {pair_count} out of {total_pairs} pairs.")
    print("Analysis complete.")
    return action_results


def analyze_car_interactions(episodes_folder):
    """
    Analyzes car interactions within a scene folder by identifying valid car pairs and applying dynamic action detection
    methods to them.

    :param scene_folder: Path to the folder containing scene data.
    :return: A dictionary where keys are episode identifiers and values are lists of interactions detected in the episode.
             Each interaction is represented as a dictionary containing 'pair' (a tuple of car IDs) and 'action' (the
             name of the detected action).
    """
    dynamic_filter = DynamicFilter(episodes_folder=episodes_folder)
    action_results = {}
    scene_info = dynamic_filter.scene_info
    valid_pairs = find_valid_car_pairs(scene_info)
    total_pairs = len(valid_pairs)
    pair_count = 0

    # Define the dynamic action methods to apply
    action_methods = [
        dynamic_filter.follow,
        dynamic_filter.pass_by,
        # dynamic_filter.collide_with,
        dynamic_filter.head_toward,
        dynamic_filter.drive_alongside
    ]
    print(f"Starting analysis of {total_pairs} car pairs.")
    # Iterate over each valid pair and analyze actions
    for car1, car2 in valid_pairs:
        pair_results = analyze_actions_for_pair(car1, car2, action_methods)
        # Merge results
        for episode, actions in pair_results.items():
            if episode not in action_results:
                action_results[episode] = []
            action_results[episode].extend(actions)
        pair_count += 1
        if pair_count % 10 == 0:  # Update after every 10 pairs processed
            print(f"Processed {pair_count} out of {total_pairs} pairs.")
    print("Analysis complete.")
    return action_results



def save_results_to_json(action_results, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    total_episodes = len(action_results)
    episodes_processed = 0
    print(f"Saving results for {total_episodes} episodes.")
    for episode, actions in action_results.items():
        episode_folder = os.path.join(output_folder, episode)
        if not os.path.exists(episode_folder):
            os.makedirs(episode_folder)

        episode_results = {}
        for action in actions:
            if action['action'] not in episode_results:
                episode_results[action['action']] = []
            episode_results[action['action']].append(action['pair'])

        file_path = os.path.join(episode_folder, f"{episode}.json")
        with open(file_path, 'w') as json_file:
            json.dump(episode_results, json_file, indent=4)

    print(f"Results saved to {output_folder}")

def save_results_to_episodes(action_results,episodes_folder):
    total_episodes = len(action_results)
    print(f"Saving results for {total_episodes} episodes.")
    for episode_folder, actions in action_results.items():
        episode_results = {}
        for action in actions:
            if action['action'] not in episode_results:
                episode_results[action['action']] = []
            episode_results[action['action']].append(action['pair'])
        file_path = os.path.join(episodes_folder,episode_folder, f"interaction.json")
        try:
            with open(file_path, 'w') as json_file:
                json.dump(episode_results, json_file, indent=4)
        except Exception as e:
            raise e
    print(f"Results saved to {episodes_folder}")



def analyze_and_save_car_interactions(scene_folder, output_folder, sample_frequency:int, episode_length:int, skip_length:int):
    action_results = analyze_car_interactions(scene_folder, sample_frequency, episode_length, skip_length)

    # Save the aggregated results into JSON files
    save_results_to_json(action_results, output_folder)


def analyze_and_save_car_interactions(episodes_folder):
    action_results = analyze_car_interactions(episodes_folder)
    save_results_to_episodes(action_results, episodes_folder)




import cv2
import os
import re

def create_videos_for_episodes(base_folder, output_folder, sample_frequency, episode_length, skip_length):
    """
    Combines images from each `{seed}_{step}` subfolder into MP4 videos based on episode ranges and
    saves them in the corresponding episode folders within the output folder.

    :param base_folder: The path to the base folder containing the `{seed}_{step}` subfolders with images.
    :param output_folder: The path to the output folder where episode folders will be created.
    :param sample_frequency: The frequency of sampling the world json file.
    :param episode_length: The length of each episode.
    :param skip_length: The length of each skip.
    """
    # Create a dictionary to map episodes to their subfolders
    episode_to_subfolders = {}

    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path) and re.match(r"\d+_\d+", subfolder):
            step = int(subfolder.split('_')[-1])

            if step % (skip_length + episode_length) < skip_length:
                continue  # Skip this step
            if step % sample_frequency != 0 or (step % (skip_length + episode_length) - skip_length) >= episode_length:
                continue  # Skip non-processed steps

            # Determine the episode range
            episode_start = step - (step % (skip_length + episode_length)) + skip_length + 1
            episode_end = episode_start + episode_length - 1
            episode = f"{episode_start}-{episode_end}"

            if episode not in episode_to_subfolders:
                episode_to_subfolders[episode] = []
            episode_to_subfolders[episode].append(subfolder_path)

    # Create videos for each episode and save them in the output folder
    for episode, subfolders in episode_to_subfolders.items():
        episode_folder = os.path.join(output_folder, episode)
        if not os.path.exists(episode_folder):
            os.makedirs(episode_folder)

        for view in ['rgb', 'top_down']:
            image_files = []
            for subfolder_path in subfolders:
                image_files.extend([os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path)
                                    if f.startswith(view) and f.endswith('.png')])

            if not image_files:
                continue  # Skip if no images found

            image_files = sorted(image_files, key=lambda x: int(re.findall(r"(\d+).png$", x)[0]))

            # Assume all images have the same dimensions
            sample_image = cv2.imread(image_files[0])
            height, width, layers = sample_image.shape
            video_name = f"{view}_{episode}.mp4"
            video_path = os.path.join(episode_folder, video_name)

            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

            for img_path in image_files:
                img = cv2.imread(img_path)
                video.write(img)

            video.release()
        print(f"Videos for episode {episode} saved in {episode_folder}")


def create_videos_for_episodes(episodes_folder):
    """
    Combines images from each `{seed}_{step}` subfolder into MP4 videos based on episode ranges and
    saves them in the corresponding episode folders within the output folder.

    :param base_folder: The path to the base folder containing the `{seed}_{step}` subfolders with images.
    :param output_folder: The path to the output folder where episode folders will be created.
    :param sample_frequency: The frequency of sampling the world json file.
    :param episode_length: The length of each episode.
    :param skip_length: The length of each skip.
    """
    # Create videos for each episode and save them in the output folder
    for episode_folder in os.listdir(episodes_folder):
        episode_path = os.path.join(episodes_folder,episode_folder)
        views = ["rgb", "top_down"]
        image_files = {
            v:[] for v in views            
        }
        for item in os.listdir(episode_path):
            item_path = os.path.join(episode_path,item)
            if os.path.isdir(item_path):
                for view in views:
                    image_files[view].extend([os.path.join(episode_path, item, f) for f in os.listdir(item_path)
                                            if f.startswith(view) and f.endswith('.png')])
        for view, images in image_files.items():
            if not images:
                continue  # Skip if no images found

            images = sorted(images, key=lambda x: int(re.findall(r"(\d+).png$", x)[0]))

            # Assume all images have the same dimensions
            sample_image = cv2.imread(images[0])
            height, width, layers = sample_image.shape
            video_name = f"{view}_{episode_folder}.mp4"
            video_path = os.path.join(episode_path, video_name)
            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
            for img_path in images:
                img = cv2.imread(img_path)
                video.write(img)
            video.release()
        print(f"Videos for episode {episode_folder} saved in {episode_path}")
        


if __name__ == "__main__":
    cwd = os.getcwd()
    config_path = os.path.join(cwd,"vqa","configs","scene_generation_config.yaml")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise e
    scene_folder = os.path.join(cwd,"verification")
    #output_folder = os.path.join(cwd,"verification","episodes")
    #analyze_and_save_car_interactions(scene_folder, output_folder, sample_frequency=config["sample_frequency"],
    #                                  episode_length=config["episode_length"],skip_length=config["skip_length"])
    analyze_and_save_car_interactions(scene_folder)
    
    """create_videos_for_episodes(scene_folder, output_folder, sample_frequency=config["sample_frequency"],
                                       episode_length=config["episode_length"],skip_length=config["skip_length"])"""
    create_videos_for_episodes(scene_folder)
