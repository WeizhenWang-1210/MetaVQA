import argparse
import json
import multiprocessing
import os
import yaml

from metadrive import MetaDriveEnv
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioDiverseEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario import utils as sd_utils
from vqa.configs.namespace import OBS_WIDTH, OBS_HEIGHT
from vqa.scenegen.metadrive_annotation import generate_episodes
from vqa.utils.common_utils import divide_into_intervals_exclusive


def session_summary(path, dataset_summary_path, source, split, collision):
    summary_dict = dict(
        dataset_summary=dataset_summary_path,
        source=source,
        split=split,
        collision=collision
    )
    json.dump(summary_dict, open(path, 'w'), indent=2)


def main(data_directory, scenarios, headless, config, num_scenarios, job_range=None):
    """
    Main function to set up the environment and generate episodes.

    Parameters:
        data_directory (str): Path to the directory containing scenario data.
        scenarios (bool): Whether to use ScenarioNet environment.
        headless (bool): Whether to run the environment in headless mode.
        config (dict): Configuration dictionary for the environment.
        num_scenarios (int): Total number of scenarios available under `data_directory`.
        job_range (tuple): A tuple indicating the range of jobs to process, e.g., (start, end).
    Returns:
        None
    """
    # Set up the gymnasium environment
    use_render = not headless
    if scenarios:
        asset_path = AssetLoader.asset_path
        assert job_range is not None
        env_config = {
            "sequential_seed": True,
            "reactive_traffic": False,
            "use_render": use_render,
            "data_directory": data_directory if data_directory is not None else AssetLoader.file_path(
                asset_path, "nuscenes", unix_style=False
            ),
            "num_scenarios": num_scenarios,
            "agent_policy": ReplayEgoCarPolicy,
            "sensors": dict(
                rgb=(RGBCamera, OBS_WIDTH, OBS_HEIGHT),
                instance=(InstanceCamera, OBS_WIDTH, OBS_HEIGHT),
                semantic=(SemanticCamera, OBS_WIDTH, OBS_HEIGHT),
                depth=(DepthCamera, OBS_WIDTH, OBS_HEIGHT)
            ),
            "vehicle_config": dict(show_lidar=False, show_navi_mark=False, show_line_to_navi_mark=False),
            "height_scale": 0.1,
        }
        print("Finished reading")
        env = ScenarioDiverseEnv(env_config)
        env.reset(seed=job_range[0])
    else:
        # Legacy MetaDrive environment setup
        env_config = dict(
            use_render=use_render,
            manual_control=True,
            traffic_density=config["map_setting"]["traffic_density"],
            num_scenarios=config["map_setting"]["num_scenarios"],
            random_agent_model=False,
            random_lane_width=True,
            random_lane_num=True,
            need_inverse_traffic=config["map_setting"]["inverse_traffic"],
            on_continuous_line_done=False,
            out_of_route_done=True,
            vehicle_config=dict(show_lidar=False, show_navi_mark=False),
            map=config["map_setting"]["map_size"] if config["map_setting"]["PG"] else config["map_setting"][
                "map_sequence"],
            start_seed=config["map_setting"]["start_seed"],
            debug=False,
            sensors=dict(
                rgb=(RGBCamera, OBS_WIDTH, OBS_HEIGHT),
                instance=(InstanceCamera, OBS_WIDTH, OBS_HEIGHT),
                semantic=(SemanticCamera, OBS_WIDTH, OBS_HEIGHT),
                depth=(DepthCamera, OBS_WIDTH, OBS_HEIGHT)
            ),
            height_scale= 0.1,
        )
        env = MetaDriveEnv(env_config)
        if not job_range:
            env.reset(seed=0)
        else:
            env.reset(seed=job_range[0])
        env.agent.expert_takeover = True
    generate_episodes(env, config["num_samples"], config["sample_frequency"], config["max_iterations"],
                      dict(batch_folder=config["storage_path"], log=True),
                      episode_length=config["episode_length"],
                      skip_length=config["skip_length"], job_range=job_range)
    env.close()


def normal():
    parser = argparse.ArgumentParser()
    cwd = os.getcwd()
    default_config_path = os.path.join(cwd, "vqa", "configs", "scene_generation_config.yaml")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to generate data")
    parser.add_argument("--headless", action='store_true', help="Rendering in headless mode")
    parser.add_argument("--scenarios", action='store_true', help="Use ScenarioNet environment")
    parser.add_argument("--data_directory", type=str, default=None,
                        help="the paths that stores the ScenarioNet data")
    parser.add_argument("--config", type=str, default=default_config_path,
                        help="path to the data generation configuration file")
    parser.add_argument("--source", type=str, default="PG", help="Indicate the source of traffic.")
    parser.add_argument("--split", type=str, default="train", help="Indicate the split of this session.")
    parser.add_argument("--start", type=int, default=0, help="Inclusive starting index")
    parser.add_argument("--end", type=int, default=None, help="Exclusive ending index")
    args = parser.parse_args()
    print("Running with the following parameters")
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))
    try:
        # If your path is not correct, run this file with root folder based at metavqa instead of vqa.
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise e
    start, end = args.start, args.end
    if args.data_directory:
        scenario_summary, scenario_ids, scenario_files = sd_utils.read_dataset_summary(args.data_directory)
        num_scenarios = len(scenario_summary.keys())
        if args.end is None:
            args.end = num_scenarios
        foo_num_scenarios = end - start
    else:
        num_scenarios = 10
        foo_num_scenarios = num_scenarios
    if not args.scenarios:
        num_scenarios = config["map_setting"]["num_scenarios"]
        foo_num_scenarios = num_scenarios
    print("{} total scenarios distributed across {} processes".format(foo_num_scenarios, args.num_proc))
    job_intervals = divide_into_intervals_exclusive(foo_num_scenarios, args.num_proc, start=start)
    processes = []
    for proc_id in range(args.num_proc):
        print("Sending job{}".format(proc_id))
        p = multiprocessing.Process(
            target=main,
            args=(
                args.data_directory,
                args.scenarios,
                args.headless,
                config,
                num_scenarios,
                job_intervals[proc_id]
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")
    session_summary(path=os.path.join(config["storage_path"], "session_summary.json"),
                    dataset_summary_path=os.path.join(args.data_directory, "dataset_summary.pkl"),
                    source="_".join([args.source]), split=args.split, collision="False"
                    )
    

if __name__ == "__main__":
    flag = os.getenv('SETTING', "NORMAL")
    if flag == "NORMAL":
        normal()
    else:
        raise NotImplementedError
