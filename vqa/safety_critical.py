"""def safety(data_directory, headless, config, num_scenarios, job_range=None, prefix=""):
    env = ScenarioDiverseEnv(
        {
            "sequential_seed": True,
            "reactive_traffic": True,
            "use_render": not headless,
            "data_directory": data_directory,
            "num_scenarios": num_scenarios,
            "agent_policy": ReplayEgoCarPolicy,
            "sensors": dict(
                rgb=(RGBCamera, OBS_WIDTH, OBS_HEIGHT),
                instance=(InstanceCamera, OBS_WIDTH, OBS_HEIGHT),
                semantic=(SemanticCamera, OBS_WIDTH, OBS_HEIGHT),
                depth=(DepthCamera, OBS_WIDTH, OBS_HEIGHT)
            ),
            "height_scale": 1
        }
    )
    generate_safe_data(env, job_range, config["storage_path"], prefix)

"""

"""def safety_critical():
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
        num_scenarios = end - start
    else:
        num_scenarios = 10
    if not args.scenarios:
        num_scenarios = config["map_setting"]["num_scenarios"]
    print("{} total scenarios distributed across {} processes".format(num_scenarios, args.num_proc))
    job_intervals = divide_into_intervals_exclusive(num_scenarios, args.num_proc, start=start)
    prefix = os.path.basename(args.data_directory)
    job_intervals = [list(range(*job_interval)) for job_interval in job_intervals]
    processes = []
    for proc_id in range(args.num_proc):
        print("Sending job{}".format(proc_id))
        p = multiprocessing.Process(
            target=safety,
            args=(
                args.data_directory,
                args.headless,
                config,
                num_scenarios,
                job_intervals[proc_id],
                prefix
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")
    session_summary(path=os.path.join(config["storage_path"], "session_summary.json"),
                    dataset_summary_path=os.path.join(args.data_directory, "dataset_summary.pkl"),
                    source="_".join(["CAT", args.source]), split=args.split, collision="True"
                    )
"""