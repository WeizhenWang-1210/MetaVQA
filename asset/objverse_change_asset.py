
from metadrive.envs.test_asset_metadrive_env import TestAssetMetaDriveEnv
from metadrive.examples import expert
import tkinter as tk
from tkinter import ttk
import threading
# Set the envrionment config
asset_metainfo = {
    "TIRE_RADIUS": 0.313,
    "TIRE_WIDTH": 0.25,
    "MASS": 1100,
    "LATERAL_TIRE_TO_CENTER": 0.815,
    "FRONT_WHEELBASE": 1.05234,
    "REAR_WHEELBASE": 1.4166,
    "MODEL_PATH": 'test/vehicle.glb',
    "MODEL_SCALE": (1, 1, 1),
    "MODEL_ROTATE": (0, 0.075, 0.),
    "MODEL_SHIFT": (0, 0, 0),
    "LENGTH": 4.515,
    "HEIGHT": 1.139,
    "WIDTH": 1.852
}
env_config={
    "manual_control": True,
    "use_render": True,
    # "controller": "keyboard",  # or joystick
    "window_size": (1600, 1100),
    "start_seed": 1000,
    "test_asset_meta_info": asset_metainfo
    # "map": "COT",
    # "environment_num": 1,
}

# Your environment initialization
env = TestAssetMetaDriveEnv(config=env_config)
o, _ = env.reset()


def update_value(name, value):
    """Updates the asset_metainfo with the new value."""
    asset_metainfo[name] = float(value)


def environment_step():
    """Steps through the environment with the updated asset_metainfo configuration and schedules itself to run again."""
    # print(asset_metainfo)
    o, r, tm, tc, info = env.step(test_asset_config_dict=asset_metainfo,
                                  actions={"default_agent": [0, 0], "test_agent": [0, 0]})
    root.after(10, environment_step)  # schedule the function to run again after 10 milliseconds


def main():
    global root
    root = tk.Tk()
    root.title("Asset MetaInfo Updater")

    # Create sliders for each asset metainfo property
    for idx, (key, value) in enumerate(asset_metainfo.items()):
        if isinstance(value, (int, float)):
            # Creating a label
            ttk.Label(root, text=key).grid(row=idx, column=0)

            # Creating a slider
            slider = ttk.Scale(root, from_=0, to=value * 2, value=value,
                               command=lambda v, key=key: update_value(key, v))
            slider.grid(row=idx, column=1)

        elif isinstance(value, tuple):
            # Creating sliders for each component of tuple values
            for sub_idx, val in enumerate(value):
                sub_key = f"{key}_{sub_idx}"

                # Creating a label
                ttk.Label(root, text=sub_key).grid(row=idx + sub_idx, column=0)

                # Creating a slider
                slider = ttk.Scale(root, from_=0, to=val * 2, value=val,
                                   command=lambda v, key=sub_key: update_value(key, v))
                slider.grid(row=idx + sub_idx, column=1)

    # Schedule the initial call to environment_step
    root.after(10, environment_step)

    # Run the GUI loop
    root.mainloop()


if __name__ == "__main__":
    main()

#
# from metadrive.envs.test_asset_metadrive_env import TestAssetMetaDriveEnv
# from metadrive.examples import expert
#
# # Set the envrionment config
# asset_metainfo = {
#     "TIRE_RADIUS": 0.313,
#     "TIRE_WIDTH": 0.25,
#     "MASS": 1100,
#     "LATERAL_TIRE_TO_CENTER": 0.815,
#     "FRONT_WHEELBASE": 1.05234,
#     "REAR_WHEELBASE": 1.4166,
#     "MODEL_PATH": 'test/vehicle.glb',
#     "MODEL_SCALE": (1, 1, 1),
#     "MODEL_ROTATE": (0, 0.075, 0.),
#     "MODEL_SHIFT": (0, 0, 0),
#     "LENGTH": 4.515,
#     "HEIGHT": 1.139,
#     "WIDTH": 1.852
# }
# env_config={
#     "manual_control": True,
#     "use_render": True,
#     # "controller": "keyboard",  # or joystick
#     "window_size": (1600, 1100),
#     "start_seed": 1000,
#     "test_asset_meta_info": asset_metainfo
#     # "map": "COT",
#     # "environment_num": 1,
# }
#
#
# # ===== Setup the training environment =====
# env = TestAssetMetaDriveEnv(config=env_config)
#
# o, _ = env.reset()
# # env.vehicle.expert_takeover = True
# count = 1
# for i in range(1, 9000000000):
#     o, r, tm, tc, info = env.step(test_asset_config_dict=asset_metainfo,actions={"default_agent": [0,0], "test_agent":[0,0]})