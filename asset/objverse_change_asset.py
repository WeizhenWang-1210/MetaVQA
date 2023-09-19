import tkinter as tk
from tkinter import ttk
from functools import partial
from metadrive.envs.test_asset_metadrive_env import TestAssetMetaDriveEnv
import json
import os
class AssetMetaInfoUpdater:
    def __init__(self, model_path, save_path=None):
        self.asset_metainfo = {
            "TIRE_RADIUS": 0.313,
            "TIRE_WIDTH": 0.25,
            "MASS": 1100,
            "LATERAL_TIRE_TO_CENTER": 0.815,
            "FRONT_WHEELBASE": 1.05234,
            "REAR_WHEELBASE": 1.4166,
            "MODEL_PATH": 'test/vehicle.glb',
            "MODEL_SCALE": (1, 1, 1),
            "MODEL_OFFSET": (0.1, 0.1, 0.1),
            "MODEL_HPR": (0, 0, 0),
            "LENGTH": 4.515,
            "HEIGHT": 1.139,
            "WIDTH": 1.852
        }
        self.RANGE_SPECS = {
            "TIRE_RADIUS": (0.01, 1),
            "TIRE_WIDTH": (0.01, 1),
            "LATERAL_TIRE_TO_CENTER": (-2, 2),
            "FRONT_WHEELBASE": (-2, 2),
            "REAR_WHEELBASE": (-2, 2),
            "MODEL_SCALE": (-2, 2),
            "MODEL_OFFSET": (-4, 4),
            "MODEL_HPR": (-180, 180)  # Assuming the model's heading, pitch, roll are in degrees.
            # ... add other attributes as needed
        }
        if save_path and os.path.exists(save_path):
            with open(save_path, 'r') as file:
                loaded_metainfo = json.load(file)
                self.asset_metainfo.update(loaded_metainfo)  # update the asset_metainfo with the values from the file
                self.asset_metainfo["MODEL_SCALE"] = tuple(self.asset_metainfo["MODEL_SCALE"])
                self.asset_metainfo["MODEL_OFFSET"] = tuple(self.asset_metainfo["MODEL_OFFSET"])
                self.asset_metainfo["MODEL_SCALE"] = tuple(self.asset_metainfo["MODEL_SCALE"])
        self.env_config = {
            "manual_control": True,
            "use_render": True,
            "window_size": (1600, 1100),
            "start_seed": 1000,
            "test_asset_meta_info": self.asset_metainfo
        }
        self.entries = {}
        self.env_config["test_asset_meta_info"]["MODEL_PATH"] = model_path
        self.env = TestAssetMetaDriveEnv(config=self.env_config)
        o, _ = self.env.reset()
        self.root = tk.Tk()
        self.root.title("Asset MetaInfo Updater")
        self.setup_ui()
        self.save_path = save_path


    def slider_command(self, v, key, idx=None):
        self.update_value(key, v, idx)

    def update_value(self, name, value, index=None):
        """Updates the asset_metainfo with the new value."""
        constraints = {
            "TIRE_RADIUS": lambda x: x if x > 0 else 0.01,
            "TIRE_WIDTH": lambda x: x if x > 0 else 0.01,
            "MODEL_SCALE": lambda x: x if x != 0 else 1,
        }

        if name.startswith("MODEL_SCALE_"):
            idx = int(name.split("_")[-1])
            current_tuple = list(self.asset_metainfo["MODEL_SCALE"])
            new_val = float(value)
            if "MODEL_SCALE" in constraints:
                new_val = constraints["MODEL_SCALE"](new_val)
            current_tuple[idx] = new_val
            self.asset_metainfo["MODEL_SCALE"] = tuple(current_tuple)
        elif  name.startswith("MODEL_OFFSET_"):
            idx = int(name.split("_")[-1])
            current_tuple = list(self.asset_metainfo["MODEL_OFFSET"])
            new_val = float(value)
            if "MODEL_OFFSET" in constraints:
                new_val = constraints["MODEL_OFFSET"](new_val)
            current_tuple[idx] = new_val
            self.asset_metainfo["MODEL_OFFSET"] = tuple(current_tuple)
        elif  name.startswith("MODEL_HPR_"):
            idx = int(name.split("_")[-1])
            current_tuple = list(self.asset_metainfo["MODEL_HPR"])
            new_val = float(value)
            if "MODEL_HPR" in constraints:
                new_val = constraints["MODEL_HPR"](new_val)
            current_tuple[idx] = new_val
            self.asset_metainfo["MODEL_HPR"] = tuple(current_tuple)
        else:
            if isinstance(self.asset_metainfo[name], tuple):
                new_val = [float(value)] * 3
                self.asset_metainfo[name] = tuple(new_val)
            else:
                new_val = float(value)
                if name in constraints:
                    new_val = constraints[name](new_val)
                self.asset_metainfo[name] = new_val

        # Update the corresponding StringVar in self.entries
        if name in self.entries:
            if isinstance(self.entries[name], dict):  # Check if it's a dictionary
                for idx, entry_var in self.entries[name].items():
                    entry_var.set(value)
            else:
                self.entries[name].set(value)
        elif "MODEL_SCALE" in name:
            if name == "MODEL_SCALE":
                for idx in range(3):
                    self.entries[f"MODEL_SCALE_{idx}"].set(value)
            else:
                if isinstance(self.entries[name], dict):  # Check if it's a dictionary
                    for idx, entry_var in self.entries[name].items():
                        entry_var.set(value)
                else:
                    self.entries[name].set(value)

    def environment_step(self):
        """Steps through the environment with the updated asset_metainfo configuration and schedules itself to run again."""
        o, r, tm, tc, info = self.env.step(test_asset_config_dict=self.asset_metainfo,
                                      actions={"default_agent": [0, 0], "test_agent": [0, 0]})
        self.root.after(10, self.environment_step)  # schedule the function to run again after 10 milliseconds

    def setup_ui(self):
        for key, value in self.asset_metainfo.items():
            frame = tk.Frame(self.root)
            frame.pack(fill=tk.X, padx=10, pady=5)

            min_val, max_val = self.RANGE_SPECS.get(key, (0, value * 2 if not isinstance(value, tuple) else 2))

            if isinstance(value, (int, float)):
                ttk.Label(frame, text=key).pack(side=tk.LEFT)
                slider_cmd = partial(self.slider_command, key=key)
                slider = ttk.Scale(frame, from_=min_val, to=max_val, value=value, command=slider_cmd, length=700,
                                   orient=tk.HORIZONTAL)
                slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

                # Create an entry next to the slider
                entry_var = tk.StringVar()
                entry_var.set(str(value))
                entry = ttk.Entry(frame, textvariable=entry_var, width=8)
                entry.pack(side=tk.RIGHT, padx=5)
                entry.bind('<Return>', partial(self.entry_command, key=key, entry_var=entry_var))
                self.entries[key] = entry_var

            elif isinstance(value, tuple):

                if key == "MODEL_SCALE":
                    # Slider to control all 3 elements together
                    sub_frame_all = tk.Frame(self.root)
                    sub_frame_all.pack(fill=tk.X, padx=10, pady=5)
                    # For controlling all 3 elements together
                    ttk.Label(sub_frame_all, text=key).pack(side=tk.LEFT)
                    slider_cmd = partial(self.slider_command, key=key)
                    slider = ttk.Scale(sub_frame_all, from_=min_val, to=max_val, value=value[0], command=slider_cmd,
                                       length=700, orient=tk.HORIZONTAL)
                    slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                    entry_var = tk.StringVar()
                    entry_var.set(str(value[0]))  # you might consider showing all 3 values in some formatted way
                    entry = ttk.Entry(sub_frame_all, textvariable=entry_var, width=8)
                    entry.pack(side=tk.RIGHT, padx=5)
                    entry.bind('<Return>', partial(self.entry_command, key=key, entry_var=entry_var))
                    self.entries[key] = entry_var  # This is a flattened structure

                    # Individual sliders for each tuple element
                    for sub_idx, val in enumerate(value):
                        sub_key = f"{key}_{sub_idx}"
                        sub_frame = tk.Frame(self.root)
                        sub_frame.pack(fill=tk.X, padx=10, pady=5)
                        ttk.Label(sub_frame, text=sub_key).pack(side=tk.LEFT)
                        slider_cmd = partial(self.slider_command, key=sub_key)  # Change here
                        slider = ttk.Scale(sub_frame, from_=min_val, to=max_val, value=val, command=slider_cmd,
                                           length=700, orient=tk.HORIZONTAL)
                        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                        entry_var = tk.StringVar()
                        entry_var.set(str(val))
                        entry = ttk.Entry(sub_frame, textvariable=entry_var, width=8)
                        entry.pack(side=tk.RIGHT, padx=5)
                        entry.bind('<Return>',
                                   partial(self.entry_command, key=sub_key, entry_var=entry_var))  # Change here
                        self.entries[sub_key] = entry_var  # This will create keys like MODEL_SCALE_0, MODEL_SCALE_1,
                else:
                    for sub_idx, val in enumerate(value):
                        sub_key = f"{key}_{sub_idx}"
                        sub_frame = tk.Frame(self.root)
                        sub_frame.pack(fill=tk.X, padx=10, pady=5)
                        ttk.Label(sub_frame, text=sub_key).pack(side=tk.LEFT)
                        slider_cmd = partial(self.slider_command, key=sub_key)  # Change here
                        slider = ttk.Scale(sub_frame, from_=min_val, to=max_val, value=val, command=slider_cmd,
                                           length=700, orient=tk.HORIZONTAL)
                        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                        entry_var = tk.StringVar()
                        entry_var.set(str(val))
                        entry = ttk.Entry(sub_frame, textvariable=entry_var, width=8)
                        entry.pack(side=tk.RIGHT, padx=5)
                        entry.bind('<Return>',
                                   partial(self.entry_command, key=sub_key, entry_var=entry_var))  # Change here
                        self.entries[sub_key] = entry_var  # This will create keys like MODEL_SCALE_0, MODEL_SCALE_1,
        save_button = ttk.Button(self.root, text="Save", command=self.save_metainfo_to_json)
        save_button.pack(pady=10)
    def slider_command(self, v, key, idx=None):
        self.update_value(key, v, idx)
        if idx is not None:
            self.entries[key][idx].set(v)  # Update the entry with the new slider value for tuple
        else:
            self.entries[key].set(v)  # Update the entry with the new slider value for single value

    def entry_command(self, event, key, entry_var, idx=None):
        value = entry_var.get()
        try:
            float_val = float(value)
            self.update_value(key, value, idx)
            # Assuming you keep a reference to your slider widgets, you can also set the slider value here.
        except ValueError:
            print("Invalid value entered.")
    def save_metainfo_to_json(self):
        """Save the modified MetaInfo to a JSON file."""
        if self.save_path is not None:
            with open(self.save_path, 'w') as file:
                json.dump(self.asset_metainfo, file)
        self.root.destroy()
    def run(self):
        self.root.after(10, self.environment_step)
        self.root.mainloop()
        self.env.close()

if __name__ == "__main__":
   model_path_input = 'test/vehicle.glb'
   updater = AssetMetaInfoUpdater(model_path_input)
   updater.run()
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