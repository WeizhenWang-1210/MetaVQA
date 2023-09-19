from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.static_object.test_new_object import TestObject

import os
import json
import tkinter as tk
import math
class AssetAdjuster:
    def __init__(self, env_config, folder_path, save_path):
        self.env = MetaDriveEnv(env_config)
        self.folder_path = folder_path
        self.env.reset()
        self.current_obj = None
        self.save_path = save_path
        self.t = 0
        self.load_and_display_objects()
    @staticmethod
    def load_json_file(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def load_and_display_objects(self):
        # Check if the save file exists
        if not os.path.exists(self.save_path):
            print(f"Save file '{self.save_path}' does not exist.")
            return

        # Load the saved data
        with open(self.save_path, 'r') as f:
            saved_data = json.load(f)

        # Spawn each object using saved parameters
        for filepath, params in saved_data.items():
            if filepath == self.save_path:
                continue
            print(filepath)
            asset_metainfo = self.load_json_file(filepath)
            x = params['x']
            y = params['y']
            theta = params['theta']
            self.env.engine.spawn_object(TestObject, position=[x, y], heading_theta=theta, random_seed=1, force_spawn=True, asset_metainfo=asset_metainfo)

        self.env.capture("spawned_obj_{}.jpg".format(self.t))

    def spawn_object(self, x, y, theta, asset_metainfo):
        if self.current_obj is not None:
            self.env.engine.clear_objects([self.current_obj.id], force_destroy=True)

        self.current_obj = self.env.engine.spawn_object(TestObject, position=[x, y], heading_theta=theta, random_seed=1, force_spawn=True, asset_metainfo=asset_metainfo)

    def load_saved_values(self, filepath):
        output_filename = self.save_path
        if os.path.exists(output_filename):
            with open(output_filename, 'r') as f:
                data = json.load(f)
                return data.get(filepath, {})
        return {}
    def adjust_parameters(self, filepath):
        asset_metainfo = self.load_json_file(filepath)
        saved_values = self.load_saved_values(filepath)

        def on_scale_change(val, entry):
            entry.delete(0, tk.END)
            entry.insert(0, val)
            x = x_scale.get()
            y = y_scale.get()
            theta = theta_scale.get()
            self.spawn_object(x, y, theta, asset_metainfo)

        def on_entry_change(entry, scale):
            try:
                scale_val = float(entry.get())
                scale.set(scale_val)
            except ValueError:
                pass  # You may want to provide a user feedback about invalid input
        def update_env():
            self.t = self.t + 1
            o, r, tm, tc, info = self.env.step([0, 0])
            self.env.capture("spawned_obj_{}.jpg".format(self.t))
            root.after(50, update_env)  # Calls itself every 50 milliseconds. Adjust this interval as needed.


        root = tk.Tk()
        root.title("Adjust parameters for " + filepath)

        def on_submit():
            x_val = x_scale.get()
            y_val = y_scale.get()
            theta_val = theta_scale.get()
            self.save_to_json(x_val, y_val, theta_val, filepath)
            self.current_obj = None  # Prevent clearing the object when spawning next
            root.destroy()

        submit_btn = tk.Button(root, text="Submit", command=on_submit)
        submit_btn.pack(pady=20)
        x_default = saved_values.get('x', 10)
        y_default = saved_values.get('y', -5)
        theta_default = saved_values.get('theta', 0)

        x_scale = tk.Scale(root, from_=-100, to=100, orient=tk.HORIZONTAL, length=300, sliderlength=30)
        x_scale.set(x_default)
        x_scale.pack(pady=20)

        x_entry = tk.Entry(root)
        x_entry.insert(0, x_scale.get())
        x_entry.bind('<Return>', lambda e: on_entry_change(x_entry, x_scale))
        x_scale.config(command=lambda val: on_scale_change(val, x_entry))
        x_entry.pack(pady=10)

        y_scale = tk.Scale(root, from_=-50, to=50, orient=tk.HORIZONTAL, length=300, sliderlength=30)
        y_scale.set(y_default)
        y_scale.pack(pady=20)

        y_entry = tk.Entry(root)
        y_entry.insert(0, y_scale.get())
        y_entry.bind('<Return>', lambda e: on_entry_change(y_entry, y_scale))
        y_scale.config(command=lambda val: on_scale_change(val, y_entry))
        y_entry.pack(pady=10)

        theta_scale = tk.Scale(root, from_=-math.pi, to=math.pi, orient=tk.HORIZONTAL, label="Heading Theta", length=300, sliderlength=30, resolution=0.01)
        theta_scale.set(theta_default)
        theta_scale.pack(pady=20)

        theta_entry = tk.Entry(root)
        theta_entry.insert(0, theta_scale.get())
        theta_entry.bind('<Return>', lambda e: on_entry_change(theta_entry, theta_scale))
        theta_scale.config(command=lambda val: on_scale_change(val, theta_entry))
        theta_entry.pack(pady=10)

        update_env()  # Start the loop to update the environment
        root.mainloop()

    def process_folder(self):
        for file in os.listdir(self.folder_path):
            if file.endswith(".json"):
                self.adjust_parameters(os.path.join(self.folder_path, file))

    def save_to_json(self, x, y, theta, filepath):
        # Name of the output file
        output_filename = self.save_path

        # Load existing data
        if os.path.exists(output_filename):
            with open(output_filename, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Update data for the current filepath
        data[filepath] = {
            'x': x,
            'y': y,
            'theta': theta
        }

        # Write updated data back to the JSON file
        with open(output_filename, 'w') as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    env_config = {
        "num_scenarios": 1,
        "traffic_density": 1,
        "traffic_mode": "hybrid",
        "start_seed": 22,
        "debug": False,
        "cull_scene": False,
        "manual_control": False,
        "use_render": True,  # Set your render value here
        "decision_repeat": 5,
        "need_inverse_traffic": False,
        "rgb_clip": True,
        "map": "X",
        "random_traffic": False,
        "random_lane_width": True,
        "driving_reward": 1.0,
        "force_destroy": False,
        "window_size": (2400, 1600),
        "vehicle_config": {
            "enable_reverse": False,
        },
    }


    # Replace 'your_folder_path_here' with the path to your JSON folder
    folder_path = "C:\\research\\gitplay\\MetaVQA\\asset"
    save_path =  "C:\\research\\gitplay\\MetaVQA\\asset\\spawn_object.json"
    adjuster = AssetAdjuster(env_config, folder_path, save_path)
    adjuster.process_folder()

