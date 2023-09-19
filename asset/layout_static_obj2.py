from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.static_object.test_new_object import TestObject, TestGLTFObject

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
        self.files = [file for file in os.listdir(folder_path) if file.endswith(".json")]
        self.current_index = 0
    @staticmethod
    def load_json_file(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def load_and_display_objects(self):
        if not os.path.exists(self.save_path):
            print(f"Save file '{self.save_path}' does not exist.")
            return

        with open(self.save_path, 'r') as f:
            saved_data = json.load(f)

        for filepath, params_list in saved_data.items():
            asset_metainfo = self.load_json_file(filepath)
            for params in params_list:
                x = params['x']
                y = params['y']
                theta = params['theta']
                if "foldername" in asset_metainfo.keys() and asset_metainfo["foldername"] is not None:
                    self.env.engine.spawn_object(TestGLTFObject, position=[x, y], heading_theta=theta, random_seed=1,
                                                 force_spawn=True, asset_metainfo=asset_metainfo)
                else:
                    self.env.engine.spawn_object(TestObject, position=[x, y], heading_theta=theta, random_seed=1,
                                                 force_spawn=True, asset_metainfo=asset_metainfo)

    def add_same_asset(self, filepath):
        self.adjust_parameters(filepath)
    def spawn_object(self, x, y, theta, asset_metainfo):
        if self.current_obj is not None:
            self.env.engine.clear_objects([self.current_obj.id], force_destroy=True)
        if "foldername" in asset_metainfo.keys() and asset_metainfo["foldername"] is not None:
            self.current_obj = self.env.engine.spawn_object(TestGLTFObject, position=[x, y], heading_theta=theta,
                                                            random_seed=1, force_spawn=True,
                                                            asset_metainfo=asset_metainfo)
        else:
            self.current_obj = self.env.engine.spawn_object(TestObject, position=[x, y], heading_theta=theta,
                                                            random_seed=1, force_spawn=True,
                                                            asset_metainfo=asset_metainfo)

    def load_saved_values(self, filepath):
        output_filename = self.save_path
        if os.path.exists(output_filename):
            with open(output_filename, 'r') as f:
                data = json.load(f)
            # Get the last set of values for this filepath, or an empty dictionary if none exists
            return data.get(filepath, [{}])[-1]
        return {}
    def onlyStep(self, capture=False, pic_name_id=1):
        step = 0
        while True:
            o, r, tm, tc, info = self.env.step([0, 0])
            if capture:
                self.env.capture("spawned{}_obj_{}.jpg".format(pic_name_id, step))
            step += 1
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
            # self.env.capture("spawned_obj_{}.jpg".format(self.t))
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

        def on_submit_and_next():
            on_submit()
            self.current_index += 1
            self.process_next()

        def on_submit_and_add_same():
            on_submit()
            # No need to increment the current_index
            self.process_next()

        submit_and_next_btn = tk.Button(root, text="Submit and Next", command=on_submit_and_next)
        submit_and_next_btn.pack(pady=20)

        submit_and_add_same_btn = tk.Button(root, text="Submit and Add Same", command=on_submit_and_add_same)
        submit_and_add_same_btn.pack(pady=20)

        next_asset_btn = tk.Button(root, text="Next Asset", command=lambda: self.go_to_next_asset(root))
        next_asset_btn.pack(pady=20)
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

    def process_next(self):
        if self.current_index < len(self.files):
            print("================Dealing with: {}".format(self.files[self.current_index]))
            self.adjust_parameters(os.path.join(self.folder_path, self.files[self.current_index]))

    def process_folder(self):
        # Now this just initiates the process
        self.process_next()
    def go_to_next_asset(self, root):
        if self.current_obj is not None:
            self.env.engine.clear_objects([self.current_obj.id], force_destroy=True)
        self.current_obj = None
        root.destroy()
        self.current_index += 1
        self.process_next()

    def save_to_json(self, x, y, theta, filepath):
        output_filename = self.save_path
        if os.path.exists(output_filename):
            with open(output_filename, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Check if this filepath already exists in the data
        if filepath not in data:
            data[filepath] = []

        data[filepath].append({
            'x': x,
            'y': y,
            'theta': theta
        })

        # Write updated data back to the JSON file
        with open(output_filename, 'w') as f:
            json.dump(data, f, indent=4)



if __name__ == "__main__":
    env_config = {
        "num_scenarios": 1,
        "traffic_density": 0.6,
        "traffic_mode": "hybrid",
        "start_seed": 35,
        "debug": False,
        "cull_scene": False,
        "manual_control": True,
        "use_render": True,  # Set your render value here
        "decision_repeat": 5,
        "need_inverse_traffic": True,
        "rgb_clip": True,
        "render_pipeline": True,
        "map": "XCrRC",
        "random_traffic": True,
        "random_lane_width": True,
        "driving_reward": 1.0,
        "force_destroy": False,
        "window_size": (2400, 1600),
        "vehicle_config": {
            "enable_reverse": True,
            "show_navi_mark": False,
        },
        "traffic_vehicle_config": {
            "show_navi_mark": False,
            "show_dest_mark": False,
            "enable_reverse": False,
            "show_lidar": False,
            "show_lane_line_detector": False,
            "show_side_detector": False,
        },
    }


    # Replace 'your_folder_path_here' with the path to your JSON folder
    folder_path = "C:\\research\\gitplay\\MetaVQA\\asset"
    save_path =  "C:\\research\\gitplay\\MetaVQA\\asset\\spawn_object2.json"
    adjuster = AssetAdjuster(env_config, folder_path, save_path)
    # adjuster.process_folder()
    adjuster.onlyStep(capture=True, pic_name_id=2)
