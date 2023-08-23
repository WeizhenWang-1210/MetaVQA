import time
import objaverse
import os
import trimesh
import subprocess
import tkinter as tk
from tkinter import simpledialog, messagebox
import json


class Objverse_filter_asset:
    def __init__(self, asset_folder_path="f:\\metaasset\\hf-objaverse-v1"):
        self.asset_folder_path = asset_folder_path
        self.json_gz_path = os.path.join(asset_folder_path, "object-paths.json.gz")
        self.json_path = os.path.join(asset_folder_path, "object-paths.json")
        self.cached_asset_uids = []
        self.processed_uids_path = os.path.join(asset_folder_path, "processed_uids.json")
        self.saved_uids_path = os.path.join(asset_folder_path, "saved_uids.json")
        self.current_session_tags = set()
    def load_cached_uids(self):
        uids = []

        for dirpath, dirnames, filenames in os.walk(self.asset_folder_path):
            if "glbs" in dirpath and any('-' in d and d.split('-')[0].isdigit() and d.split('-')[1].isdigit() for d in
                                         dirpath.split(os.sep)):
                for filename in filenames:
                    if filename.endswith('.glb'):
                        uid = filename.replace('.glb', '')
                        uids.append(uid)

        print(len(uids))
        return uids

    def load_processed_uids(self, filename=None):
        if filename is None:
            filename = self.processed_uids_path
        if os.path.exists(filename):
            with open(filename, "r") as file:
                return json.load(file)
        return []

    def save_processed_uid(self, uid, filename=None):
        if filename is None:
            filename = self.processed_uids_path
        processed_uids = self.load_processed_uids(filename)
        if uid not in processed_uids:
            processed_uids.append(uid)
            with open(filename, "w") as file:
                json.dump(processed_uids, file)

    def get_existing_tags(self, filename=None):
        if filename is None:
            filename = self.saved_uids_path

        existing_tags = set()

        if os.path.exists(filename):
            with open(filename, "r") as file:
                saved_assets = json.load(file)
            for _, tag_list in saved_assets.items():
                for tag in tag_list:
                    existing_tags.add(tag)

        # Merge with current session tags
        merged_tags = list(existing_tags.union(self.current_session_tags))
        return merged_tags

    def get_user_input(self):
        root = tk.Tk()
        root.title("Mesh Input")

        tk.Label(root, text="Use this mesh?").pack(pady=10)

        # y/n/q Radiobuttons
        choice = tk.StringVar()
        tk.Radiobutton(root, text="Yes", variable=choice, value="y").pack(anchor="w")
        tk.Radiobutton(root, text="No", variable=choice, value="n").pack(anchor="w")
        tk.Radiobutton(root, text="Quit", variable=choice, value="q").pack(anchor="w")

        tk.Label(root, text="Enter tags separated by commas:").pack(pady=10)
        tags_entry = tk.Entry(root, width=50)
        tags_entry.pack(pady=5)

        # Display a list of previously specified tags
        tk.Label(root, text="Or select previous tags:").pack(pady=10)
        tags_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
        tags_listbox.pack(pady=5)

        existing_tags = self.get_existing_tags()
        for tag in existing_tags:
            tags_listbox.insert(tk.END, tag)

        def submit_and_destroy():
            root.choice_value = choice.get()

            selected_tags = [tags_listbox.get(i) for i in tags_listbox.curselection()]
            input_tags = [tag.strip() for tag in tags_entry.get().split(",")]

            # Update the current session tags
            self.current_session_tags.update(input_tags)

            root.tags_value = selected_tags + input_tags

            root.destroy()

        tk.Button(root, text="Submit", command=submit_and_destroy).pack(pady=20)

        root.mainloop()

        user_input = root.choice_value
        tags = root.tags_value if user_input == "y" else []

        return user_input, tags

    def filter_uid_raw(self, uids):
        objects = objaverse.load_objects(uids=uids)
        saved_assets = {}  # This will contain UID:tags key-value pairs
        processed_uids = self.load_processed_uids()
        count, total = -1, len(uids)
        for uid, mesh_path in objects.items():
            count += 1
            if uid in processed_uids:
                print("Ignore uid #{}\{}: {}".format(count, total, uid))
                continue
            print("Processing uid #{}\{}: {}".format(count, total, uid))
            p = subprocess.Popen(["python", "show_mesh.py", mesh_path])
            time.sleep(2)
            user_input, tags = self.get_user_input()
            p.terminate()

            if user_input == "q":
                break
            elif user_input == "y":
                saved_assets[uid] = tags  # Save tags for this UID
                self.saved_assets_to_json({uid: tags})  # Save this asset immediately

            self.save_processed_uid(uid)  # Save this uid as processed

        return saved_assets

    def saved_assets_to_json(self, saved_assets, filename=None):
        if filename is None:
            filename = self.saved_uids_path

        # Load existing data
        existing_data = {}
        if os.path.exists(filename):
            with open(filename, "r") as file:
                existing_data = json.load(file)

        # Merge the new data with the existing data
        for uid, tags in saved_assets.items():
            existing_data[uid] = tags  # This will overwrite tags for existing UIDs and add new ones

        # Save the merged data back to the JSON file
        with open(filename, "w") as file:
            json.dump(existing_data, file)


if __name__ == "__main__":
    objaverse_filter_helper = Objverse_filter_asset()
    cached_uid_lists = objaverse_filter_helper.load_cached_uids()
    saved_assets = objaverse_filter_helper.filter_uid_raw(cached_uid_lists)

    # objaverse_filter_helper.saved_assets_to_json(saved_assets)
