"""
This script is used to display and review Objaverse assets downloaded using download_assets.py.
It provides functionality for deciding whether to use certain assets and for saving this selection,
along with any annotations (like tags), for later reference.
The script offers a graphical user interface (GUI) for this review and annotation process.

Classes:
- Objverse_filter_asset:
  - __init__: Initializes the Objverse_filter_asset instance with configuration options.
  - load_cached_uids: Loads UIDs of assets that have been downloaded and cached locally.
  - load_processed_uids: Loads UIDs that have already been processed.
  - save_processed_uid: Saves a UID to the list of processed UIDs.
  - get_existing_tags: Retrieves list of annotated tags previously assigned to assets.
  - get_user_input: Captures user input for asset selection and annotation.
  - filter_uid_raw: Processes and filters UIDs based on user input, annotating them.
  - saved_assets_to_json: Saves annotated assets to a JSON file.
  - get_tags_selection: Captures annotation tag selections from the user through a GUI interface.
  - get_uids_by_tags: Retrieves UIDs that match specific annotations.
  - save_matched_uids_to_json: Saves UIDs with matched annotations to a JSON file.
"""
import time
import objaverse
import os
import subprocess
import tkinter as tk
import json
import yaml
from asset.read_config import configReader

class Objverse_filter_asset:
    def __init__(
                    self,
                    isTagList = False,
                    asset_folder_path="./objaverse/hf-objaverse-v1",
                    raw_asset_path_folder = "./raw_asset_path_folder",
                    filter_asset_path_folder = "./filter_asset_path_folder",
                    match_uid_path_folder = "./match_uid_path_folder",
                    tag = "Test",

                 ):
        """
        Initialize the Objverse_filter_asset instance.

        Parameters:
        - isTagList (bool): Determines if a different JSON naming convention is used (a single tag or tag list).
        - asset_folder_path (str): Base directory where assets are stored.
        - raw_asset_path_folder (str): Folder where object path json are stored.
        - filter_asset_path_folder (str): Folder to save processed and saved UIDs.
        - match_uid_path_folder (str): Folder to store matched UIDs.
        - tag (str): Tag for saving resulting JSON files.

        Returns:
        - None
        """
        # Base directory for assets
        self.asset_folder_path = asset_folder_path

        # Determine which JSON naming convention to use
        self.isTagList = isTagList
        # json path is saved from download_assets.py. It is a json file containing asset uids and asset paths.
        if self.isTagList:
            self.json_path = os.path.join(raw_asset_path_folder, "object-list-paths-{}.json".format(tag))
        else:
            self.json_path = os.path.join(raw_asset_path_folder, "object-paths-{}.json".format(tag))
        if not os.path.exists(filter_asset_path_folder):
            os.mkdir(filter_asset_path_folder)
        # Path to save all the uids you have taken a look, avoid to re-check them in the future
        self.processed_uids_path = os.path.join(filter_asset_path_folder, "processed_uids_{}.json".format(tag))
        # Path to save the uids you want to use later along with corresponding annotation you write
        self.saved_uids_path = os.path.join(filter_asset_path_folder, "saved_uids_{}.json".format(tag))
        if not os.path.exists(match_uid_path_folder):
            os.mkdir(match_uid_path_folder)
        self.match_uid_path_folder = match_uid_path_folder
        self.cached_asset_uids = []
        self.current_session_tags = set()
    def load_cached_uids(self):
        """
        Go over the asset folder, and then find asset file within each subfolder.
        Save asset's UID and return
        The asset folder should follow the Objaverse standard
        Example: hf-objaverse-v1/123456/7zfedb.glb

        Returns:
        - list[str]: UIDs of assets that have been downloaded and cached by objaverse on local.
        """
        uids = []
        print(self.json_path)
        with open(self.json_path, "r") as f:
            uid_dict = json.load(f)
        for dirpath, dirnames, filenames in os.walk(self.asset_folder_path):
            if "glbs" in dirpath and any('-' in d and d.split('-')[0].isdigit() and d.split('-')[1].isdigit() for d in
                                         dirpath.split(os.sep)):
                for filename in filenames:
                    if filename.endswith('.glb'):
                        uid = filename.replace('.glb', '')
                        if uid in uid_dict.keys():
                            uids.append(uid)

        print(len(uids))
        return uids

    def load_processed_uids(self, filename=None):
        """
        Load UIDs that have been processed from the saved json file.

         Parameters:
        - filename (str, optional): Filename of the JSON file containing processed UIDs.

        Returns:
        - list[str]: A list of processed UIDs.
        """
        if filename is None:
            filename = self.processed_uids_path
        if os.path.exists(filename):
            with open(filename, "r") as file:
                return json.load(file)
        return []

    def save_processed_uid(self, uid, filename=None):
        """
        Saves a UID to the list of processed UIDs in a JSON file.

        Parameters:
        - uid (str): UID to save.
        - filename (str, optional): Filename of the JSON file where processed UIDs are saved.

        Returns:
        - None
        """
        if filename is None:
            filename = self.processed_uids_path
        processed_uids = self.load_processed_uids(filename)
        if uid not in processed_uids:
            processed_uids.append(uid)
            with open(filename, "w") as file:
                json.dump(processed_uids, file)

    def get_existing_tags(self, filename=None):
        """
        Retrieves previously assigned tags from a saved JSON file.

        Parameters:
        - filename (str, optional): Filename of the JSON file containing saved tags.

        Returns:
        - list[str]: List of tags from the saved JSON file.
        """
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
        """
        Captures user input for asset selection and tagging through a GUI interface.

        Returns:
        - str: User's choice (y/n/q).
        - list[str]: List of tags specified or selected by the user.
        """
        root = tk.Tk()
        root.title("Mesh Input")

        tk.Label(root, text="Use this mesh?").pack(pady=10)

        # y/n/q Radiobuttons
        choice = tk.StringVar()
        choice = tk.StringVar()
        yes_button = tk.Radiobutton(root, text="Yes", variable=choice, value="y")
        no_button = tk.Radiobutton(root, text="No", variable=choice, value="n")
        quit_button = tk.Radiobutton(root, text="Quit", variable=choice, value="q")
        yes_button.pack(anchor="w")
        no_button.pack(anchor="w")
        quit_button.pack(anchor="w")

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

        def on_key_press(event):
            if event.char == 'y' or event.char == 'Y':
                choice.set('y')
            elif event.char == 'n' or event.char == 'N':
                choice.set('n')
            elif event.char == 'q' or event.char == 'Q':
                choice.set('q')

        def on_enter_press(event):
            submit_and_destroy()
        # Bind the keys to the root window
        root.bind('<KeyPress-Y>', on_key_press)
        root.bind('<KeyPress-y>', on_key_press)
        root.bind('<KeyPress-N>', on_key_press)
        root.bind('<KeyPress-n>', on_key_press)
        root.bind('<KeyPress-Q>', on_key_press)
        root.bind('<KeyPress-q>', on_key_press)
        root.bind('<Return>', on_enter_press)
        root.mainloop()

        user_input = root.choice_value
        tags = root.tags_value if user_input == "y" else []

        return user_input, tags

    def filter_uid_raw(self, uids):
        """
        Processes and filters UIDs based on user decision and annotating them with user's tags.

        Parameters:
        - uids (list[str]): List of UIDs to process.

        Returns:
        - dict: Dictionary with UIDs as keys and lists of tags as values.
        """
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
        """
        Saves annotated assets to a JSON file.

        Parameters:
        - saved_assets (dict): Dictionary with UIDs as keys and lists of tags as values.
        - filename (str, optional): Filename of the JSON file to save annotated assets.

        Returns:
        - None
        """
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


    def get_tags_selection(self):
        """
        Captures tag selections from the user through a GUI interface.

        Returns:
        - list[str]: List of tags selected by the user.
        """
        root = tk.Tk()
        root.title("Select Tags")

        tk.Label(root, text="Select tags:").pack(pady=10)
        tags_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
        tags_listbox.pack(pady=5)

        existing_tags = self.get_existing_tags()
        for tag in existing_tags:
            tags_listbox.insert(tk.END, tag)

        def submit_and_destroy():
            root.selected_tags = [tags_listbox.get(i) for i in tags_listbox.curselection()]
            root.destroy()

        tk.Button(root, text="Submit", command=submit_and_destroy).pack(pady=20)

        root.mainloop()

        return root.selected_tags

    def get_uids_by_tags(self, selected_tags):
        """
        Retrieves UIDs matching specific annotated tags.

        Parameters:
        - selected_tags (list[str]): List of tags to match.

        Returns:
        - dict: Dictionary with matched UIDs as keys and file paths as values.
        """
        # Load the saved UIDs with tags
        with open(self.saved_uids_path, "r") as file:
            saved_assets = json.load(file)

        matched_uids = {}

        # Iterate over saved UIDs and check for tag matches
        for uid, tags in saved_assets.items():
            if any(tag in selected_tags for tag in tags):
                # We'll look for the .glb file corresponding to this UID by searching the directory structure.
                for dirpath, dirnames, filenames in os.walk(self.asset_folder_path):
                    if "glbs" in dirpath and any(
                            '-' in d and d.split('-')[0].isdigit() and d.split('-')[1].isdigit() for d in
                            dirpath.split(os.sep)):
                        for filename in filenames:
                            if filename.endswith('.glb') and filename.replace('.glb', '') == uid:
                                matched_uids[uid] = os.path.relpath(os.path.join(dirpath, filename),
                                                                    self.asset_folder_path)

        return matched_uids


    def save_matched_uids_to_json(self, matched_uids, filename="matched_uids.json"):
        """
        Saves UIDs with matched annotations to a JSON file.

        Parameters:
        - matched_uids (dict): Dictionary with matched UIDs as keys and file paths as values.
        - filename (str): Filename of the JSON file to save matched UIDs.

        Returns:
        - None
        """
        full_path = os.path.join(self.match_uid_path_folder, filename)
        with open(full_path, "w") as file:
            json.dump(matched_uids, file)
        print(f"Matched UIDs and paths saved to {full_path}")


if __name__ == "__main__":
    config = configReader()
    path_config = config.loadPath()
    asset_folder_path = path_config["assetfolder"]
    raw_asset_path_folder = path_config["raw_asset_path_folder"]
    filter_asset_path_folder = path_config["filter_asset_path_folder"]
    match_uid_path_folder = path_config["match_uid_path_folder"]

    tag_config = config.loadTag()
    tag = tag_config["tag"]
    istaglist = tag_config["istaglist"]
    objaverse_filter_helper = Objverse_filter_asset(
        isTagList=istaglist,
        asset_folder_path= asset_folder_path,
        raw_asset_path_folder= raw_asset_path_folder,
        filter_asset_path_folder= filter_asset_path_folder,
        match_uid_path_folder = match_uid_path_folder,
        tag= tag,
    )

    # Filter and annotate each asset you downloaded.
    cached_uid_lists = objaverse_filter_helper.load_cached_uids()
    saved_assets = objaverse_filter_helper.filter_uid_raw(cached_uid_lists)

    # Select the annotaions you have made so far, and return all uids has a matched annotation
    selected_tags = objaverse_filter_helper.get_tags_selection()
    matched_uids = objaverse_filter_helper.get_uids_by_tags(selected_tags)

    # Save the matched UIDs and their paths to a JSON file
    filename = "matched_uids_{}.json".format("_".join(selected_tags))
    objaverse_filter_helper.save_matched_uids_to_json(matched_uids, filename=filename)
