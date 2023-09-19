import json
import os
import shutil
from asset.objverse_change_asset_static import StaticAssetMetaInfoUpdater
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def copy_file(uid, src_folder, src, dst_folder, tag):
    """
    Copy the source file to the destination folder and rename it using the uid.
    Returns the path to the copied file.
    """
    true_filename = os.path.basename(src)
    dst = f"{tag}-{uid}.glb"
    return_path = os.path.join(dst_folder, dst)
    shutil.copy(os.path.join(src_folder,src), return_path)
    return return_path

def delete_file(filepath):
    """
    Delete a file given its filepath.
    """
    if os.path.exists(filepath):
        os.remove(filepath)
    else:
        print(f"Error: {filepath} not found!")

def save_ignore_list(file_path, ignore_list):
    """Save the ignore list to a file."""
    with open(file_path, 'w') as file:
        json.dump(ignore_list, file)
if __name__ == "__main__":
    # tag = "bicycle"
    # destination_folder = 'C:\\research\\gitplay\\MetaVQA\\metadrive\\assets\\models\\test'  # Folder where you want to copy the files
    # json_path = "C:\\research\\dataset\\.objaverse\\hf-objaverse-v1\\matched_uids_{}.json".format(tag)
    # save_path_folder = 'C:\\research\\gitplay\\MetaVQA\\asset'
    # src_parent_folder = 'C:\\research\\dataset\\.objaverse\\hf-objaverse-v1'
    # ignore_list_path = 'C:\\research\\dataset\\.objaverse\\hf-objaverse-v1\\ignore_list_{}.json'.format(tag)
    tag = "car"
    destination_folder = 'C:\\research\\gitplay\\MetaVQA\\metadrive\\assets\\models\\test'  # Folder where you want to copy the files
    json_path = "C:\\research\\dataset\\hf-objaverse-v1\\matched_uids_{}.json".format(tag)
    save_path_folder = 'C:\\research\\gitplay\\MetaVQA\\asset'
    src_parent_folder = 'C:\\research\\dataset\\hf-objaverse-v1'
    ignore_list_path = 'C:\\research\\dataset\\hf-objaverse-v1\\ignore_list_{}.json'.format(tag)
    if os.path.exists(ignore_list_path):
        ignore_list = load_json(ignore_list_path)
    else:
        ignore_list = []
    data = load_json(json_path)
    for uid, relative_path in data.items():
        if uid in ignore_list:
            print(f"UID {uid} is in the ignore list. Skipping...")
            continue
        print("dealing with: {}".format(uid))
        copied_model_path = copy_file(uid, src_parent_folder, relative_path,
                                      destination_folder, tag=tag)  # This will now be your new model_path_input
        save_path = os.path.join(save_path_folder, f"{tag}-{uid}.json")
        updater = StaticAssetMetaInfoUpdater(os.path.basename(copied_model_path), save_path)
        # updater = StaticAssetMetaInfoUpdater("stop sign-8be31e33b3df4d6db7c75730ff11dfd8.glb", save_path)
        save_flag = updater.run()
        if not save_flag:
            delete_file(copied_model_path)
            ignore_list.append(uid)
            print(ignore_list)


        save_ignore_list(ignore_list_path, ignore_list)
    # for root, dirs, files in os.walk(destination_folder):
    #     for file_name in files:
    #         # Check if the file name contains the word
    #         if "car" in file_name:
    #             file_path = os.path.join(root, file_name)
    #             save_path = os.path.join(save_path_folder, f"{file_name}.json")
    #             updater = StaticAssetMetaInfoUpdater(file_name, save_path)
    #             save_flag = updater.run()
