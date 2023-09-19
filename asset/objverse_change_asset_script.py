import json
import os
import shutil
from asset.objverse_change_asset import AssetMetaInfoUpdater
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def copy_file(uid, src_folder, src, dst_folder):
    """
    Copy the source file to the destination folder and rename it using the uid.
    Returns the path to the copied file.
    """
    true_filename = os.path.basename(src)
    dst = f"{uid}.glb"
    assert dst == true_filename
    return_path = os.path.join("test", dst)
    shutil.copy(os.path.join(src_folder,src), os.path.join(dst_folder, dst))
    return return_path


if __name__ == "__main__":
    destination_folder = 'C:\\research\\gitplay\\MetaVQA\\metadrive\\assets\\models\\test'  # Folder where you want to copy the files
    json_path = "C:\\research\\dataset\\hf-objaverse-v1\\matched_uids.json"
    save_path_folder = 'C:\\research\\gitplay\\MetaVQA\\asset'
    src_parent_folder = 'C:\\research\\dataset\\hf-objaverse-v1'
    banned_list = ['234fa8768f1f48d18b558b029e294a93',
                   "2866efdfa943484391ef8313768e074d",
                   "58c912f75b5d49d0984d7c935d350526",
                   "fd33b16f18cd4d4b9786f8e55a5198c0",
                   "c7c7e9920ef5406192102e39ba6c0ac3"]
    data = load_json(json_path)
    for uid, relative_path in data.items():
        if uid in banned_list:
            continue
        print("dealing with: {}".format(uid))
        copied_model_path = copy_file(uid, src_parent_folder, relative_path,
                                      destination_folder)  # This will now be your new model_path_input
        save_path = os.path.join(save_path_folder, f"{uid}.json")
        updater = AssetMetaInfoUpdater(copied_model_path, save_path)
        updater.run()