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
    destination_folder = 'E:\\MetaVQA\metadrive\\assets\models\\test'  # Folder where you want to copy the files
    json_path = 'F:\\metaasset\\hf-objaverse-v1\\matched_uids.json'
    save_path_folder = 'E:\MetaVQA\\asset'
    src_parent_folder = 'F:\\metaasset\\hf-objaverse-v1'
    data = load_json(json_path)
    for uid, relative_path in data.items():
        copied_model_path = copy_file(uid, src_parent_folder, relative_path,
                                      destination_folder)  # This will now be your new model_path_input
        save_path = os.path.join(save_path_folder, f"{uid}.json")
        updater = AssetMetaInfoUpdater(copied_model_path, save_path)
        updater.run()