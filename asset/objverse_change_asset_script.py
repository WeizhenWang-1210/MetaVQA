import json
import os
import shutil
from asset.objverse_change_asset import AssetMetaInfoUpdater
from asset.objverse_change_asset_static import StaticAssetMetaInfoUpdater
def load_json(file_path):
    """
    Load json file
    """
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
def delete_folder(folder_path):
    """
    Delete a folder and all its contents.

    :param folder_path: Path to the folder to be deleted.
    :return: None
    """
    # Check if the folder exists before attempting to delete
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder '{folder_path}' does not exist!")
def model_update(is_car_model, destination_folder, json_path, save_path_folder, src_parent_folder, ignore_list_path):
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
        # First copy the model into metadrive's model folder
        copied_model_path = copy_file(uid, src_parent_folder, relative_path,
                                      destination_folder, tag=tag)  # This will now be your new model_path_input
        save_path = os.path.join(save_path_folder, f"{tag}-{uid}.json")
        if is_car_model:
            updater = AssetMetaInfoUpdater("test/" + os.path.basename(copied_model_path), save_path)
        else:
            updater = StaticAssetMetaInfoUpdater(os.path.basename(copied_model_path), save_path)
        save_flag = updater.run()
        if not save_flag:
            delete_file(copied_model_path)
            ignore_list.append(uid)
            print(ignore_list)
        save_ignore_list(ignore_list_path, ignore_list)
def gltf_updater(destination_folder, save_path_folder, src_parent_folder, ignore_list_path):
    if os.path.exists(ignore_list_path):
        ignore_list = load_json(ignore_list_path)
    else:
        ignore_list = []
    for subfolder in os.listdir(src_parent_folder):
        subfolder_path = os.path.join(src_parent_folder, subfolder)
        if subfolder in ignore_list:
            print(f"Ignore folder {subfolder}")
        # Check if it's actually a directory and not a file
        if os.path.isdir(subfolder_path):
            gltf_file_path = os.path.join(subfolder_path, 'scene.gltf')
            # Check if the 'scene.gltf' file exists in this subfolder
            if os.path.exists(gltf_file_path):
                print(f"Found 'scene.gltf' in {subfolder}")
                save_path = os.path.join(save_path_folder, f"{tag}-{subfolder}.json")
                dest_subfolder_path = os.path.join(destination_folder, subfolder)
                if not os.path.exists(dest_subfolder_path):
                    shutil.copytree(subfolder_path, dest_subfolder_path)
                    print(f"Copied {subfolder} to {dest_subfolder_path}")
                updater = StaticAssetMetaInfoUpdater("scene.gltf", save_path, folder_name=subfolder, isGLTF=True)
            else:
                print(f"No 'scene.gltf' found in {subfolder}")
            save_flag = updater.run()
            if not save_flag:
                delete_folder(dest_subfolder_path)
                ignore_list.append(subfolder)
                print(ignore_list)
            save_ignore_list(ignore_list_path, ignore_list)


if __name__ == "__main__":
    # ===========================================Car Model=======================================
    # tag = "car"
    # # Folder where you want to copy the asset into, should be this path, otherwise metadrive won't recognize it.
    # destination_folder = 'C:\\research\\gitplay\\MetaVQA\\metadrive\\assets\\models\\test'  # Folder where you want to copy the files
    # # Assets you want to use, with their paths. Generated from objverse_filter_asset.py
    # json_path = "C:\\research\\dataset\\hf-objaverse-v1\\matched_uids_{}.json".format(tag)
    # # Folder where you want to save Adjusted parameters to
    # save_path_folder = 'C:\\research\\gitplay\\MetaVQA\\asset'
    # # Original asset parent folder (the path from above matched.json is relative)
    # src_parent_folder = 'C:\\research\\dataset\\hf-objaverse-v1'
    # # List of uids you want to ignore.
    # ignore_list_path = 'C:\\research\\dataset\\hf-objaverse-v1\\ignore_list_{}.json'.format(tag)
    #
    # model_update(is_car_model=True,
    #              destination_folder=destination_folder,
    #              json_path = json_path,
    #              save_path_folder= save_path_folder,
    #              src_parent_folder=src_parent_folder,
    #              ignore_list_path=ignore_list_path)
    # ===========================================Static Model=======================================
    # tag = "traffic light"
    # destination_folder = 'C:\\research\\gitplay\\MetaVQA\\metadrive\\assets\\models\\test'  # Folder where you want to copy the files
    # json_path = "C:\\research\\dataset2\\.objaverse\\hf-objaverse-v1\\matched_uids_{}.json".format(tag)
    # save_path_folder = 'C:\\research\\gitplay\\MetaVQA\\asset'
    # src_parent_folder = 'C:\\research\\dataset2\\.objaverse\\hf-objaverse-v1'
    # ignore_list_path = 'C:\\research\\dataset2\\.objaverse\\hf-objaverse-v1\\ignore_list_{}.json'.format(tag)
    # model_update(is_car_model=False,
    #              destination_folder=destination_folder,
    #              json_path = json_path,
    #              save_path_folder= save_path_folder,
    #              src_parent_folder=src_parent_folder,
    #              ignore_list_path=ignore_list_path)
    # ===========================================GLTF Model=======================================
    tag = "newpede"
    destination_folder = 'C:\\research\\gitplay\\MetaVQA\\metadrive\\assets\\models'  # Folder where you want to copy the files
    save_path_folder = 'C:\\research\\gitplay\\MetaVQA\\asset'
    src_parent_folder = 'C:\\research\\dataset\\download_asset\\pede'
    ignore_list_path = 'C:\\research\\dataset\\download_asset\\pede\\ignore_list.json'
    gltf_updater(destination_folder=destination_folder,save_path_folder=save_path_folder,src_parent_folder=src_parent_folder,
                 ignore_list_path=ignore_list_path)