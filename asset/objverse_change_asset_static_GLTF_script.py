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

def save_ignore_list(file_path, ignore_list):
    """Save the ignore list to a file."""
    with open(file_path, 'w') as file:
        json.dump(ignore_list, file)
if __name__ == "__main__":
    tag = "newpede"
    destination_folder = 'C:\\research\\gitplay\\MetaVQA\\metadrive\\assets\\models'  # Folder where you want to copy the files
    save_path_folder = 'C:\\research\\gitplay\\MetaVQA\\asset'
    src_parent_folder = 'C:\\research\\dataset\\download_asset\\pede'
    ignore_list_path = 'C:\\research\\dataset\\download_asset\\pede\\ignore_list.json'
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
