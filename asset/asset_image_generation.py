import json
import os
import trimesh
from asset.read_config import configReader
from trimesh.viewer import SceneViewer
config = configReader()
path_config = config.loadPath()
# Define the paths for the JSON and GLB directories
json_folder_path = path_config["adj_parameter_folder"]
glb_folder_path = 'D:\\research\\test-backup'
output_folder_path = 'D:\\research\\dataset\\generated_image'

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Iterate over each JSON file in the folder
for json_file in os.listdir(json_folder_path):
    if json_file.endswith('.json'):
        json_path = os.path.join(json_folder_path, json_file)

        # Read the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)

        if json_file.startswith('car_'):
            # Use MODEL_PATH and remove 'test/' prefix
            # model_path = data['MODEL_PATH']
            # glb_filename = model_path.replace('test/', '')
            continue
        else:
            # Regular file handling
            glb_filename = data['filename']
            general_type = data['general']['general_type']
            detail_type = data['general']['detail_type']
        if detail_type != "Wheelchair" and detail_type != "Scooter":
            continue
        # Construct the path for the GLB file
        glb_path = os.path.join(glb_folder_path, glb_filename)

        # Load the GLB file
        loaded = trimesh.load(glb_path)

        # Check if the loaded object is a Scene or a single Mesh
        if isinstance(loaded, trimesh.Scene):
            scene = loaded
        else:
            # Create a Scene if a single Mesh is loaded
            scene = trimesh.Scene(loaded)

        # Take a screenshot
        png = scene.save_image(resolution=[1920, 1080], visible=True)

        # Construct the output filename
        output_filename = f"{general_type}-{detail_type}-{glb_filename}.jpg"
        output_path = os.path.join(output_folder_path, output_filename)

        # Save the screenshot
        with open(output_path, 'wb') as file:
            file.write(png)

print("Processing complete.")