import yaml
import os
from pathlib import Path
from typing import Dict
class configReader:
    def __init__(self, config_path = "config.yaml"):
        with open("config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
    def loadSubPath(self, parent_folder: str, child_folder_dict: Dict):
        unified_parent_folder = Path(parent_folder)
        result_folder_dict = {}
        for key, path in child_folder_dict.items():
            result_folder_dict[key] = os.path.join(unified_parent_folder, Path(path))
        return result_folder_dict

    def loadPath(self):
        result_folder_dict = {}
        path_dict = self.config['path']
        for key, path in path_dict.items():
            if key == "subfolders":
                concat_path_dict = self.loadSubPath(parent_folder = path_dict["parentfolder"],
                                                 child_folder_dict = path
                                                 )
                result_folder_dict.update(concat_path_dict)
            else:
                result_folder_dict[key] = path
        return result_folder_dict
    def loadTag(self):
        return self.config["tag"]