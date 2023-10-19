import yaml
import os
from pathlib import Path
from typing import Dict
class configReader:
    def __init__(self, config_path = "config.yaml"):
        self.spawnPosDict = None
        self.spawnNumDict = None
        self.reverseType = None
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
    def loadType(self):
        return self.config["type"]
    def loadTypeInfo(self):
        with open("config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        return self.config["typeinfo"]
    def loadColorList(self):
        return self.config["others"]["color"]
    def loadCarType(self):
        return self.config["type"]["vehicle"].keys()
    def getReverseType(self):
        self.reverseType = dict()
        for general_type, detail_type_dict in self.config['type'].items():
            for detail_type in detail_type_dict.keys():
                self.reverseType[detail_type] = general_type
    def getSpawnNum(self, detail_type):
        if self.reverseType is None:
            self.getReverseType()
        return self.config['type'][self.reverseType[detail_type]][detail_type]["spawnnum"]
    def getSpawnPos(self, detail_type):
        if self.reverseType is None:
            self.getReverseType()
        return self.config['type'][self.reverseType[detail_type]][detail_type]["spawnpos"]
    def updateTypeInfo(self, new_info_dict):
        for key, val in new_info_dict.items():
            self.config["typeinfo"][key] = val
            with open("config.yaml", "w") as file:
                yaml.safe_dump(self.config, file)