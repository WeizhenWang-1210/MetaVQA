import yaml
import os
from pathlib import Path
from typing import Dict
class configReader:
    def __init__(self, config_path = "path_config.yaml"):
        self.spawnPosDict = None
        self.spawnNumDict = None
        self.reverseType = None
        self.path_config_path = "./path_config.yaml"
        self.asset_config_path = "./asset_config.yaml"
        with open(self.path_config_path, "r") as file:
            self.path_config = yaml.safe_load(file)
        with open(self.asset_config_path, "r") as file2:
            self.asset_config = yaml.safe_load(file2)
    def loadSubPath(self, parent_folder: str, child_folder_dict: Dict):
        unified_parent_folder = Path(parent_folder)
        result_folder_dict = {}
        for key, path in child_folder_dict.items():
            result_folder_dict[key] = os.path.join(unified_parent_folder, Path(path))
        return result_folder_dict

    def loadPath(self):
        result_folder_dict = {}
        path_dict = self.path_config['path']
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
        return self.asset_config["tag"]
    def loadType(self):
        return self.asset_config["type"]
    def loadTypeInfo(self):
        with open(self.asset_config_path, "r") as file:
            self.asset_config = yaml.safe_load(file)
        return self.asset_config["typeinfo"]
    def loadColorList(self):
        return self.asset_config["others"]["color"]
    def loadCarType(self):
        return self.asset_config["type"]["vehicle"].keys()
    def getReverseType(self):
        self.reverseType = dict()
        for general_type, detail_type_dict in self.asset_config['type'].items():
            for detail_type in detail_type_dict.keys():
                self.reverseType[detail_type] = general_type
    def getSpawnNum(self, detail_type):
        if self.reverseType is None:
            self.getReverseType()
        return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["spawnnum"]
    def getSpawnPos(self, detail_type):
        if self.reverseType is None:
            self.getReverseType()
        return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["spawnpos"]
    def getSpawnHeading(self, detail_type):
        if self.reverseType is None:
            self.getReverseType()
        if "spawnheading" in self.asset_config['type'][self.reverseType[detail_type]][detail_type].keys():
            return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["spawnheading"]
        return False
    def updateTypeInfo(self, new_info_dict):
        for key, val in new_info_dict.items():
            self.asset_config["typeinfo"][key] = val
            with open(self.asset_config_path, "w") as file:
                yaml.safe_dump(self.asset_config, file)