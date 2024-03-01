import os

def delete_files_with_prefix(directory, prefix):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")




dict(
    id = 0,

    rgb = dict(
        front = ["..."],
        left = [],
        back = [],
        right = []),
    lidar = ["path_to_lidar"],
    question = "What is the ...",
    answer = "[]..",
    answer_from= "str",
    question_type= "localization",
    pos_ego_view= "",
    types_referred= "",
    source="Waymo",
)


from dataclasses import dataclass
@dataclass
class MetaData:
    id:int
    source:str
    types_referred:list[str]
    objects_relatiec_pos:list[list]
    question_form:str
    answer_form:str

@dataclass
class RGBobservation:
    front:list
    left:list
    back:list
    right:list
    steps:int
@dataclass
class Observation:
    lidar:list
    rgb: RGBobservation
    steps:int

@dataclass
class MetaVQAData:
    obs:Observation
    question:str
    answer:str
    metadata:MetaData










if __name__ == '__main__':
    delete_files_with_prefix("./verification", "highlighted")
