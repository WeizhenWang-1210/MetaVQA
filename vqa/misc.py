import json, os, pickle
from PIL import Image
def try_load_observations(observations_path):
    base = os.path.dirname(observations_path)
    obseravations = json.load(open(observations_path, "r"))
    for modality in obseravations.keys():
        if modality == "lidar":
            try_read_pickle(os.path.join(base, obseravations[modality]))
        else:
            # rgb then
            for perspective in obseravations[modality]:
                Image.open(os.path.join(base, obseravations[modality][perspective]))
                print(modality, perspective)


def try_read_pickle(filepath):
    content = pickle.load(open(filepath, 'rb'))
    print(content)

