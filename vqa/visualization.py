from vqa.utils import highlight
import cv2
import json
import os
import numpy as np
def generate_highlighted(path_to_mask, path_to_mapping, folder, ids, colors, prefix = "highlighted"):
    """
    Take in an instance segmentation masks to recolor pixels that belong to 
    objects with ids into the provided colors
    """
    try:
        img = cv2.imread(path_to_mask)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert img from bgr to rgb. This is the order of channels in our mapping.
        with open(path_to_mapping, "r") as f:
            mapping = json.load(f)
        highlighted = highlight(img, ids, colors, mapping)
        name = "{}_{}.png".format(prefix,ids[0])
        path = os.path.join(folder, name)
        cv2.imwrite(path,highlighted)
    except Exception as e:
        raise e
    

if __name__ == "__main__":
    path_to_mask = "some_folder/10_40/mask_10_40.png"
    path_to_mapping = "some_folder/10_40/metainformation_10_40.json"
    folder = "some_folder/10_40"
    generate_highlighted(path_to_mask, path_to_mapping, folder, ["3a056b4d-bf7a-438d-a633-1d7cef82e499","fa209feb-dadf-424f-ab54-a136d6166c73"],
                         [(1,1,1),(1,1,1)])

    