from vqa.utils import highlight
import cv2
import json
import os
from PIL import Image
import numpy as np


def generate_highlighted(path_to_mask, path_to_mapping, folder, ids, colors, prefix="highlighted"):
    """
    Take in an instance segmentation masks to recolor pixels that belong to 
    objects with ids into the provided colors
    """
    try:
        img = cv2.imread(path_to_mask)
        img = cv2.cvtColor(img,
                           cv2.COLOR_BGR2RGB)  # convert img from bgr to rgb. This is the order of channels in our mapping.
        with open(path_to_mapping, "r") as f:
            mapping = json.load(f)
        highlighted = highlight(img, ids, colors, mapping)
        name = "{}_{}.png".format(prefix, ids[0])
        path = os.path.join(folder, name)
        cv2.imwrite(path, highlighted)
    except Exception as e:
        raise e


def multiview_visualization(images, output_path):
    if len(images) != 6:
        raise ValueError("Exactly 6 images are required.")

    # Open images using PIL
    imgs = [Image.open(img_path) for img_path in images]

    # Assuming all images are the same size
    img_width, img_height = imgs[0].size

    # Create a new image with appropriate size (3 columns, 2 rows)
    grid_img = Image.new('RGB', (img_width * 3, img_height * 2))

    # Place images in the new image
    for index, img in enumerate(imgs):
        # Calculate the position of this image
        x = (index % 3) * img_width
        y = (index // 3) * img_height
        grid_img.paste(img, (x, y))

    # Save or show the new image
    grid_img.save(output_path)
    # grid_img.show()


if __name__ == "__main__":
    """path_to_mask = "some_folder/10_40/mask_10_40.png"
    path_to_mapping = "some_folder/10_40/metainformation_10_40.json"
    folder = "some_folder/10_40"
    generate_highlighted(path_to_mask, path_to_mapping, folder, ["3a056b4d-bf7a-438d-a633-1d7cef82e499","fa209feb-dadf-424f-ab54-a136d6166c73"],
                         [(1,1,1),(1,1,1)])"""
    images = [
        "verification_multiview/95_210_239/95_210/rgb_leftf_95_210.png",
        "verification_multiview/95_210_239/95_210/rgb_front_95_210.png",
        "verification_multiview/95_210_239/95_210/rgb_rightf_95_210.png",
        "verification_multiview/95_210_239/95_210/rgb_leftb_95_210.png",
        "verification_multiview/95_210_239/95_210/rgb_back_95_210.png",
        "verification_multiview/95_210_239/95_210/rgb_rightb_95_210.png",
    ]
    masks = [
        "verification_multiview/95_210_239/95_210/mask_leftf_95_210.png",
        "verification_multiview/95_210_239/95_210/mask_front_95_210.png",
        "verification_multiview/95_210_239/95_210/mask_rightf_95_210.png",
        "verification_multiview/95_210_239/95_210/mask_leftb_95_210.png",
        "verification_multiview/95_210_239/95_210/mask_back_95_210.png",
        "verification_multiview/95_210_239/95_210/mask_rightb_95_210.png",
    ]

    multiview_visualization(images, "verification_multiview/95_210_239/95_210/multiview_rgb_95_210.png")
    multiview_visualization(masks, "verification_multiview/95_210_239/95_210/multiview_mask_95_210.png")


#chain of thought true false: are ther more x than y? yes becaus we have a x and b y.
#control signal/context inserted as text.