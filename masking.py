import json
import os.path

import numpy as np
from typing import List
from PIL import Image
import cv2


def contrastive_color(image, center):
    center = center[1], center[0]
    surrounding_size = 10  # Number of pixels around the center to sample
    height, width, _ = image.shape

    # Get the surrounding pixels' coordinates (clamp to image boundaries)
    y, x = center
    y_min = max(0, y - surrounding_size)
    y_max = min(height, y + surrounding_size + 1)
    x_min = max(0, x - surrounding_size)
    x_max = min(width, x + surrounding_size + 1)
    # Sample the surrounding pixels' colors
    surrounding_pixels = image[y_min:y_max, x_min:x_max]
    # Calculate the mean color of the surrounding pixels
    average_color = surrounding_pixels.mean(axis=(0, 1))
    # Determine a contrastive color (simple inversion approach)
    contrastive_color = 255 - average_color  # Invert the color for contrast
    # Convert to integer tuple for OpenCV color usage
    contrastive_color = tuple([int(c) for c in list(contrastive_color)])
    return contrastive_color


def contrastive_background(rgb):
    """
    Borrowed from Microsoft's implementation of Set of Marks
    :param rgb:
    :return: (0,0,0) or (255,255,255)
    """
    R, G, B = rgb
    # Calculate the Y value
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    # If Y value is greater than 128, it's closer to white so return black. Otherwise, return white.
    return (0, 0, 0) if Y > 128 else (255, 255, 255)


def find_center(bitmask):
    """
    :param bitmask: 2D numpy array containing values of 0 and 1
    :return: the center pixel according to l2 distance transform.
    """
    # Assume `bitmask` is a binary mask with values 0 and 1
    # Convert the bitmask to a format suitable for distance transform (0 and 255)
    bitmask_255 = (bitmask * 255).astype(np.uint8)
    bitmask = bitmask.astype(np.uint8)
    # Set the top and bottom rows to zero
    bitmask_255[:4, :] = 0
    bitmask_255[-4:] = 0
    # Set the left and right columns to zero
    bitmask_255[:, :4] = 0
    bitmask_255[:, -4:] = 0

    dist_transform = cv2.distanceTransform(bitmask_255, cv2.DIST_L2, 5)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
    contours, _ = cv2.findContours(bitmask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if np.sum(bitmask) < 2000:
        # The masked area is too small, need to move the text outside.
        for cnt in contours:
            # Create a "hollowed rectangle" shape around the bounding box.
            x, y, w, h = cv2.boundingRect(cnt)
            image = np.zeros((bitmask_255.shape[0], bitmask_255.shape[1], 3))
            cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 255, 255), thickness=-1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)
            bitmask_free = np.all(image == [255, 255, 255], axis=-1).astype(np.uint8)
            #cv2.imwrite("regions.png", bitmask_free * 255)
            # Find available space in this "hollowed rectangle"
            dist_transform = cv2.distanceTransform(bitmask_free, cv2.DIST_L2, 5)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
    # The location of the maximum value is the center of the region
    center = max_loc
    return center, contours


def put_text(image, text, center, color=(255, 255, 255)):
    """
    Put <text> in <image> centered at <center> in <color>
    :param image: (H, W, C) numpy array
    :param text: str
    :param center: tuple(int, int)
    :param color: tuple(int, int, int)
    :return: None. Modify <image> in-place.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 1
    # Write text at the specified location (center)
    # Get the text size to draw the background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Calculate the top-left corner of the rectangle based on the center
    x, y = center
    top_left = (x + text_width, y - text_height)
    # Set the background color (e.g., white) and text color (e.g., black)
    background_color = contrastive_background(color)  # White
    text_color = color  # Black
    # Draw the filled rectangle (background) around the text
    cv2.rectangle(image, center, top_left, background_color, thickness=-1)
    cv2.putText(image, text, center, font, font_scale, text_color, thickness, cv2.LINE_AA)


def put_rectangle(image, text, center, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 1
    # Write text at the specified location (center)
    # Get the text size to draw the background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Calculate the top-left corner of the rectangle based on the center
    x, y = center
    top_left = (x + text_width, y - text_height)
    # Set the background color (e.g., white) and text color (e.g., black)
    background_color = color
    # Draw the filled rectangle (background) around the text
    cv2.rectangle(image, center, top_left, background_color, thickness=-1)
    return image


def find_areas(img: np.array, colors: List, mode="RGB"):
    """
    Find the areas occupied by each color in <colors> in <img>. Default color indexing is "RGB"
    :param img: (H, W, C) numpy array
    :param colors: List of colors converted to 0-255 scale
    :param mode: if in anything other than "RGB", will swap channels here
    :return: areas(int) in list and corresponding boolean bitmasks(np.array) for each color.
    """

    flattened = img.reshape(-1, img.shape[2])
    unique_colors, counts = np.unique(flattened, axis=0, return_counts=True)
    unique_colors, counts = unique_colors.tolist(), counts.tolist()
    if mode == "BGR":
        unique_colors = [(r, g, b) for (b, g, r) in unique_colors]
    unique_colors = [(round(r, 5), round(g, 5), round(b, 5)) for r, g, b in unique_colors]
    color_mapping = {
        color: count for color, count in zip(unique_colors, counts)
    }
    results = []
    masks = []
    for color in colors:
        if color in color_mapping.keys():
            results.append(color_mapping[color])
            bitmask = np.zeros((img.shape[0], img.shape[1]))
            mask = np.all(img[..., ::-1] == color, axis=-1)
            bitmask = np.logical_or(bitmask, mask)
            masks.append(bitmask)
        else:
            results.append(0)
    return results, masks


import glob


def id2label(episode_path: str, perspective: str = "front"):
    """
    Find all front-visible objects in the episode and provide them with unique labels(within the scope of this episode).
    :param perspective:
    :param episode_path: str. Path to an episode
    :return: None. Will write to the episode folder an "id2label.json" file.
    """
    scene_graph_template = os.path.join(episode_path, "**/world**.json")
    scene_graphs = glob.glob(scene_graph_template)
    result = {}
    currentlabel = 0
    for scene_graph in scene_graphs:
        world = json.load(open(scene_graph, "r"))
        for obj in world["objects"]:
            if perspective in obj["observing_camera"]:
                if obj["id"] in result.keys():
                    continue
                result[obj["id"]] = currentlabel
                currentlabel += 1
    json.dump(result, open(os.path.join(episode_path, "id2label_{}.json").format(perspective), "w"))


def labelframe(frame_path: str, perspective: str = "front", save_path: str = None, save_label: bool = False):
    """
    :param frame_path:
    :param perspective: choose from "front"|"leftf"|"leftb"|"rightf"|"rightb"|"back"
    :param save_path:
    :return:
    """

    episode_path = os.path.dirname(frame_path)
    identifier = os.path.basename(frame_path)
    world = os.path.join(frame_path, "world_{}.json".format(identifier))
    id2c = os.path.join(frame_path, "id2c_{}.json".format(identifier))
    instance_seg = os.path.join(frame_path, "mask_{}_{}.png".format(perspective, identifier))
    base_img = os.path.join(frame_path, "rgb_{}_{}.png".format(perspective, identifier))
    world = json.load(open(world))
    id2c = json.load(open(id2c))
    id2l = json.load(open(os.path.join(episode_path, "id2label_{}.json".format(perspective))))
    print(id2l.keys())
    mask_img = cv2.imread(instance_seg)
    mask = np.array(mask_img)
    base_img = np.array(cv2.imread(base_img))
    canvas = np.zeros_like(base_img)
    query_ids = []
    for obj in world["objects"]:
        if perspective in obj["observing_camera"]:
            query_ids.append(obj["id"])
    print(query_ids)
    colors = [id2c[query_id] for query_id in query_ids]
    colors = [(round(color[0] * 255, 0), round(color[1] * 255, 0), round(color[2] * 255, 0)) for color in colors]
    areas, binary_masks = find_areas(mask, colors, "BGR")
    tuples = [(query_id, color, area, binary_mask)
              for query_id, color, area, binary_mask
              in zip(query_ids, colors, areas, binary_masks)]
    area_ascending = sorted(tuples, key=lambda x: x[2])
    center_list, contour_list = [], []

    text_boxes = np.zeros_like(base_img)

    for i in range(len(area_ascending)):
        query_id, color, area, binary_mask = area_ascending[i]
        legal_mask = binary_mask
        for j in range(i):
            legal_mask = np.logical_and(legal_mask, ~area_ascending[j][3])
            occupied_text = np.all(text_boxes == [255, 255, 255], axis=-1)
            legal_mask = np.logical_and(legal_mask, ~occupied_text)
        colored_mask = base_img.copy()
        colored_mask[legal_mask == 1] = color
        alpha = 0.0  # Transparency factor
        base_img = cv2.addWeighted(base_img, 1 - alpha, colored_mask, alpha, 0)
        center, contours = find_center(legal_mask)
        center_list.append(center)
        contour_list.append(contours)
        text_boxes = put_rectangle(text_boxes, str(id2l[query_id]), center)
        if save_label:
            put_text(canvas, str(id2l[area_ascending[i][0]]), center, color=area_ascending[i][1])

    for i in range(len(area_ascending)):
        query_id, color, area, binary_mask = area_ascending[i]
        cv2.drawContours(base_img, contour_list[i], -1, color, 2)

    for i in range(len(area_ascending)):
        query_id, color, area, binary_mask = area_ascending[i]
        put_text(base_img, str(id2l[query_id]), center_list[i], color=color)

    cv2.imwrite(os.path.join(frame_path, "textboxes_{}_{}.png".format(perspective, identifier)), text_boxes)

    if save_path is not None:
        cv2.imwrite(save_path, base_img)
    else:
        cv2.imwrite(os.path.join(frame_path, "labeled_{}_{}.png".format(perspective, identifier)), base_img)
        if save_label:
            cv2.imwrite(os.path.join(frame_path, "label_{}_{}.png".format(perspective, identifier)), canvas)


def label_transfer(nusc_img_path, label_img_path):
    from PIL import Image
    src = Image.open(nusc_img_path)
    label = Image.open(label_img_path)
    label = label.resize((1600, 900))

    mask = Image.new('L', label.size, 0)  # Create a new grayscale image for the mask
    for x in range(label.width):
        for y in range(label.height):
            r, g, b = label.getpixel((x, y))
            if (r, g, b) != (0, 0, 0):  # Check if the pixel is not black
                mask.putpixel((x, y), 255)  # Set mask pixel to white (fully opaque)

    src.paste(label, mask=mask)
    src.save("test_transfer.png")


if __name__ == "__main__":
    frame_path = "/bigdata/weizhen/metavqa_iclr/scenarios/test_wide/scene-0061_91_100/0_91"
    id2label(os.path.dirname(frame_path), "front")
    labelframe(frame_path, "front", save_label=True)
    #label_transfer("/bigdata/weizhen/metavqa_iclr/scenarios/nuscenes/scene-0509_76_125/400_96/real_front_400_96.png",
    #              "/bigdata/weizhen/metavqa_iclr/scenarios/nuscenes/scene-0509_76_125/400_96/label_front_400_96.png")
