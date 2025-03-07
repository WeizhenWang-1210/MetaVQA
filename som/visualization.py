from typing import Iterable

import cv2
import json
import os
from PIL import Image
import numpy as np
import imageio
import glob


def highlight(img: np.array, ids: Iterable[str], colors: Iterable, mapping: dict, ) -> np.array:
    """
    Hight light imgs. If the color actually exists in the image, hightlight it into white
    """
    H, W, C = img.shape
    img = img / 255  # needed as the r,g,b values in the mapping is clipped.
    flattened = img.reshape(H * W, C)
    for id, high_light in zip(ids, colors):
        if id not in mapping.keys():
            continue
        color = mapping[id]
        masks = np.all(np.isclose(flattened, color), axis=1)  # Robust against floating-point arithmetic
        flattened[masks] = high_light
    flattened = flattened * 255  # Restore into 0-255 so that cv2.imwrite can property write the image
    flattened = flattened[:, [2, 1, 0]]  # Convert rgb back to bgr
    return flattened.reshape(H, W, C)


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
    imgs = [Image.open(img_path) for img_path in images]
    grid_array = gridify_imarrays(imgs)
    Image.fromarray(grid_array).save(output_path)


def create_video(frame_arrays, filename, fps=5):
    """
    frames in (h,w,c) numpy arrays, unint8
    output_path should be str
    create a mp4 video file implied
    """
    output_path = filename  # f'{filename}.mp4'
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frame_arrays:
        writer.append_data(frame)
    writer.close()


def gridify_imarrays(imarrays, orders=None) -> np.ndarray:
    """
    Convert 6 images into 6 3 * 2 grid.
    lf f rf          ^ +x
    lb b rb    +y <--|
    Assert images a list of numpy arrays in shape (h,w,c)
    orders should be a list of len()
    """
    imgs = imarrays
    if orders is not None:
        for old_pos, new_pos in enumerate(orders):
            imgs[new_pos] = imarrays[old_pos]
    top_row = np.concatenate(imgs[:3], axis=1)

    # Concatenate three arrays for the second row
    bottom_row = np.concatenate(imgs[3:], axis=1)

    # Concatenate the two rows to get the final grid shape
    final_grid = np.concatenate((top_row, bottom_row), axis=0)
    return final_grid


def concatenate_frame(framepath, modality="rgb"):
    perspectives = ["leftf", "front", "rightf", "leftb", "back", "rightb"]
    imlist = []
    for perspective in perspectives:
        template = f"{framepath}/{modality}_{perspective}*.png"
        print(template)
        frame_file = glob.glob(template)[0]
        imlist.append(frame_file)
    multiview_visualization(imlist, f"{framepath}/multiview_{modality}.png")


def concatenate_frames(framepaths):
    for framepath in framepaths:
        concatenate_frame(framepath)

import numpy as np

import re
def visualizae_closed_loop(trial_directory):
    def extract_numbers(filename):
        #print(filename)
        pattern = r"(.*?)_(\d.*).jpg$"
        index = re.findall(pattern, filename)[-1][-1]
        return int(index)

    obs = glob.glob(os.path.join(trial_directory, "obs*.jpg"))
    fronts = glob.glob(os.path.join(trial_directory, "front*.jpg"))
    obs_ordered = sorted(obs, key=extract_numbers)
    fronts_ordered = sorted(fronts, key=extract_numbers)
    create_video(
        frame_arrays=[np.asarray(Image.open(ob)) for ob in obs_ordered],
        filename=os.path.join(trial_directory, "obs.mp4"),
        fps=20)
    create_video(
        frame_arrays=[np.asarray(Image.open(ob)) for ob in fronts_ordered],
        filename=os.path.join(trial_directory, "front.mp4"),
        fps=20)



def visualize_session(root_dir):
    def extract_numbers(filename):
        # print(os.path.basename(filename))
        basename = os.path.basename(filename)
        basename = basename.split(".")[0]
        basename = basename.split("_")
        x, y = int(basename[-2]), int(basename[-1])
        # identifier = filename.split("/")[-2]
        # x, y = identifier.split('_')
        return x, y  # (int(x), int(y))

    contents = os.listdir(root_dir)
    for content in contents:
        if os.path.isdir(os.path.join(root_dir, content)):
            # print(content)
            perspectives = ["front"]#["leftf", "front", "rightf", "leftb", "back", "rightb"]
            imarrays = {
                perspective: [] for perspective in perspectives
            }
            for perspective in perspectives:
                path_template = f"{root_dir}/{content}/**/rgb_{perspective}*.png"
                frame_files = sorted(glob.glob(path_template, recursive=True), key=extract_numbers)
                # print(frame_files)
                imarrays[perspective] = [np.asarray(Image.open(frame_file)) for frame_file in frame_files]
            num_frames = len(imarrays[perspectives[0]])
            # print(num_frames)
            concatenated_arrays = []
            for frame in range(num_frames):
                arrays = []
                for perspective in perspectives:
                    arrays.append(imarrays[perspective][frame])
                #concatenated_arrays.append(gridify_imarrays(arrays))
                concatenated_arrays.append(arrays)
            create_video(concatenated_arrays, f"{root_dir}/{content}/episode_rgb.mp4")

            for perspective in perspectives:
                path_template = f"{root_dir}/{content}/**/real_{perspective}*.png"
                frame_files = sorted(glob.glob(path_template, recursive=True), key=extract_numbers)
                # print(frame_files)
                imarrays[perspective] = [np.asarray(Image.open(frame_file)) for frame_file in frame_files]
            num_frames = len(imarrays[perspectives[0]])
            # print(num_frames)
            concatenated_arrays = []
            for frame in range(num_frames):
                arrays = []
                for perspective in perspectives:
                    arrays.append(imarrays[perspective][frame])
                #concatenated_arrays.append(gridify_imarrays(arrays))
                concatenated_arrays.append(arrays)
            create_video(concatenated_arrays, f"{root_dir}/{content}/episode_real.mp4", fps=2)

            for perspective in perspectives:
                path_template = f"{root_dir}/{content}/**/mask_{perspective}*.png"
                frame_files = sorted(glob.glob(path_template, recursive=True), key=extract_numbers)
                # print(frame_files)
                imarrays[perspective] = [np.asarray(Image.open(frame_file)) for frame_file in frame_files]
            num_frames = len(imarrays[perspectives[0]])
            # print(num_frames)
            concatenated_arrays = []
            for frame in range(num_frames):
                arrays = []
                for perspective in perspectives:
                    arrays.append(imarrays[perspective][frame])
                #concatenated_arrays.append(gridify_imarrays(arrays))
                concatenated_arrays.append(arrays)
            create_video(concatenated_arrays, f"{root_dir}/{content}/episode_mask.mp4")
            
            
            
            for perspective in perspectives:
                path_template = f"{root_dir}/{content}/**/depth_{perspective}*.png"
                frame_files = sorted(glob.glob(path_template, recursive=True), key=extract_numbers)
                # print(frame_files)
                imarrays[perspective] = [np.asarray(Image.open(frame_file)) for frame_file in frame_files]
            num_frames = len(imarrays[perspectives[0]])
            # print(num_frames)
            concatenated_arrays = []
            for frame in range(num_frames):
                arrays = []
                for perspective in perspectives:
                    arrays.append(imarrays[perspective][frame])
                #concatenated_arrays.append(gridify_imarrays(arrays))
                concatenated_arrays.append(arrays)
            create_video(concatenated_arrays, f"{root_dir}/{content}/episode_depth.mp4")
            
            for perspective in perspectives:
                path_template = f"{root_dir}/{content}/**/semantic_{perspective}*.png"
                frame_files = sorted(glob.glob(path_template, recursive=True), key=extract_numbers)
                # print(frame_files)
                imarrays[perspective] = [np.asarray(Image.open(frame_file)) for frame_file in frame_files]
            num_frames = len(imarrays[perspectives[0]])
            # print(num_frames)
            concatenated_arrays = []
            for frame in range(num_frames):
                arrays = []
                for perspective in perspectives:
                    arrays.append(imarrays[perspective][frame])
                #concatenated_arrays.append(gridify_imarrays(arrays))
                concatenated_arrays.append(arrays)
            create_video(concatenated_arrays, f"{root_dir}/{content}/episode_semantic.mp4")
            

            top_down_template = f"{root_dir}/{content}/**/top_down*.png"
            frame_files = sorted(glob.glob(top_down_template, recursive=True), key=extract_numbers)
            imarrays = [np.asarray(Image.open(frame_file)) for frame_file in frame_files]
            create_video(imarrays, f"{root_dir}/{content}/episode_top_down.mp4")

def visualize_frames(root_dir):
    perspectives = ["leftf", "front", "rightf", "leftb", "back", "rightb"]
    for content in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, content)):
            # this is an episode folder
            joined_path = os.path.join(root_dir, content)
            frame_folders = os.listdir(joined_path)
            frame_folders = [stuff for stuff in frame_folders if os.path.isdir(os.path.join(joined_path, stuff))]
            for frame_folder in frame_folders:
                print("Working on frame {}".format(frame_folder))
                frame_path = os.path.join(joined_path, frame_folder)
                img_paths = [os.path.join(frame_path, f"rgb_{perspective}_{frame_folder}.png") for perspective in
                             perspectives]
                multiview_rendering_path = os.path.join(frame_path, "multiview_rendering.png")
                multiview_real_path = os.path.join(frame_path, "multiview_real.png")
                multiview_visualization(img_paths, multiview_rendering_path)
                path_template = os.path.join(frame_path, "real*.png")
                have_real = len(glob.glob(path_template)) > 0
                if have_real:
                    real_paths = [os.path.join(frame_path, f"real_{perspective}_{frame_folder}.png") for perspective in
                                  perspectives]
                    multiview_visualization(real_paths, multiview_real_path)


from collections import defaultdict
def demo(directory):
    def extract_numbers(filename):
        # print(os.path.basename(filename))
        basename = os.path.basename(filename)
        basename = basename.split(".")[0]
        basename = basename.split("_")
        x, y = int(basename[-2]), int(basename[-1])
        # identifier = filename.split("/")[-2]
        # x, y = identifier.split('_')
        return x, y  # (int(x), int(y))

    perspectives = ["leftf", "front", "rightf", "leftb", "back", "rightb"]
    imgs = {perspective:[] for perspective in perspectives}
    for perspective in perspectives:
        perspective_folder = os.path.join(directory, perspective)
        img_paths = sorted(glob.glob(os.path.join(perspective_folder, "*.png")), key=extract_numbers)
        imgs[perspective] = img_paths
    img_ordered_by_frames = {i:[] for i in range(20)}
    for i in img_ordered_by_frames.keys():
        for perspective in perspectives:
            img_ordered_by_frames[i].append(imgs[perspective][i])
        multiview_visualization(img_ordered_by_frames[i], os.path.join(directory, f"multiview_{i}.png"))

if __name__ == "__main__":
    trials = glob.glob(os.path.join("/home/weizhen/closed_loops/internvl2_finetuned_som/*"))
    trials = [f for f in trials if not f.endswith(".json")]
    for trial in trials:
        visualizae_closed_loop(trial_directory=trial)