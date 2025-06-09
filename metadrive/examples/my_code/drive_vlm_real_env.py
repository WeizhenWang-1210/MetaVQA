#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import os.path
import random
import time
from PIL import Image, ImageDraw, ImageFont
import glob
import numpy as np
import cv2
from som.visualize_closed_loop import create_video
import re
from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, InterventionPolicy
import json


def string2choices(control):
    if control == "TURN_LEFT":
        return "(A)"
    elif control == "TURN_RIGHT":
        return "(B)"
    elif control == "SLOW_DOWN":
        return "(C)"
    elif control == "BRAKE":
        return "(D)"
    elif control == "KEEP_STRAIGHT":
        return "(E)"
    elif control == "SPEED_UP":
        return "(F)"
    elif control == "BIG_LEFT":
        return "(G)"
    elif control == "BIG_RIGHT":
        return "(H)"
    else:
        raise ValueError


def control2string(control):
    if control[0] == 0.15 and control[1] == 0.8:
        return "TURN_LEFT"
    elif control[0] == -0.15 and control[1] == 0.8:
        return "TURN_RIGHT"
    elif control[0] == 0 and control[1] == -0.135:
        return "SLOW_DOWN"
    elif control[0] == 0 and control[1] == -0.26:
        return "BRAKE"
    elif control[0] == 0 and control[1] == 0.15:
        return "KEEP_STRAIGHT"
    elif control[0] == 0 and control[1] == 0.3:
        return "SPEED_UP"
    elif control[0] == 0.6 and control[1] == 0.2:
        return "BIG_LEFT"
    elif control[0] == -0.6 and control[1] == 0.2:
        return "BIG_RIGHT"
    else:
        raise ValueError


def extract_numbers(filename):
    # print(filename)
    pattern = r"(.*?)_(\d.*).jpg$"
    index = re.findall(pattern, filename)[-1][-1]
    return int(index)


def get_actions(action_buffer_path):
    action_buffer = json.load(open(action_buffer_path))
    duration = 5
    actions = []
    prompts = []
    navigations = []
    scene = None
    for key, value in action_buffer.items():
        actions += [value["action"] for _ in range(duration)]
        prompts += [value["prompt"] for _ in range(duration)]
        navigations += [value["navigation"] for _ in range(duration)]
        scene = value["scene"]
    return scene, actions, prompts, navigations


def fine_som_observations(action_buffer_path):
    folder = os.path.dirname(action_buffer_path)
    obs = glob.glob(os.path.join(folder, "obs*.jpg"))
    fronts = glob.glob(os.path.join(folder, "front*.jpg"))
    obs_ordered = sorted(obs, key=extract_numbers)
    fronts_ordered = sorted(fronts, key=extract_numbers)
    return obs_ordered, fronts_ordered


def get_trajectory(env):
    """
    n,2 array
    """
    scenario = env.engine.data_manager.current_scenario
    ego_id = scenario["metadata"]["sdc_id"]
    ego_track = scenario["tracks"][ego_id]
    ego_traj = ego_track["state"]["position"][..., :2]
    return ego_traj


from PIL import Image

import cv2
import numpy as np
import textwrap


def borderline(im_array):
    #im_array[:2, :, :] = (0, 0, 0)
    #im_array[:, :2, :] = (0, 0, 0)
    #im_array[-2:, 0, :] = (0, 0, 0)
    #im_array[:, -2:, :] = (0, 0, 0)
    return im_array


def render_text(text, font=cv2.FONT_HERSHEY_DUPLEX, font_scale=3, font_color=(0, 0, 0), thickness=2, line_spacing=1.5):
    """
    Render text on an image with automatic line breaks and scaling.
    """
    image = np.full((600, 2400, 3), 255, dtype=np.uint8)
    # Image dimensions
    img_height, img_width = image.shape[:2]

    # Maximum width for text in pixels
    max_text_width = img_width - 20  # Leave some margin
    # Determine initial font scale
    base_font_scale = font_scale
    base_thickness = thickness

    # Break the text into multiple lines that fit the width
    wrapped_lines = []
    for line in text.split('\n'):
        wrapped_lines.extend(textwrap.wrap(line, width=125))  # Adjust width for approximate line length

    # Determine the height of one line of text
    test_size = cv2.getTextSize("Test", font, base_font_scale, base_thickness)
    line_height = test_size[0][1] + int(line_spacing * test_size[1])  # Include spacing

    # Calculate total height of the text block
    total_text_height = len(wrapped_lines) * line_height

    # Adjust font scale to fit height if needed
    if total_text_height > img_height:
        base_font_scale *= img_height / total_text_height
        line_height = int(line_height * (img_height / total_text_height))

    # Start position for text (centered vertically)
    y_offset = max((img_height - total_text_height) // 2, line_height)

    for i, line in enumerate(wrapped_lines):
        # Position for each line (centered horizontally)
        text_size = cv2.getTextSize(line, font, base_font_scale, base_thickness)
        text_width = text_size[0][0]
        x_offset = 200  # (img_width - text_width) // 2
        y = y_offset + i * line_height

        # Render text on the image
        cv2.putText(image, line, (x_offset, y), font, base_font_scale, font_color, base_thickness)
    return image


def overlay(text, image, size=40):
    """
    Already in strings. image in Image
    Return PIL Image
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size)
    text = text
    # Calculate the position at the bottom center using textbbox
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    image_width, image_height = image.size
    x = (image_width - text_width) / 2
    y = image_height - text_height - 20  # 10 pixels from the bottom
    # Add text to the image
    draw.text((x, y), text, font=font, fill="Black")  # Set text color as needed
    return image


def warning(text, image, size=40):
    """
        Already in strings. image in Image
        Return PIL Image
        """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size)
    text = text
    # Calculate the position at the bottom center using textbbox
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    image_width, image_height = image.size
    x = (image_width - text_width) / 2
    y = 20  # 10 pixels from the bottom
    # Add text to the image
    draw.text((x, y), text, font=font, fill="Blue")  # Set text color as needed
    return image


from PIL import Image, ImageDraw, ImageFont
import textwrap


def find_navigation(prompt):
    pattern = r". Currently"
    cleaned_prompt = re.sub(pattern, ".\nCurrently", prompt)
    pattern = r"distance\("
    cleaned_prompt = re.sub(pattern, "distance (", cleaned_prompt)
    pattern = r"speed\("
    cleaned_prompt = re.sub(pattern, "speed (", cleaned_prompt)
    pattern = r"mph\). "
    cleaned_prompt = re.sub(pattern, "mph).\n", cleaned_prompt)
    return cleaned_prompt


def overlay_wrapped(image, text, font_path="arial.ttf", font_size=50, text_color=(0, 0, 0), margin=20, line_spacing=8):
    """
    Draw wrapped text on an image with PIL.ImageDraw.

    Args:
        image: PIL.Image object to draw on.
        text: The text to draw.
        font_path: Path to the font file (e.g., .ttf).
        font_size: Size of the font.
        text_color: Tuple for text color (R, G, B).
        margin: Margin for the text from the left and top edges.
        line_spacing: Space between lines in pixels.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    emphasized_font = ImageFont.truetype(font_path, 50)

    # Image dimensions
    img_width, img_height = image.size

    # Calculate max line width in pixels (adjust for margins)
    max_line_width = img_width - 2 * margin

    # Wrap the text into lines
    lines = []

    for idx, line in enumerate(text.split("\n")):
        if idx == len(text.split("\n")) - 1:
            lines.append("\n")
        lines.extend(textwrap.wrap(line, width=100))  # Adjust width based on approximate characters per line
        if idx == 0:
            lines.append("\n")

    # Draw each line
    y = margin
    for idx, line in enumerate(lines):
        # Calculate text size for each line
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]  # Width
        line_height = bbox[3] - bbox[1]  # Height

        # Draw the text
        if idx == len(lines) - 1:
            draw.text((margin, y), line, fill=text_color, font=emphasized_font, stroke_width=1)
        elif idx == 2 or idx == 3 or idx == 4:
            draw.text((margin, y), line, fill=text_color, font=font, stroke_width=1)
        else:
            draw.text((margin, y), line, fill=text_color, font=font)

        # Move to the next line
        y += line_height + line_spacing

        # Stop drawing if text exceeds image height
        if y > img_height - margin:
            break
    return image


def reduce_prompt(prompt):
    import re
    pattern = r", if chosen.*\n"
    cleaned_prompt = re.sub(pattern, " ", prompt, count=3)
    cleaned_prompt = re.sub(pattern, "\n", cleaned_prompt, count=1)
    cleaned_prompt = re.sub(pattern, " ", cleaned_prompt, count=3)
    cleaned_prompt = re.sub(pattern, "\n", cleaned_prompt)
    cleaned_prompt = "\n".join(cleaned_prompt.split("\n")[:-1])
    return cleaned_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactive_traffic", action="store_true")
    parser.add_argument("--top_down", "--topdown", action="store_true")
    parser.add_argument("--waymo", action="store_true")
    args = parser.parse_args()
    extra_args = dict(film_size=(2000, 2000)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    #print(HELP_MESSAGE)
    data_directory = "E:/Bolei/scenarios"
    num_scenarios = 120

    model = "E:/closed_loops/qwen2_waymonusc"

    template = f"{model}/*/action_buffer.json"

    save_path = "E:/closed_loops_visualization"
    modelname = os.path.basename(model)
    action_buffers = glob.glob(template)
    tmp = dict()
    ps = dict()
    navs = dict()
    ades = dict()
    obs = dict()
    fronts = dict()

    save_dict = dict()

    for action_buffer in action_buffers:
        seed = os.path.basename(os.path.dirname(action_buffer))
        vis_save_folder = os.path.join(save_path, modelname, seed)
        os.makedirs(vis_save_folder, exist_ok=True)
        file, actions, prompts, navigations = get_actions(action_buffer)
        obs_ordered, fronts_ordered = fine_som_observations(action_buffer)
        tmp[file] = actions
        ps[file] = prompts
        navs[file] = navigations
        obs[file] = obs_ordered
        fronts[file] = fronts_ordered
        save_dict[file] = vis_save_folder
    try:
        env = ScenarioEnv(
            {
                "sequential_seed": True,
                "use_render": False,  #True,
                "data_directory": data_directory,
                "num_scenarios": num_scenarios,
                "agent_policy": InterventionPolicy,
                "vehicle_config": dict(vehicle_model="static_default")
            }
        )
        for seed in range(120):
            env.reset(seed)
            collision = ""
            offroad = ""
            filename = env.engine.data_manager.current_scenario_file_name
            print(f"Working on {seed},")
            gt_trajectory = get_trajectory(env)
            if filename not in tmp.keys():
                continue
            actions = tmp[filename]
            navigations = navs[filename]
            prompts = ps[filename]
            observations = obs[filename]
            front_cams = fronts[filename]
            save_folder = save_dict[filename]
            print(f"Working on {seed}, to be saved at {save_folder}")
            continue
            count = 0
            frames = []
            #fronts = []
            top_downs = []
            captures = []
            #texts = []
            for idx, action in enumerate(actions):
                Time = f"{round(idx / 10, 1)}s"
                Inference_Time = f"{round((idx // 5) / 2, 1)}s"
                o, r, tm, tc, info = env.step(action)
                collision = "COLLISION" if len(env.agent.crashed_objects) > 0 else ""
                offroad = "OFFROAD" if info["out_of_road"] else ""

                front_cam = np.array(Image.open(front_cams[idx // 5]).resize(size=(800, 450)))
                front_cam = np.array(overlay(f"Scenario at t = {Inference_Time}", Image.fromarray(front_cam),
                                             size=45))[:, :, ::-1]

                som_obs = np.array(Image.open(observations[idx // 5]))
                som_obs = np.array(
                    overlay(
                        f"Observation at t = {Inference_Time} | Action = {control2string(action)}",
                        Image.fromarray(som_obs),
                        size=45
                    )
                )[:, :, ::-1]

                prompt = prompts[idx // 5]
                prompt = reduce_prompt(prompt)
                prompt = find_navigation(prompt)
                #print(prompt)
                #exit()
                display_text = f"Prompt at t = {Inference_Time}:\n{prompt}\nAction: {string2choices(control2string(action))} {control2string(action)}"
                text_buffer = np.full((600, 2400, 3), 255, dtype=np.uint8)
                text_box = np.array(overlay_wrapped(Image.fromarray(text_buffer), display_text))

                if filename.find("trainval") != -1:
                    o = env.render(
                        mode="top_down",
                        target_agent_heading_up=True,
                        # text=dict(Time=Time),
                        film_size=(20000, 20000),
                        screen_size=(800, 450)
                    )
                else:
                    o = env.render(
                        mode="top_down",
                        target_agent_heading_up=True,
                        screen_size=(800, 450),
                        # text=dict(Time=Time)
                    )
                top_down = o[:, :, ::-1]
                top_down = np.array(overlay(f"Top-down at t = {Time}", Image.fromarray(top_down), size=45))

                if collision == "" and offroad == "":
                    status = ""
                elif collision == "" and offroad != "":
                    status = offroad
                elif collision != "" and offroad == "":
                    status = collision
                else:
                    status = f"{collision} | {offroad}"

                top_down = np.array(warning(status, Image.fromarray(top_down), size=45))
                buffer = np.full((1500, 2400, 3), 255, dtype=np.uint8)
                buffer[:450, :800, :] = borderline(front_cam)
                buffer[450:900, :800, :] = borderline(top_down)
                buffer[:900, 800:, :] = borderline(som_obs)
                buffer[900:, :, :] = borderline(text_box)
                #cv2.imshow("frame", buffer)
                #cv2.imwrite("demo_full.png", buffer)
                #exit()
                #fronts.append(front_cam[:, :, ::-1])
                captures.append(som_obs[:, :, ::-1])
                top_downs.append(top_down[:, :, ::-1])
                #texts.append(text_box[:, :, ::-1])
                frames.append(buffer[:, :, ::-1])

                #cv2.waitKey(1)

            dir_id = os.path.basename(save_folder)
            video_path = os.path.join(save_folder, f"{dir_id}_demo.mp4")

            for idx, frame in enumerate(frames):
                im_path = os.path.join(save_folder, f"{dir_id}_{idx}.png")
                Image.fromarray(frame).save(im_path)
            for idx, frame in enumerate(top_downs):
                im_path = os.path.join(save_folder, f"top_{dir_id}_{idx}.png")
                Image.fromarray(frame).save(im_path)
            """for idx, frame in enumerate(fronts):
                im_path = os.path.join(save_folder, f"front_{dir_id}_{idx}.png")
                Image.fromarray(frame).save(im_path)"""
            for idx, frame in enumerate(captures):
                im_path = os.path.join(save_folder, f"obs_{dir_id}_{idx}.png")
                Image.fromarray(frame).save(im_path)
            """for idx, frame in enumerate(texts):
                im_path = os.path.join(save_folder, f"text_{dir_id}_{idx}.png")
                Image.fromarray(frame).save(im_path)"""
            print("save video to {}".format(video_path))
            create_video(frames, video_path, fps=5)
    finally:
        json.dump(save_dict, open("mapping.json", "w"), indent=1)
        env.close()
