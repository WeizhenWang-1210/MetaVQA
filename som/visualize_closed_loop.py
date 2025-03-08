import re
from PIL import Image, ImageDraw, ImageFont
import json
import imageio
import glob
import os
import numpy as np
def overlay(navigation, action, image):
    """
    Already in strings. image in Image
    Return PIL Image
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 40)
    text = f"Navigation: {navigation}| Action: {action}"
    # Calculate the position at the bottom center using textbbox
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    image_width, image_height = image.size
    x = (image_width - text_width) / 2
    y = image_height - text_height - 20  # 10 pixels from the bottom
    # Add text to the image
    draw.text((x, y), text, font=font, fill="White")  # Set text color as needed
    # Save or display the image
    #image.show()  # To display
    #image.save("image_with_text.jpg")
    return image

def extract_numbers(filename):
    #print(filename)
    pattern = r"(.*?)_(\d.*).jpg$"
    index = re.findall(pattern, filename)[-1][-1]
    return int(index)

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
     elif control[0] == 0.15 and control[1] == 0.3:
         return "SPEED_UP"
     elif control[0] == 0.6 and control[1] == 0.2:
         return "BIG_LEFT"
     elif control[0] == -0.6 and control[1] == 0.2:
         return "BIG_RIGHT"
     else:
         raise ValueError


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


def prepend(abs_path, prefix):
    base = os.path.basename(abs_path)
    dir = os.path.dirname(abs_path)
    return os.path.join(dir, f"{prefix}_{base}")

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
def visualize_closed_loop(folder):
    action_buffer = json.load(open(os.path.join(folder, "action_buffer.json"), "r"))
    obs = glob.glob(os.path.join(folder, "obs*.jpg"))
    fronts = glob.glob(os.path.join(folder, "front*.jpg"))
    obs_ordered = sorted(obs, key=extract_numbers)
    fronts_ordered = sorted(fronts, key=extract_numbers)
    assert len(fronts_ordered) == len(fronts) == len(action_buffer)
    overlayed_fronts = []
    overlayed_obs = []
    for idx, (key, value) in enumerate(action_buffer.items()):
        navigation = value["navigation"]
        action = control2string(value["action"])
        overlayed_front = overlay(navigation,action, Image.open(fronts_ordered[idx]))
        overlayed_ob = overlay(navigation, action, Image.open(obs_ordered[idx]))
        overlayed_front.save(prepend(fronts_ordered[idx], "demo"))
        overlayed_ob.save(prepend(obs_ordered[idx],"demo"))
        overlayed_fronts.append(overlayed_front)
        overlayed_obs.append(overlayed_ob)
    create_video(
        frame_arrays=[np.asarray(front) for front in overlayed_fronts],
        filename=os.path.join(folder, "front.mp4"),
        fps=2)
    create_video(
        frame_arrays=[np.asarray(ob) for ob in overlayed_obs],
        filename=os.path.join(folder, "obs.mp4"),
        fps=2)



if __name__ == "__main__":
    baselines = [
        "E:/closed_loops/llama_waymonusc/*/",
    ]
    for baseline in baselines:
        sessions = glob.glob(baseline)
        for session in sessions[:1]:
            visualize_closed_loop(session)
    exit()
    templates = glob.glob("E:/closed_loops_visualization/zeroshot/24/[0-9]*_[0-9]*.png")


    def e(filename):
        # print(filename)
        pattern = r"(.*?)_(\d.*).png$"
        index = re.findall(pattern, filename)[-1][-1]
        print(index)
        return int(index)
    templates = sorted(templates, key=e)

    print(templates)

    ims = [np.array(Image.open(template)) for template in templates]

    create_video(ims, "E:/closed_loops_visualization/zeroshot/24/24_demo.mp4", 5)


