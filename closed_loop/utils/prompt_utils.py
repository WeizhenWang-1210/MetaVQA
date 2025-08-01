import cv2
import numpy as np

from closed_loop.closed_loop_utils import classify_speed
from closed_loop.embodied_utils import ACTION
from closed_loop.embodied_utils import classify_distance
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from vqa.configs.namespace import MIN_OBSERVABLE_PIXEL, MAX_DETECT_DISTANCE
from vqa.configs.namespace import POSITION2CHOICE
from vqa.scenegen.utils.annotation_utils import get_visible_object_ids
from vqa.vqagen.set_of_marks import find_center, put_text, put_rectangle
from vqa.vqagen.utils.metadrive_utils import l2_distance


def observe(env):
    position = (0., -2, 1.4)
    hpr = [0, 0, 0]
    obs = env.engine.get_sensor("rgb").perceive(to_float=False, new_parent_node=env.agent.origin, position=position,
                                                hpr=hpr)
    return obs


def observe_som(env, font_scale=1, bounding_box=True, background_color=(0, 0, 0)):
    def find_areas(img, colors, mode="RGB"):
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
                mask = np.all(img == color, axis=-1)
                bitmask = np.logical_or(bitmask, mask)
                masks.append(bitmask)
            else:
                results.append(0)
        return results, masks

    position = (0., -2, 1.4)
    hpr = [0, 0, 0]
    obs = env.engine.get_sensor("rgb").perceive(to_float=True, new_parent_node=env.agent.origin, position=position,
                                                hpr=hpr)
    instance = env.engine.get_sensor("instance").perceive(to_float=True, new_parent_node=env.agent.origin,
                                                          position=position, hpr=hpr)
    instance_5 = np.round(instance, 5)
    base_img = np.copy(obs)
    color2id = env.engine.c_id
    filter = lambda r, g, b, c: not (r == 1 and g == 1 and b == 1) and not (r == 0 and g == 0 and b == 0) and (
            c > MIN_OBSERVABLE_PIXEL)
    visible_object_ids, log_mapping = get_visible_object_ids(instance, color2id, filter)
    valid_objects = env.engine.get_objects(
        lambda x: l2_distance(x, env.agent) <= MAX_DETECT_DISTANCE and x.id != env.agent.id and not isinstance(x,
                                                                                                               BaseTrafficLight))
    query_ids = [object_id for object_id in valid_objects.keys() if object_id in visible_object_ids]
    colors = [log_mapping[query_id] for query_id in query_ids]
    colors = [(round(b, 5), round(g, 5), round(r, 5)) for r, g, b in colors]
    areas, binary_masks = find_areas(instance_5, colors)
    tuples = [(query_id, color, area, binary_mask)
              for query_id, color, area, binary_mask
              in zip(query_ids, colors, areas, binary_masks)]
    area_ascending = sorted(tuples, key=lambda x: x[2])
    center_list, contour_list = [], []
    text_boxes = np.zeros_like(instance)
    id2l = {
        query_id: idx for idx, query_id in enumerate(query_ids)
    }
    for i in range(len(area_ascending)):
        query_id, color, area, binary_mask = area_ascending[i]
        legal_mask = binary_mask
        for j in range(i):
            legal_mask = np.logical_and(legal_mask, ~area_ascending[j][3])
            occupied_text = np.all(text_boxes == [1., 1., 1.], axis=-1)
            legal_mask = np.logical_and(legal_mask, ~occupied_text)
        colored_mask = base_img.copy()
        center, contours = find_center(legal_mask)
        center_list.append(center)
        contour_list.append(contours)
        text_boxes = put_rectangle(text_boxes, str(id2l[query_id]), center, [1., 1., 1.], font_scale)
    # Put boxes/shapes/contours
    for i in range(len(area_ascending)):
        query_id, color, area, binary_mask = area_ascending[i]
        if bounding_box:
            contours, _ = cv2.findContours((binary_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            tlxs = []
            tlys = []
            brxs = []
            brys = []
            for contour in contours:
                # Get the tight bounding rectangle around the contour
                x, y, w, h = cv2.boundingRect(contour)
                tlxs.append(x)
                tlys.append(y)
                brxs.append(x + w)
                brys.append(y + h)
                # Draw the bounding box on the original mask or another image
                # For example, create a copy of the mask to visualize
            cv2.rectangle(base_img, (min(tlxs), min(tlys)), (max(brxs), max(brys)), color, 2)  # Draw green bounding box
        else:
            cv2.drawContours(base_img, contour_list[i], -1, color, 2)
    # Put the text
    for i in range(len(area_ascending)):
        query_id, color, area, binary_mask = area_ascending[i]
        put_text(base_img, str(id2l[query_id]), center_list[i], color=color, font_scale=font_scale,
                 bg_color=background_color)
    return (base_img * 255).astype(np.uint8), id2l


def prepare_prompt(speed, navigation):
    speed_class = classify_speed(speed)
    criteria = {
        "slow": "(0-10 mph)",
        "moderate": "(10-30 mph)",
        "fast": "(30-50 mph)",
        "very fast": "(50+ mph)",
    }
    desc = criteria[speed_class]
    explanation = dict(
        TURN_LEFT="if chosen, your steering wheel will be turned slightly left while your speed will remain relatively constant",
        TURN_RIGHT="if chosen, your steering wheel will be turned slightly right while your speed will remain relatively constant",
        SLOW_DOWN="if chosen, your brake will be gently pressed to reduce speed while your steering wheel will be unturned",
        BRAKE="if chosen, your brake will be pressed hard in order to fully stop while your steering wheel will be unturned",
        KEEP_STRAIGHT="if chosen, your steering wheel will be unturned while your speed will remain relatively constant",
        SPEED_UP="if chosen, your steering wheel will be unturned while your speed will steadily increase",
        BIG_LEFT="if chosen, your steering wheel will be turned significantly left while your speed will remain relatively constant",
        BIG_RIGHT="if chosen, your steering wheel will be turned significantly right while your speed will remain relatively constant"
    )
    question = (
        f"You are the driver of a vehicle on the road, following given navigation commands. All possible navigations are as the follwing:\n"
        f"(1) \"forward\", your immediate destination is in front of you.\n"
        f"(2) \"go left\", your immediate destination is to your left-front.\n"
        f"(3) \"go right\", your immediate destination is to your right-front.\n"
        f"Currently, you are driving with {speed_class} speed{desc}, and your navigation command is \"{navigation}\". The image is observed in front of you. "
        f"Carefully examine the image, and choose the safest action to execute for the next 0.5 seconds from the following options:\n"
        f"(A) TURN_LEFT, {explanation[ACTION.get_action(ACTION.TURN_LEFT)]}.\n"
        f"(B) TURN_RIGHT, {explanation[ACTION.get_action(ACTION.TURN_RIGHT)]}.\n"
        f"(C) SLOW_DOWN, {explanation[ACTION.get_action(ACTION.SLOW_DOWN)]}.\n"
        f"(D) BRAKE, {explanation[ACTION.get_action(ACTION.BRAKE)]}.\n"
        f"(E) KEEP_STRAIGHT, {explanation[ACTION.get_action(ACTION.KEEP_STRAIGHT)]}.\n"
        f"(F) SPEED_UP, {explanation[ACTION.get_action(ACTION.SPEED_UP)]}.\n"
        f"(G) BIG_LEFT, {explanation[ACTION.get_action(ACTION.BIG_LEFT)]}.\n"
        f"(H) BIG_RIGHT, {explanation[ACTION.get_action(ACTION.BIG_RIGHT)]}.\n"
        f"You will attempt to follow the navigation command, but you are allowed to choose other actions to avoid potential hazards. "
        f"Answer in a single capitalized character chosen from [\"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\"].")
    return question


def prepare_prompt_dest(speed, dir, dist):
    sector_string = POSITION2CHOICE[dir]
    distance = classify_distance(dist)
    speed_class = classify_speed(speed)
    criteria = {
        "slow": "(0-10 mph)",
        "moderate": "(10-30 mph)",
        "fast": "(30-50 mph)",
        "very fast": "(50+ mph)",
    }
    dist_creteria = {
        "very close": "(0-2 m)",
        "close": "(2-10 m)",
        "medium": "(10-30 m)",
        "far": "(30+ m)"
    }
    desc = criteria[speed_class]
    explanation = dict(
        TURN_LEFT="if chosen, your steering wheel will be turned slightly left while your speed will remain relatively constant",
        TURN_RIGHT="if chosen, your steering wheel will be turned slightly right while your speed will remain relatively constant",
        SLOW_DOWN="if chosen, your brake will be gently pressed to reduce speed while your steering wheel will be unturned",
        BRAKE="if chosen, your brake will be pressed hard in order to fully stop while your steering wheel will be unturned",
        KEEP_STRAIGHT="if chosen, your steering wheel will be unturned while your speed will remain relatively constant",
        SPEED_UP="if chosen, your steering wheel will be unturned while your speed will steadily increase",
        BIG_LEFT="if chosen, your steering wheel will be turned significantly left while your speed will remain relatively constant",
        BIG_RIGHT="if chosen, your steering wheel will be turned significantly right while your speed will remain relatively constant"
    )
    question = (
        f"You are the driver of a vehicle on the road, and your final destination is at {distance} distance{dist_creteria[distance]} to your {sector_string} at this moment. "
        f"Currently, you are driving with {speed_class} speed{desc}. The image is observed in front of you. "
        f"In order to reach your final destination, carefully examine the image. Choose the best action to execute for the next 0.5 seconds from the following options:\n"
        f"(A) TURN_LEFT, {explanation[ACTION.get_action(ACTION.TURN_LEFT)]}.\n"
        f"(B) TURN_RIGHT, {explanation[ACTION.get_action(ACTION.TURN_RIGHT)]}.\n"
        f"(C) SLOW_DOWN, {explanation[ACTION.get_action(ACTION.SLOW_DOWN)]}.\n"
        f"(D) BRAKE, {explanation[ACTION.get_action(ACTION.BRAKE)]}.\n"
        f"(E) KEEP_STRAIGHT, {explanation[ACTION.get_action(ACTION.KEEP_STRAIGHT)]}.\n"
        f"(F) SPEED_UP, {explanation[ACTION.get_action(ACTION.SPEED_UP)]}.\n"
        f"(G) BIG_LEFT, {explanation[ACTION.get_action(ACTION.BIG_LEFT)]}.\n"
        f"(H) BIG_RIGHT, {explanation[ACTION.get_action(ACTION.BIG_RIGHT)]}.\n"
        f"Safety and swiftness to the final destination are both important factors to consider."
        f"Answer in a single capitalized character chosen from [\"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\"].")
    return question
