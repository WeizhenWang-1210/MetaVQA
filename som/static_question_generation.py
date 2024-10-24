from som.utils import enumerate_frame_labels, get, fill_in_label
import os
import json
import random
from vqa.object_node import nodify, extrapolate_bounding_boxes, box_trajectories_overlap, box_trajectories_intersect
from vqa.scene_graph import SceneGraph
from vqa.dataset_utils import get_distance
from vqa.functionals import identify_heading
from vqa.dataset_utils import transform_heading
from vqa.configs.NAMESPACE import NAMESPACE, POSITION2CHOICE
import numpy as np
from vqa.static_question_generator import NAMED_MAPPING
from masking import labelframe, id2label, static_id2label
from vqa.nusc_devkit_annotation import ALL_TYPE
from vqa.static_question_generation import generate_all_frame
import itertools
from copy import deepcopy
import traceback

current_directory = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = json.load(open(os.path.join(current_directory, "questions_templates.json"), "r"))

TYPES_WITHOUT_HEADINGS=["Cone", "Barrier", "Warning", "TrafficLight", "Trailer"]

DIRECTION_MAPPING = {
    "l": "to the left of",
    "r": "to the right of",
    "f": "directly in front of",
    "b": "directly behind",
    "lf": "to the left and in front of",
    "rf": "to the right and in front of",
    "lb": "to the left and behind",
    "rb": "to the right and behind",
    "m": "in close proximity to"
}


#TODO refactor this ugly code.
def create_options(present_values, num_options, answer, namespace, transform=None):
    if len(present_values) < num_options:
        space = set(namespace)
        result = set(present_values)
        diff = space.difference(result)
        choice = np.random.choice(np.array(list(diff)), size=num_options - len(present_values), replace=False)
        result = list(result) + list(choice)
    elif len(present_values) == num_options:
        result = present_values
    else:
        answer = {answer}
        space = set(present_values)
        diff = space.difference(answer)
        choice = np.random.choice(np.array(list(diff)), size=num_options - 1, replace=False)
        result = list(answer) + list(choice)
    if transform:
        if callable(transform):
            result = [transform(o) for o in result]
        elif isinstance(transform, dict):
            result = [transform[o] for o in result]
    #paired_list = list(enumerate(result))
    np.random.shuffle(result)
    #shuffled_list = [element for _, element in paired_list]
    return result  #, {answer: index for index, answer in paired_list}


def find_label(obj_id, label2id):
    for label, id in label2id.items():
        if id == obj_id:
            return label
    return None


def create_multiple_choice(options):
    assert len(options) < 26, "no enough alphabetic character"
    result = []
    answer_to_choice = {}
    for idx, option in enumerate(options):
        label = chr(idx + 64 + 1)
        result.append(
            "({}) {}".format(label, option)
        )
        answer_to_choice[option] = label
    return "; ".join(result) + ".", answer_to_choice


def generate(frame_path: str, question_type: str, perspective: str = "front", verbose: bool = False,
             id2label_path: str = None):
    identifier = os.path.basename(frame_path)
    world_path = os.path.join(frame_path, "world_{}.json".format(identifier))
    world = json.load(open(world_path, "r"))
    label2id, invalid_ids = enumerate_frame_labels(frame_path, perspective, id2label_path)
    #print(label2id)
    #exit()
    labels = list(label2id.keys())
    #labels = [label for label in label2id.keys() if get(world, label2id[label]) is not None]
    question = None
    answer = None
    explanation = None
    ego_id, nodelist = nodify(world, multiview=False)
    graph = SceneGraph(ego_id, nodelist, frame_path)
    ego_node = graph.get_ego_node()
    ids_of_interest = []
    option2answer = {}
    if question_type == "identify_color":
        # randomly choose a non-ego labelled object
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 1:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:

            selected_label = random.choice(non_ego_labels)
            # Fill the question template's <id...> with the chosen label.
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
            # Getting the answer from SceneGraph. In addition, grab extra information from the scene graph for explanation string.
            object = get(world, label2id[selected_label])
            color, type = object["color"], object["type"]
            assert len(color)>0, "Empty string for color"
            # First, get a pool of sensible answers for the multiple-choice setup from the scenario. If no sufficient number
            # of options present, we grab from the answer space
            multiple_choice_options = create_options(graph.statistics["<p>"], 4, color, NAMESPACE["color"])
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            answer = answer2label[color]
            # Postpend the multiple-choice string.
            question = " ".join([question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            pos = ego_node.compute_relation_string(node=graph.get_node(object["id"]), ref_heading=ego_node.heading)
            type_str = NAMED_MAPPING[type]["singular"]
            explanation = f"The color of this {type_str} (<{selected_label}>) {DIRECTION_MAPPING[pos]} us is {color.lower()}."  #.format(, , , )
            ids_of_interest.append(label2id[selected_label])
    elif question_type == "identify_type":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 1:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            type_space = [NAMED_MAPPING[obj]["singular"] for obj in NAMED_MAPPING.keys()]
            #print(type_space)
            selected_label = random.choice(non_ego_labels)
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
            object = get(world, label2id[selected_label])
            color, type = object["color"], object["type"]
            #print(type)

            options = [NAMED_MAPPING[type]["singular"] for type in graph.statistics["<t>"]]
            #print(options)
            multiple_choice_options = create_options(options, 4, NAMED_MAPPING[type]["singular"], type_space)
            #print(multiple_choice_options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            for key, value in NAMED_MAPPING.items():
                multiple_choice_string = multiple_choice_string.replace(value["singular"], value["singular"].capitalize())
            #print(multiple_choice_string)
            #exit()
            question = " ".join([question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[NAMED_MAPPING[type]["singular"]]
            pos = ego_node.compute_relation_string(node=graph.get_node(object["id"]), ref_heading=ego_node.heading)



            color_str= f" {color.lower()} " if len(color)>0 else " "
            label_str= str(selected_label)
            pos_str = DIRECTION_MAPPING[pos]
            type_str = NAMED_MAPPING[type]["singular"]
            explanation = f"The type of this{color_str}object(<{label_str}>) {pos_str} of us is {type_str}."

            ids_of_interest.append(label2id[selected_label])
    elif question_type == "identify_distance":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 1:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_label = random.choice(non_ego_labels)
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
            object = graph.get_node(label2id[selected_label])
            distance = get_distance(object.pos, ego_node.pos)
            color, type = object.color, object.type
            if 0 < distance <= 2:
                #relative_dist = "very close"
                answer = "A"
            elif 2 < distance <= 10:
                #relative_dist = "close"
                answer = "B"
            elif 10 < distance <= 30:
                #relative_dist = "medium"
                answer = "C"
            else:
                #relative_dist = "far"
                answer = "D"
            option2answer = {
                "A": "very close", "B": "close", "C": "medium", "D": "far"
            }
            pos = ego_node.compute_relation_string(node=graph.get_node(object.id), ref_heading=ego_node.heading)

            color_str = f" {color.lower()} " if len(color) > 0 else " "
            label_str = str(selected_label)
            pos_str = DIRECTION_MAPPING[pos]
            type_str = NAMED_MAPPING[type]["singular"]
            explanation = f"The{color_str}{type_str}(<{label_str}>) is {round(distance)} meters {pos_str} us."

            #explanation = "The {} {}(<{}>) is {} meters {} us.".format(color.lower(), NAMED_MAPPING[type]["singular"],
            #                                                           selected_label, round(distance),
            #                                                           DIRECTION_MAPPING[pos])
            ids_of_interest.append(label2id[selected_label])
    elif question_type == "identify_position":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 1:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_label = random.choice(non_ego_labels)
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
            object = graph.get_node(label2id[selected_label])
            color, type = object.color, object.type
            pos = ego_node.compute_relation_string(node=graph.get_node(object.id), ref_heading=ego_node.heading)
            multiple_choice_options = create_options(list(POSITION2CHOICE.keys()), 4, pos, list(POSITION2CHOICE.keys()),
                                                     POSITION2CHOICE)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join([question,
                                 f"Choose the best answer from option (A) through (D): {multiple_choice_string}"])
            answer = answer2label[POSITION2CHOICE[pos]]
            explanation = "The {} {}(<{}>) is {} us.".format(color.lower(), NAMED_MAPPING[type]["singular"],
                                                             selected_label,
                                                             DIRECTION_MAPPING[pos])
            ids_of_interest.append(label2id[selected_label])
    elif question_type == "identify_heading":
        non_ego_labels = [label for label in labels if
                          graph.get_node(label2id[label]).type not in TYPES_WITHOUT_HEADINGS and label != -1]
        if len(non_ego_labels) < 1:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_label = random.choice(non_ego_labels)
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
            object = graph.get_node(label2id[selected_label])
            heading = identify_heading(ego_node.pos, ego_node.heading)([[object]])
            heading = heading[object.id]
            options = [((heading - 1 + i * 3) % 12) + 1 for i in range(4)]
            transform = lambda x: "{} o'clock".format(x)
            multiple_choice_options = create_options(options, 4, heading, list(range(1, 13)), transform)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            color, type = object.color, object.type
            pos = ego_node.compute_relation_string(node=graph.get_node(object.id),
                                                   ref_heading=ego_node.heading)
            answer = answer2label[transform(heading)]
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D):  {}".format(multiple_choice_string)])
            explanation = "The {} {}(<{}>) {} us is facing the {} o'clock direction.".format(color.lower(),
                                                                                             NAMED_MAPPING[type][
                                                                                                 "singular"],
                                                                                             selected_label,
                                                                                             DIRECTION_MAPPING[pos],
                                                                                             heading)
            ids_of_interest.append(label2id[selected_label])
    elif question_type == "pick_closer":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 2:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_labels = np.random.choice(np.array(non_ego_labels), size=2, replace=False)
            id1, id2 = selected_labels
            options = [f"<{id1}> and <{id2}> are about the same distance", f"<{id1}> is closer", f"<{id2}> is closer"]
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0],
                                     {"<id1>": str(id1), "<id2>": str(id2)})
            object1, object2 = graph.get_node(label2id[id1]), graph.get_node(label2id[id2])
            distance1, distance2 = get_distance(object1.pos, ego_node.pos), get_distance(object2.pos, ego_node.pos)
            color1, type1 = object1.color, object1.type
            pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id), ref_heading=ego_node.heading)
            color2, type2 = object2.color, object2.type
            pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id), ref_heading=ego_node.heading)
            if abs(distance1 - distance2) <= 2:
                index = 0
                explanation = ("Object <{}>, a {} {} located to our {}, is about the same distance from us as object <{"
                               "}>, a {} {} located to our {}.").format(
                    id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], id2, color2.lower(),
                    NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2]
                )
            else:
                if distance1 < distance2:
                    index = 1
                    explanation = ("Object <{}>, a {} {} {} us , is closer to us than object <{}>, "
                                   "a {} {} {} us.").format(
                        id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], id2,
                        color2.lower(),
                        NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2]
                    )
                else:
                    index = 2
                    explanation = ("Object <{}>, a {} {} {} us, is closer to us than object <{}>, "
                                   "a {} {} {} us.").format(
                        id2, color2.lower(), NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2], id1,
                        color1.lower(),
                        NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1]
                    )

            multiple_choice_options = create_options(options, 3, options[index], options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            answer = answer2label[options[index]]
            question = " ".join(
                [question, "Choose the best answer from option (A) through (C): {}".format(multiple_choice_string)])
            ids_of_interest += [label2id[l] for l in list(selected_labels)]
    elif question_type == "predict_crash_ego_still":
        #TODO examine the trajectories. via visualization
        non_ego_labels = [label for label in labels if
                          graph.get_node(label2id[label]).type not in TYPES_WITHOUT_HEADINGS and label != -1]
        if len(non_ego_labels) < 1:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_label = random.choice(non_ego_labels)
            object = graph.get_node(label2id[selected_label])
            init_center = object.pos
            extrapolated_centers = [list(np.array(object.heading) * i + np.array(init_center)) for i in range(50)]
            extrapolated_boxes = extrapolate_bounding_boxes(extrapolated_centers,
                                                            np.arctan2(object.heading[1], object.heading[0]),
                                                            object.bbox)
            ego_boxes = [ego_node.bbox for i in range(50)]
            crash = box_trajectories_overlap(extrapolated_boxes, ego_boxes)
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
            object = graph.get_node(label2id[selected_label])
            color, type = object.color, object.type
            heading = identify_heading(ego_node.pos, ego_node.heading)([[object]])
            heading = heading[object.id]
            pos = ego_node.compute_relation_string(node=graph.get_node(object.id),
                                                   ref_heading=ego_node.heading)

            options = ["Yes", "No"]
            multiple_choice_options = create_options(options, 2, "Yes" if crash else "No", options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join(
                [question, "Choose the best answer between option (A) and (B): {}".format(multiple_choice_string)])
            answer = answer2label["Yes" if crash else "No"]
            if crash:
                explanation = "Yes, this {} {}(<{}>) {} us and heading at {} o\' clock will run into us if it drives along its current heading.".format(
                    color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)
            else:
                explanation = "No, this {} {}(<{}>) {} us and heading at {} o\' clock will not run into us if it drives along its current heading.".format(
                    color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)
            ids_of_interest.append(label2id[selected_label])
    elif question_type == "predict_crash_ego_dynamic":
        #TODO examine the trajectories. via visualization
        non_ego_labels = [label for label in labels if
                          graph.get_node(label2id[label]).type not in TYPES_WITHOUT_HEADINGS and label != -1]
        if len(non_ego_labels) < 1:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_label = random.choice(non_ego_labels)
            object = graph.get_node(label2id[selected_label])
            init_center = object.pos
            extrapolated_centers = [list(np.array(object.heading) * i + np.array(init_center)) for i in range(50)]
            extrapolated_boxes = extrapolate_bounding_boxes(extrapolated_centers,
                                                            np.arctan2(object.heading[1], object.heading[0]),
                                                            object.bbox)
            ego_centers = [list(np.array(ego_node.heading) * i + np.array(ego_node.pos)) for i in range(50)]
            ego_boxes = extrapolate_bounding_boxes(ego_centers,
                                                   np.arctan2(ego_node.heading[1], ego_node.heading[0]), ego_node.bbox)

            crash = box_trajectories_overlap(extrapolated_boxes, ego_boxes)
            intersect = box_trajectories_intersect(extrapolated_boxes, ego_boxes)
            options = ["Yes", "No"]

            multiple_choice_options = create_options(options, 2, "Yes" if crash else "No", options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
            question = " ".join(
                [question, "Choose the best answer between option (A) and (B): {}".format(multiple_choice_string)])
            answer = answer2label["Yes" if crash else "No"]

            object = graph.get_node(label2id[selected_label])
            color, type = object.color, object.type
            heading = identify_heading(ego_node.pos, ego_node.heading)([[object]])
            heading = heading[object.id]
            pos = ego_node.compute_relation_string(node=graph.get_node(object.id),
                                                   ref_heading=ego_node.heading)
            if crash:
                explanation = "Yes, this {} {}(<{}>) {} us and heading at {} o\' clock will run into us if it proceed along its current heading.".format(
                    color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)
            else:
                if intersect:
                    explanation = "No, this {} {}(<{}>) {} us and heading at {} o\' clock will not run into us even though our paths will intersect.".format(
                        color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)
                else:
                    explanation = "No, this {} {}(<{}>) {} us and heading at {} o\' clock will not run into us.".format(
                        color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)
            ids_of_interest.append(label2id[selected_label])
    elif question_type == "relative_distance":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 2:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_labels = np.random.choice(np.array(non_ego_labels), size=2, replace=False)
            id1, id2 = selected_labels
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0],
                                     {"<id1>": str(id1), "<id2>": str(id2)})
            object1, object2 = graph.get_node(label2id[id1]), graph.get_node(label2id[id2])
            # distance1, distance2 = get_distance(object1.pos, ego_node.pos), get_distance(object2.pos, ego_node.pos)
            distance12 = get_distance(object1.pos, object2.pos)
            if 0 < distance12 <= 2:
                relative_dist = "very close"
                answer = "A"
            elif 2 < distance12 <= 10:
                relative_dist = "close"
                answer = "B"
            elif 10 < distance12 <= 30:
                relative_dist = "medium"
                answer = "C"
            else:
                relative_dist = "far"
                answer = "D"
            option2answer = {
                "A": "very close", "B": "close", "C": "medium", "D": "far"
            }
            pos2to1 = object2.compute_relation_string(node=object1, ref_heading=ego_node.heading)
            color1, type1 = object1.color, object1.type
            pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),
                                                    ref_heading=ego_node.heading)
            color2, type2 = object2.color, object2.type
            pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id),
                                                    ref_heading=ego_node.heading)
            explanation = "Object <{}>, a {} {} {} us, is {} object <{}>, a {} {} {} us, at a {} distance.".format(
                id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1],
                DIRECTION_MAPPING[pos2to1],
                id2, color2.lower(), NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2], relative_dist)

            ids_of_interest += [label2id[l] for l in list(selected_labels)]
    elif question_type == "relative_position":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 2:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_labels = np.random.choice(np.array(non_ego_labels), size=2, replace=False)
            id1, id2 = selected_labels
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0],
                                     {"<id1>": str(id1), "<id2>": str(id2)})
            object1, object2 = graph.get_node(label2id[id1]), graph.get_node(label2id[id2])
            #distance1, distance2 = get_distance(object1.pos, ego_node.pos), get_distance(object2.pos, ego_node.pos)
            pos2to1 = object2.compute_relation_string(node=object1, ref_heading=ego_node.heading)
            color1, type1 = object1.color, object1.type
            pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),
                                                    ref_heading=ego_node.heading)
            color2, type2 = object2.color, object2.type
            pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id),
                                                    ref_heading=ego_node.heading)

            multiple_choice_options = create_options(list(POSITION2CHOICE.keys()), 4, pos2to1,
                                                     list(POSITION2CHOICE.keys()),
                                                     POSITION2CHOICE)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[POSITION2CHOICE[pos2to1]]
            explanation = "Object <{}>, a {} {} {} us, is {} object <{}>, a {} {} {} us.".format(
                id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1],
                DIRECTION_MAPPING[pos2to1],
                id2, color2.lower(), NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2])
            ids_of_interest += [label2id[l] for l in list(selected_labels)]
    elif question_type == "relative_heading":
        non_ego_labels = [label for label in labels if
                          graph.get_node(label2id[label]).type not in TYPES_WITHOUT_HEADINGS and label != -1]
        if len(non_ego_labels) < 2:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_labels = np.random.choice(np.array(non_ego_labels), size=2, replace=False)
            id1, id2 = selected_labels
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0],
                                     {"<id1>": str(id1), "<id2>": str(id2)})
            object1, object2 = graph.get_node(label2id[id1]), graph.get_node(label2id[id2])
            # distance1, distance2 = get_distance(object1.pos, ego_node.pos), get_distance(object2.pos, ego_node.pos)
            angle1to2 = np.rad2deg(transform_heading(object2.heading, object1.pos, object1.heading))
            pos2to1 = object2.compute_relation_string(node=object1, ref_heading=ego_node.heading)
            color1, type1 = object1.color, object1.type
            pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),
                                                    ref_heading=ego_node.heading)
            color2, type2 = object2.color, object2.type
            pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id),
                                                    ref_heading=ego_node.heading)
            options = ["Yes", "No"]
            answer = "Yes" if abs(angle1to2) < 20 else "No"
            multiple_choice_options = create_options(options, 2, answer, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            answer = answer2label[answer]
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join(
                [question, "Choose the best answer between option (A) and (B): {}".format(multiple_choice_string)])

            ids_of_interest += [label2id[l] for l in list(selected_labels)]
            if abs(angle1to2) < 20:
                explanation = "Yes. Object <{}>, a {} {} {} us, is heading toward roughly the same direction as object <{}>, a {} {} {} us.".format(
                    id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], id2, color2.lower(),
                    NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2])
            else:
                explanation = "No. Object <{}>, a {} {} {} us, is not heading toward the same direction as object <{}>, a {} {} located {} us. In particular, object <{}>'s heading differs by {} degrees counterclockwise from that of object <{}>.".format(
                    id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], id2, color2.lower(),
                    NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2], id2, round(angle1to2), id1)
    elif question_type == "relative_predict_crash_still":
        #TODO examine the trajectories. via visualization
        non_ego_labels = [label for label in labels if
                          graph.get_node(label2id[label]).type not in TYPES_WITHOUT_HEADINGS and label != -1]
        if len(non_ego_labels) < 2:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            # TODO select cars only
            selected_labels = np.random.choice(np.array(non_ego_labels), size=2, replace=False)
            id1, id2 = selected_labels
            object1, object2 = graph.get_node(label2id[id1]), graph.get_node(label2id[id2])
            init_center = object1.pos
            extrapolated_centers = [list(np.array(object1.heading) * i + np.array(init_center)) for i in range(50)]
            extrapolated_boxes = extrapolate_bounding_boxes(extrapolated_centers,
                                                            np.arctan2(object1.heading[1], object1.heading[0]),
                                                            object1.bbox)
            object2_boxes = [object2.bbox for i in range(50)]
            crash = box_trajectories_overlap(extrapolated_boxes, object2_boxes)
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0],
                                     {"<id1>": str(id1), "<id2>": str(id2)})
            color1, type1 = object1.color, object1.type
            pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),
                                                    ref_heading=ego_node.heading)
            color2, type2 = object2.color, object2.type
            pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id),
                                                    ref_heading=ego_node.heading)

            options = ["Yes", "No"]
            multiple_choice_options = create_options(options, 2, "Yes" if crash else "No", options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join(
                [question, "Choose the best answer between option (A) and (B): {}".format(multiple_choice_string)])
            answer = answer2label["Yes" if crash else "No"]

            if crash:
                explanation = "Yes, object <{}> will run into object <{}>.".format(
                    id1, id2)
            else:
                explanation = "No, object <{}> will not run into object <{}>.".format(
                    id1, id2)
            ids_of_interest += [label2id[id1], label2id[id2]]
    elif question_type == "relative_predict_crash_dynamic":
        #TODO examine the trajectories. via visualization
        non_ego_labels = [label for label in labels if
                          graph.get_node(label2id[label]).type not in TYPES_WITHOUT_HEADINGS and label != -1]
        if len(non_ego_labels) < 2:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:

            # TODO select cars only
            selected_labels = np.random.choice(np.array(non_ego_labels), size=2, replace=False)
            id1, id2 = selected_labels

            object1, object2 = graph.get_node(label2id[id1]), graph.get_node(label2id[id2])
            init_center = object1.pos
            extrapolated_centers = [list(np.array(object1.heading) * i + np.array(init_center)) for i in range(50)]
            #print(extrapolated_centers)
            extrapolated_boxes1 = extrapolate_bounding_boxes(extrapolated_centers,
                                                             np.arctan2(object1.heading[1], object1.heading[0]),
                                                             object1.bbox)
            extrapolated_centers2 = [list(np.array(object2.heading) * i + np.array(object2.pos)) for i in range(50)]
            extrapolated_boxes2 = extrapolate_bounding_boxes(extrapolated_centers2,
                                                             np.arctan2(object2.heading[1], object2.heading[0]),
                                                             object2.bbox)
            crash = box_trajectories_overlap(extrapolated_boxes1, extrapolated_boxes2)
            intersect = box_trajectories_intersect(extrapolated_boxes1, extrapolated_boxes2)
            question = fill_in_label(TEMPLATES["static"][question_type]["text"][0],
                                     {"<id1>": str(id1), "<id2>": str(id2)})
            color1, type1 = object1.color, object1.type
            pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),
                                                    ref_heading=ego_node.heading)
            color2, type2 = object2.color, object2.type
            pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id),
                                                    ref_heading=ego_node.heading)
            options = ["Yes", "No"]
            multiple_choice_options = create_options(options, 2, "Yes" if crash else "No", options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join(
                [question, "Choose the best answer between option (A) and (B): {}".format(multiple_choice_string)])
            answer = answer2label["Yes" if crash else "No"]
            if crash:
                explanation = "Yes, object <{}> will run into object <{}>.".format(
                    id1, id2)
            else:
                explanation = "No, object <{}> will not run into object <{}>.".format(
                    id1, id2)
            ids_of_interest += [label2id[l] for l in list(selected_labels)]
    elif question_type == "order_closest":
        def dist(label):
            return get_distance(
                graph.get_node(label2id[label]).pos, ego_node.pos
            )

        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_labels = list(np.random.choice(np.array(non_ego_labels), size=4, replace=False))
            ordered_labels = sorted(selected_labels, key=dist)
            all_permutations = list(itertools.permutations(selected_labels))
            distinct_permutations = random.sample(all_permutations, 4)
            if tuple(ordered_labels) not in distinct_permutations:
                distinct_permutations[-1] = tuple(ordered_labels)
            for idx, distinct_permutation in enumerate(distinct_permutations):
                new_tuple = [
                    f"<{val}>" for val in distinct_permutation
                ]
                new_tuple = ", ".join(new_tuple)
                distinct_permutations[idx] = new_tuple
            options = distinct_permutations
            answer_ordering = ", ".join([f"<{val}>" for val in ordered_labels])
            multiple_choice_options = create_options(options, 4, answer_ordering, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = fill_in_label(
                template_str=TEMPLATES["static"][question_type]["text"][0],
                replacement={code: str(selected_labels[idx])
                             for idx, code in enumerate(TEMPLATES["static"][question_type]["params"])
                             }
            )
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[answer_ordering]
            object1, object2, object3, object4 = [graph.get_node(label2id[label]) for label in ordered_labels]
            explanation = "The {} {}(<{}>) is closest to us, and the {} {}(<{}>) is furthest away from us. The {} {}(<{}>) and the {} {}(<{}>) are in between.".format(
                object1.color.lower(), NAMED_MAPPING[object1.type]["singular"], ordered_labels[0],
                object4.color.lower(), NAMED_MAPPING[object4.type]["singular"], ordered_labels[3],
                object2.color.lower(), NAMED_MAPPING[object2.type]["singular"], ordered_labels[1],
                object3.color.lower(), NAMED_MAPPING[object3.type]["singular"], ordered_labels[2]
            )
            ids_of_interest = [label2id[label] for label in selected_labels]
    elif question_type == "order_leftmost":
        def dist(label):
            o = graph.get_node(label2id[label])
            left_vec = np.array([-ego_node.heading[1], ego_node.heading[0]])
            return (np.array(o.pos) - np.array(ego_node.pos)).dot(left_vec)

        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            selected_labels = list(np.random.choice(np.array(non_ego_labels), size=4, replace=False))
            ordered_labels = sorted(selected_labels, key=dist, reverse=True)
            all_permutations = list(itertools.permutations(selected_labels))
            distinct_permutations = random.sample(all_permutations, 4)
            if tuple(ordered_labels) not in distinct_permutations:
                distinct_permutations[-1] = tuple(ordered_labels)
            for idx, distinct_permutation in enumerate(distinct_permutations):
                new_tuple = [
                    f"<{val}>" for val in distinct_permutation
                ]
                new_tuple = ", ".join(new_tuple)
                distinct_permutations[idx] = new_tuple
            options = distinct_permutations
            answer_ordering = ", ".join([f"<{val}>" for val in ordered_labels])
            multiple_choice_options = create_options(options, 4, answer_ordering, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = fill_in_label(
                template_str=TEMPLATES["static"][question_type]["text"][0],
                replacement={code: str(selected_labels[idx])
                             for idx, code in enumerate(TEMPLATES["static"][question_type]["params"])
                             }
            )
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[answer_ordering]
            object1, object2, object3, object4 = [graph.get_node(label2id[label]) for label in ordered_labels]
            explanation = "The {} {}(<{}>) is at the far left, and the {} {}(<{}>) is at the far right. The {} {}(<{}>) and the {} {}(<{}>) are in between.".format(
                object1.color.lower(), NAMED_MAPPING[object1.type]["singular"], ordered_labels[0],
                object4.color.lower(), NAMED_MAPPING[object4.type]["singular"], ordered_labels[3],
                object2.color.lower(), NAMED_MAPPING[object2.type]["singular"], ordered_labels[1],
                object3.color.lower(), NAMED_MAPPING[object3.type]["singular"], ordered_labels[2]
            )
            ids_of_interest = [label2id[label] for label in selected_labels]
    elif question_type == "order_rightmost":
        def dist(label):
            o = graph.get_node(label2id[label])
            right_vec = np.array([ego_node.heading[1], -ego_node.heading[0]])
            return (np.array(o.pos) - np.array(ego_node.pos)).dot(right_vec)

        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:

            selected_labels = list(np.random.choice(np.array(non_ego_labels), size=4, replace=False))
            ordered_labels = sorted(selected_labels, key=dist, reverse=True)
            all_permutations = list(itertools.permutations(selected_labels))
            distinct_permutations = random.sample(all_permutations, 4)
            if tuple(ordered_labels) not in distinct_permutations:
                distinct_permutations[-1] = tuple(ordered_labels)
            for idx, distinct_permutation in enumerate(distinct_permutations):
                new_tuple = [
                    f"<{val}>" for val in distinct_permutation
                ]
                new_tuple = ", ".join(new_tuple)
                distinct_permutations[idx] = new_tuple
            options = distinct_permutations
            answer_ordering = ", ".join([f"<{val}>" for val in ordered_labels])
            multiple_choice_options = create_options(options, 4, answer_ordering, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = fill_in_label(
                template_str=TEMPLATES["static"][question_type]["text"][0],
                replacement={code: str(selected_labels[idx])
                             for idx, code in enumerate(TEMPLATES["static"][question_type]["params"])
                             }
            )
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[answer_ordering]
            object1, object2, object3, object4 = [graph.get_node(label2id[label]) for label in ordered_labels]
            explanation = "The {} {}(<{}>) is at the far right, and the {} {}(<{}>) is at the far left. The {} {}(<{}>) and the {} {}(<{}>) are in between.".format(
                object1.color.lower(), NAMED_MAPPING[object1.type]["singular"], ordered_labels[0],
                object4.color.lower(), NAMED_MAPPING[object4.type]["singular"], ordered_labels[3],
                object2.color.lower(), NAMED_MAPPING[object2.type]["singular"], ordered_labels[1],
                object3.color.lower(), NAMED_MAPPING[object3.type]["singular"], ordered_labels[2]
            )
            ids_of_interest = [label2id[label] for label in selected_labels]
    elif question_type == "order_frontmost":
        def dist(label):
            o = graph.get_node(label2id[label])
            front_vec = np.array(ego_node.heading)
            return (np.array(o.pos) - np.array(ego_node.pos)).dot(front_vec)

        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:

            selected_labels = list(np.random.choice(np.array(non_ego_labels), size=4, replace=False))
            ordered_labels = sorted(selected_labels, key=dist, reverse=True)
            all_permutations = list(itertools.permutations(selected_labels))
            distinct_permutations = random.sample(all_permutations, 4)
            if tuple(ordered_labels) not in distinct_permutations:
                distinct_permutations[-1] = tuple(ordered_labels)
            for idx, distinct_permutation in enumerate(distinct_permutations):
                new_tuple = [
                    f"<{val}>" for val in distinct_permutation
                ]
                new_tuple = ", ".join(new_tuple)
                distinct_permutations[idx] = new_tuple
            options = distinct_permutations
            answer_ordering = ", ".join([f"<{val}>" for val in ordered_labels])
            multiple_choice_options = create_options(options, 4, answer_ordering, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = fill_in_label(
                template_str=TEMPLATES["static"][question_type]["text"][0],
                replacement={code: str(selected_labels[idx])
                             for idx, code in enumerate(TEMPLATES["static"][question_type]["params"])
                             }
            )
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[answer_ordering]
            object1, object2, object3, object4 = [graph.get_node(label2id[label]) for label in ordered_labels]
            explanation = "The {} {}(<{}>) is at the furthest along our heading direction, and the {} {}(<{}>) is the closest. The {} {}(<{}>) and the {} {}(<{}>) are in between.".format(
                object1.color.lower(), NAMED_MAPPING[object1.type]["singular"], ordered_labels[0],
                object4.color.lower(), NAMED_MAPPING[object4.type]["singular"], ordered_labels[3],
                object2.color.lower(), NAMED_MAPPING[object2.type]["singular"], ordered_labels[1],
                object3.color.lower(), NAMED_MAPPING[object3.type]["singular"], ordered_labels[2]
            )
            ids_of_interest = [label2id[label] for label in selected_labels]
    elif question_type == "order_backmost":
        def dist(label):
            o = graph.get_node(label2id[label])
            back_vec = -np.array(ego_node.heading)
            return (np.array(o.pos) - np.array(ego_node.pos)).dot(back_vec)

        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:

            selected_labels = list(np.random.choice(np.array(non_ego_labels), size=4, replace=False))
            ordered_labels = sorted(selected_labels, key=dist, reverse=True)
            all_permutations = list(itertools.permutations(selected_labels))
            distinct_permutations = random.sample(all_permutations, 4)
            if tuple(ordered_labels) not in distinct_permutations:
                distinct_permutations[-1] = tuple(ordered_labels)
            for idx, distinct_permutation in enumerate(distinct_permutations):
                new_tuple = [
                    f"<{val}>" for val in distinct_permutation
                ]
                new_tuple = ", ".join(new_tuple)
                distinct_permutations[idx] = new_tuple
            options = distinct_permutations
            answer_ordering = ", ".join([f"<{val}>" for val in ordered_labels])
            multiple_choice_options = create_options(options, 4, answer_ordering, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = fill_in_label(
                template_str=TEMPLATES["static"][question_type]["text"][0],
                replacement={code: str(selected_labels[idx])
                             for idx, code in enumerate(TEMPLATES["static"][question_type]["params"])
                             }
            )
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[answer_ordering]
            object1, object2, object3, object4 = [graph.get_node(label2id[label]) for label in ordered_labels]
            explanation = "The {} {}(<{}>) is at the closest along our heading direction, and the {} {}(<{}>) is the furthest. The {} {}(<{}>) and the {} {}(<{}>) are in between.".format(
                object1.color.lower(), NAMED_MAPPING[object1.type]["singular"], ordered_labels[0],
                object4.color.lower(), NAMED_MAPPING[object4.type]["singular"], ordered_labels[3],
                object2.color.lower(), NAMED_MAPPING[object2.type]["singular"], ordered_labels[1],
                object3.color.lower(), NAMED_MAPPING[object3.type]["singular"], ordered_labels[2]
            )
            ids_of_interest = [label2id[label] for label in selected_labels]
    elif question_type == "identify_closest":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            objects = [graph.get_node(label2id[label]) for label in labels if label != -1]
            min_dist, closest_id = 10000, objects[0].id
            for o in objects:
                cur_dist = get_distance(o.pos, ego_node.pos)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_id = o.id
            question = TEMPLATES["static"][question_type]["text"][0]
            answer_label = find_label(closest_id, label2id)
            options = np.random.choice(np.array(non_ego_labels), size=3, replace=False)
            while answer_label in list(options) or -1 in list(options):
                options = np.random.choice(np.array(labels), size=3, replace=False)
            options = list(options) + [answer_label]
            options = [f"<{option}>" for option in options]
            answer_label = f"<{answer_label}>"
            multiple_choice_options = create_options(options, 4, answer_label, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[answer_label]
            object = graph.get_node(closest_id)
            color, type = object.color, object.type
            explanation = "The {} {}({}) is the closest labeled object from us.".format(
                color.lower(), NAMED_MAPPING[type]["singular"], answer_label)
    elif question_type == "identify_leftmost":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            objects = [graph.get_node(label2id[label]) for label in labels if label != -1]
            max_dist, closest_id = -10000, objects[0].id
            left_vec = -ego_node.heading[1], ego_node.heading[0]
            for o in objects:
                displacement = np.array(o.pos) - np.array(ego_node.pos)
                left_dist = displacement.dot(np.array(left_vec))
                if left_dist > max_dist:
                    max_dist = left_dist
                    closest_id = o.id
            question = TEMPLATES["static"][question_type]["text"][0]
            answer_label = find_label(closest_id, label2id)
            options = np.random.choice(np.array(non_ego_labels), size=3, replace=False)
            while answer_label in list(options) or -1 in list(options):
                options = np.random.choice(np.array(labels), size=3, replace=False)
            options = list(options) + [answer_label]
            options = [f"<{option}>" for option in options]
            answer_label = f"<{answer_label}>"
            multiple_choice_options = create_options(options, 4, answer_label, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[answer_label]
            object = graph.get_node(closest_id)
            color, type = object.color, object.type
            explanation = "The {} {}({}) is the leftmost labeled object from us.".format(
                color.lower(), NAMED_MAPPING[type]["singular"], answer_label)
    elif question_type == "identify_rightmost":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            objects = [graph.get_node(label2id[label]) for label in labels if label != -1]
            max_dist, closest_id = -10000, objects[0].id
            right_vec = ego_node.heading[1], -ego_node.heading[0]
            for o in objects:
                displacement = np.array(o.pos) - np.array(ego_node.pos)
                left_dist = displacement.dot(np.array(right_vec))
                if left_dist > max_dist:
                    max_dist = left_dist
                    closest_id = o.id
            question = TEMPLATES["static"][question_type]["text"][0]
            answer_label = find_label(closest_id, label2id)
            options = np.random.choice(np.array(non_ego_labels), size=3, replace=False)
            while answer_label in list(options) or -1 in list(options):
                options = np.random.choice(np.array(labels), size=3, replace=False)
            options = [f"<{option}>" for option in options]
            answer_label = f"<{answer_label}>"
            options = list(options) + [answer_label]
            multiple_choice_options = create_options(options, 4, answer_label, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[answer_label]
            object = graph.get_node(closest_id)
            color, type = object.color, object.type
            explanation = "The {} {}({}) is the rightest labeled object from us.".format(
                color.lower(), NAMED_MAPPING[type]["singular"], answer_label)
    elif question_type == "identify_frontmost":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            objects = [graph.get_node(label2id[label]) for label in labels if label != -1]
            max_dist, closest_id = -10000, objects[0].id
            front_vec = ego_node.heading
            for o in objects:
                displacement = np.array(o.pos) - np.array(ego_node.pos)
                left_dist = displacement.dot(np.array(front_vec))
                if left_dist > max_dist:
                    max_dist = left_dist
                    closest_id = o.id
            question = TEMPLATES["static"][question_type]["text"][0]
            answer_label = find_label(closest_id, label2id)
            options = np.random.choice(np.array(non_ego_labels), size=3, replace=False)
            while answer_label in list(options) or -1 in list(options):
                options = np.random.choice(np.array(labels), size=3, replace=False)
            options = list(options) + [answer_label]
            options = [f"<{option}>" for option in options]
            answer_label = f"<{answer_label}>"
            multiple_choice_options = create_options(options, 4, answer_label, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[answer_label]
            object = graph.get_node(closest_id)
            color, type = object.color, object.type
            explanation = "The {} {}({}) is the object furthest along front direction us.".format(
                color.lower(), NAMED_MAPPING[type]["singular"], answer_label)
    elif question_type == "identify_backmost":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            objects = [graph.get_node(label2id[label]) for label in labels if label != -1]
            max_dist, closest_id = -10000, objects[0].id
            back_vec = -ego_node.heading[0], -ego_node.heading[1]
            for o in objects:
                displacement = np.array(o.pos) - np.array(ego_node.pos)
                left_dist = displacement.dot(np.array(back_vec))
                if left_dist > max_dist:
                    max_dist = left_dist
                    closest_id = o.id
            question = TEMPLATES["static"][question_type]["text"][0]

            answer_label = find_label(closest_id, label2id)
            options = np.random.choice(np.array(non_ego_labels), size=3, replace=False)
            while answer_label in list(options) or -1 in list(options):
                options = np.random.choice(np.array(labels), size=3, replace=False)
            options = list(options) + [answer_label]
            options = [f"<{option}>" for option in options]
            answer_label = f"<{answer_label}>"
            multiple_choice_options = create_options(options, 4, answer_label, options)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            answer = answer2label[answer_label]
            object = graph.get_node(closest_id)
            color, type = object.color, object.type
            explanation = "The {} {}({}) is the object closest along the front direction of us.".format(
                color.lower(), NAMED_MAPPING[type]["singular"], answer_label)
    elif question_type == "describe_scenario":
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) < 1:
            print(f"Not enough items in the scene. Skip generating {question_type} for {frame_path}")
        else:
            def discrete_dist(distance):
                if 0 < distance <= 2:
                    # relative_dist = "very close"
                    answer = "very close"
                elif 2 < distance <= 10:
                    # relative_dist = "close"
                    answer = "close"
                elif 10 < distance <= 30:
                    # relative_dist = "medium"
                    answer = "medium"
                else:
                    # relative_dist = "far"
                    answer = "far"
                return answer

            type_space = [NAMED_MAPPING[obj]["singular"].capitalize() for obj in NAMED_MAPPING.keys() if
                          obj != "nil" and obj != "vehicle"]
            color_space = [color for color in NAMESPACE["color"]]
            pos_space = list(POSITION2CHOICE.values())
            dist_space = ["Very Close(0-2m)", "Close(2-10m)", "Medium(10-30m)", "Far(30m-)"]
            type_qualifier_string = "Choose type from [{}].".format("; ".join(type_space))
            color_qualifier_string = "Choose color from [{}].".format("; ".join(color_space))
            pos_qualifier_string = "Choose positional relationship from [{}].".format("; ".join(pos_space))
            dist_qualifier_string = "Choose distance from [{}].".format("; ".join(dist_space))
            qualifier = "\n".join(
                [type_qualifier_string, color_qualifier_string, pos_qualifier_string, dist_qualifier_string]
            )
            question = "\n".join([TEMPLATES["static"][question_type]["text"][0], qualifier])
            explanation = []
            for label in sorted(labels):
                if label == -1:
                    continue
                object = graph.get_node(label2id[label])
                color, type = object.color, object.type
                pos = ego_node.compute_relation_string(node=graph.get_node(object.id), ref_heading=ego_node.heading)
                distance = discrete_dist(get_distance(object.pos, ego_node.pos))
                heading = identify_heading(ego_node.pos, ego_node.heading)([[object]])[object.id]
                if type not in TYPES_WITHOUT_HEADINGS:
                    description_string = "{} {} {} positioned in our {} sector at {} distance. It heads toward our {} o'clock direction".format(
                        ("A" if color[0] not in ["A", "E", "I", "O", "U"] else "An") if len(color) > 0
                        else ("A" if NAMED_MAPPING[type]["singular"][0] not in ["a", "e", "i", "o", "u"] else "An"),
                        color.lower(),
                        NAMED_MAPPING[type]["singular"], POSITION2CHOICE[pos], distance, heading
                    )
                else:
                    description_string = "{} {} {} positioned in our {} sector at {} distance. Since it is {} {}, it doesn't have a heading".format(
                        ("A" if color[0] not in ["A", "E", "I", "O", "U"] else "An") if len(color) > 0
                        else ("A" if NAMED_MAPPING[type]["singular"][0] not in ["a", "e", "i", "o", "u"] else "An"),
                        color.lower(),
                        NAMED_MAPPING[type]["singular"],
                        POSITION2CHOICE[pos], distance,
                        "a" if NAMED_MAPPING[type]["singular"][0] not in ["a", "e", "i", "o", "u"] else "an",
                        NAMED_MAPPING[type]["singular"])
                explanation.append(
                    f"<{label}>: {description_string}."
                )
            explanation = "\n".join(explanation)
            answer = ""
    else:
        print('Something wrong!')
    if verbose:
        print(question)
        print(answer, explanation)
    return question, answer, explanation, ids_of_interest, option2answer


from collections import defaultdict


def parameterized_generate(frame_path, question_type, param, perspective="front", verbose=True,
                           id2label_path: str = None):
    identifier = os.path.basename(frame_path)
    world_path = os.path.join(frame_path, "world_{}.json".format(identifier))
    world = json.load(open(world_path, "r"))
    label2id, invalid_ids = enumerate_frame_labels(frame_path, perspective, id2label_path)
    labels = list(label2id.keys())
    ego_id, nodelist = nodify(world, multiview=False)
    graph = SceneGraph(ego_id, nodelist, frame_path)
    ego_node = graph.get_ego_node()
    question = None
    answer = None
    explanation = None
    ids_of_interest = []
    option2answer = {}
    if question_type == "describe_sector":
        sector = param["<pos>"]

        def satisfying_set(labels):
            for label in labels:
                if ego_node.compute_relation_string(
                        graph.get_node(label2id[label]),
                        ego_node.heading
                ) != sector:
                    return False
            return True

        selected_labels = [label for label in labels if label != -1]
        objects = [graph.get_node(label2id[label]) for label in selected_labels]
        pos_stat = defaultdict(lambda: set())
        for label_idx, object in enumerate(objects):
            dir = ego_node.compute_relation_string(node=object, ref_heading=ego_node.heading)
            pos_stat[dir].add(selected_labels[label_idx])

        answer_set = pos_stat[sector]
        if len(answer_set) > 4:
            answer_tuple = tuple(sorted(random.sample(list(answer_set), 4)))
            option_length = 4
        elif len(answer_set) > 0:
            answer_tuple = tuple(sorted(list(answer_set)))
            option_length = len(answer_set)
        else:
            answer_tuple = ()
            option_length = 4
        all_combinations = list(itertools.combinations(selected_labels, option_length))
        if len(all_combinations) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} with {sector} for {frame_path}")
        else:
            non_answer_combinations = []
            for combination in all_combinations:
                if not satisfying_set(combination):
                    non_answer_combinations.append(combination)
            if len(non_answer_combinations) < 3:
                print(f"Not enough items in the scene. Skip generating {question_type} with {sector} for {frame_path}")
            else:
                if answer_tuple == ():
                    distinct_combinations = random.sample(non_answer_combinations, 3)
                    options = [tuple(sorted(list(item))) for item in distinct_combinations]
                else:
                    distinct_combinations = random.sample(non_answer_combinations, 2)
                    options = [tuple(sorted(list(item))) for item in distinct_combinations]
                    options.append(())
                options.append(answer_tuple)
                old_options = deepcopy(options)
                for idx, distinct_combination in enumerate(options):
                    new_tuple = [
                        f"<{val}>" for val in distinct_combination
                    ]
                    new_tuple = ", ".join(new_tuple)
                    new_tuple = "".join(["[", new_tuple, "]"])
                    options[idx] = new_tuple
                answer_ordering = ", ".join([f"<{val}>" for val in answer_tuple])
                answer_ordering = "".join(["[", answer_ordering, "]"])
                multiple_choice_options = create_options(options, 4, answer_ordering, options)
                multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
                option2answer = {
                    val: key for key, val in answer2label.items()
                }
                question = fill_in_label(
                    template_str=TEMPLATES["static"][question_type]["text"][0],
                    replacement={
                        "<pos>": POSITION2CHOICE[param["<pos>"]]
                    }
                )
                question = " ".join(
                    [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
                answer = answer2label[answer_ordering]
                wrong_reasons = {
                    val: "" for val in answer2label.values()
                }
                for option_id, option in enumerate(old_options):
                    unsats = defaultdict(lambda: [])
                    if len(option) == 0 and len(answer_tuple) != 0:
                        wrong_choice = answer2label["[]"]
                        wrong_reasoning = "there exists at least {} objects({}) in the specified({}) sector".format(
                            len(answer_tuple),
                            ", ".join([f"<{l}>" for l in answer_tuple]),
                            POSITION2CHOICE[sector]
                        )
                        wrong_reasons[wrong_choice] = wrong_reasoning
                        continue
                    for label in option:
                        pos_str = ego_node.compute_relation_string(
                            node=graph.get_node(label2id[label]),
                            ref_heading=ego_node.heading)
                        if pos_str != sector:
                            unsats[pos_str].append(label)
                    if len(unsats.keys()) > 0:
                        ordering = ", ".join([f"<{val}>" for val in option])
                        ordering = "".join(["[", ordering, "]"])
                        wrong_choice = answer2label[ordering]
                        wrong_reasonings = []
                        for pos, stuff in unsats.items():
                            if len(stuff) > 1:
                                first_plurality = True
                            wrong_reasoning = "{} object{}({}) in the {} sector".format(
                                len(stuff), "" if len(stuff) <= 1 else "s", ", ".join([f"<{l}>" for l in stuff]),
                                POSITION2CHOICE[pos]
                            )
                            wrong_reasonings.append(wrong_reasoning)

                        if len(wrong_reasonings) == 1:
                            wrong_str = "".join(["there exists ", wrong_reasonings[0]])
                        elif len(wrong_reasonings) == 2:
                            wrong_str = " and ".join([wrong_reasonings[0], wrong_reasonings[-1]])
                            wrong_str = "".join(["there exists ", wrong_str])
                        else:
                            wrong_str = ", ".join(wrong_reasonings[:-1])
                            wrong_str = " and ".join([wrong_str, wrong_reasonings[-1]])
                            wrong_str = "".join(["there exists ", wrong_str])
                        wrong_reasons[wrong_choice] = wrong_str
                explanation = []
                for choice, wrong_reason in wrong_reasons.items():
                    if choice == answer:
                        continue
                    sentence = f"Option ({choice}) is wrong since {wrong_reason}"
                    explanation.append(sentence)
                explanation = "; ".join(explanation)
                explanation = "".join([explanation, "."])
                ids_of_interest = []
    elif question_type == "describe_distance":
        sector = param["<dist>"]

        def discretize_dist(abs_dist):
            if abs_dist < 2:
                discrete_dist = "very close"
            elif abs_dist < 10:
                discrete_dist = "close"
            elif abs_dist < 30:
                discrete_dist = "medium"
            else:
                discrete_dist = "far"
            return discrete_dist

        def satisfying_set(labels):
            for label in labels:
                abs_dist = get_distance(ego_node.pos, graph.get_node(label2id[label]).pos)
                if discretize_dist(abs_dist) != sector:
                    return False
            return True

        selected_labels = [label for label in labels if label != -1]
        objects = [graph.get_node(label2id[label]) for label in selected_labels]
        dist_stat = defaultdict(lambda: set())
        for label_idx, object in enumerate(objects):
            discrete_dist = discretize_dist(
                get_distance(object.pos, ego_node.pos)
            )
            dist_stat[discrete_dist].add(selected_labels[label_idx])

        answer_set = dist_stat[sector]
        if len(answer_set) > 4:
            answer_tuple = tuple(sorted(random.sample(list(answer_set), 4)))
            option_length = 4
        elif len(answer_set) > 0:
            answer_tuple = tuple(sorted(list(answer_set)))
            option_length = len(answer_set)
        else:
            answer_tuple = ()
            option_length = 4
        all_combinations = list(itertools.combinations(selected_labels, option_length))
        if len(all_combinations) < 4:
            print(f"Not enough items in the scene. Skip generating {question_type} with {sector} for {frame_path}")
        else:
            non_answer_combinations = []
            for combination in all_combinations:
                if not satisfying_set(combination):
                    non_answer_combinations.append(combination)
            if len(non_answer_combinations) < 3:
                print(f"Not enough items in the scene. Skip generating {question_type} with {sector} for {frame_path}")
            else:
                if answer_tuple == ():
                    distinct_combinations = random.sample(non_answer_combinations, 3)
                    options = [tuple(sorted(list(item))) for item in distinct_combinations]
                else:
                    distinct_combinations = random.sample(non_answer_combinations, 2)
                    options = [tuple(sorted(list(item))) for item in distinct_combinations]
                    options.append(())
                options.append(answer_tuple)
                old_options = deepcopy(options)
                for idx, distinct_combination in enumerate(options):
                    new_tuple = [
                        f"<{val}>" for val in distinct_combination
                    ]
                    new_tuple = ", ".join(new_tuple)
                    new_tuple = "".join(["[", new_tuple, "]"])
                    options[idx] = new_tuple
                answer_ordering = ", ".join([f"<{val}>" for val in answer_tuple])
                answer_ordering = "".join(["[", answer_ordering, "]"])
                multiple_choice_options = create_options(options, 4, answer_ordering, options)
                multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
                option2answer = {
                    val: key for key, val in answer2label.items()
                }
                question = fill_in_label(
                    template_str=TEMPLATES["static"][question_type]["text"][0],
                    replacement={
                        "<dist>": param["<dist>"]
                    }
                )
                question = " ".join(
                    [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
                answer = answer2label[answer_ordering]
                wrong_reasons = {
                    val: "" for val in answer2label.values()
                }
                for option_id, option in enumerate(old_options):
                    unsats = defaultdict(lambda: [])
                    if len(option) == 0 and len(answer_tuple) != 0:
                        wrong_choice = answer2label["[]"]
                        wrong_reasoning = "there exists at least {} objects({}) positioned at specified({}) distance from us".format(
                            len(answer_tuple),
                            ", ".join([f"<{l}>" for l in answer_tuple]),
                            sector
                        )
                        wrong_reasons[wrong_choice] = wrong_reasoning
                        continue
                    for label in option:
                        dist_str = discretize_dist(
                            get_distance(graph.get_node(label2id[label]).pos, ego_node.pos)
                        )
                        if dist_str != sector:
                            unsats[dist_str].append(label)
                    if len(unsats.keys()) > 0:
                        ordering = ", ".join([f"<{val}>" for val in option])
                        ordering = "".join(["[", ordering, "]"])
                        wrong_choice = answer2label[ordering]
                        wrong_reasonings = []
                        for dist, stuff in unsats.items():
                            if len(stuff) > 1:
                                first_plurality = True
                            wrong_reasoning = "{} object{}({}) positioned at {} distance from us".format(
                                len(stuff), "" if len(stuff) <= 1 else "s", ", ".join([f"<{l}>" for l in stuff]),
                                dist
                            )
                            wrong_reasonings.append(wrong_reasoning)

                        if len(wrong_reasonings) == 1:
                            wrong_str = "".join(["there exists ", wrong_reasonings[0]])
                        elif len(wrong_reasonings) == 2:
                            wrong_str = " and ".join([wrong_reasonings[0], wrong_reasonings[-1]])
                            wrong_str = "".join(["there exists ", wrong_str])
                        else:
                            wrong_str = ", ".join(wrong_reasonings[:-1])
                            wrong_str = " and ".join([wrong_str, wrong_reasonings[-1]])
                            wrong_str = "".join(["there exists ", wrong_str])
                        wrong_reasons[wrong_choice] = wrong_str
                explanation = []
                for choice, wrong_reason in wrong_reasons.items():
                    if choice == answer:
                        continue
                    sentence = f"Option ({choice}) is wrong since {wrong_reason}"
                    explanation.append(sentence)
                explanation = "; ".join(explanation)
                explanation = "".join([explanation, "."])
                ids_of_interest = []
    else:
        print("Not yet implemented")
        exit()
    if verbose:
        print(question)
        print(answer)
        print(explanation)
    return question, answer, explanation, ids_of_interest, option2answer


def cfg_adapater(frame_path, question_type, perspective, verbose, id2label_path: str = None):
    identifier = os.path.basename(frame_path)
    world_path = os.path.join(frame_path, "world_{}.json".format(identifier))
    world = json.load(open(world_path, "r"))
    label2id, invalid_ids = enumerate_frame_labels(frame_path, perspective, id2label_path)

    records, counts = generate_all_frame(
        templates=
        {"counting": {
            "text": [
                "How many <o> can be observed at this moment?",
                "How many <o> are there in observation currently?",
                "Give me the number of <o> at this moment.",
                "Count the number of observed <o> at current moment.",
                "Determine how many <o> are observed at current moment.",
                "Record the quantity of <o> at current moment."
            ],
            "params": [
                "<o>"
            ],
            "end_filter": "count",
            "constraint": []}}
        , frame=world_path, attempts=100, max=4, id_start=0, verbose=verbose, multiview=False)
    for id, record in records.items():
        print(id, record)


def verify_grounding(frame_path, perspective, verbose, id2label_path: str = None, box: bool = False):
    identifier = os.path.basename(frame_path)
    world_path = os.path.join(frame_path, "world_{}.json".format(identifier))
    world = json.load(open(world_path, "r"))
    label2id, invalid_ids = enumerate_frame_labels(frame_path, perspective, id2label_path)
    labels = list(label2id.keys())
    non_ego_labels = [label for label in labels if label != -1]
    records = {}
    if len(non_ego_labels) < 0:
        print("Too few objects for grounding")
    else:
        randomly_selected_labels = random.sample(non_ego_labels, min(len(non_ego_labels), 4))
        idx = 0
        for selected_label in tqdm.tqdm(randomly_selected_labels, desc="Generating Grounding Question",
                                        unit="question"):
            solo_labeled_path = os.path.join(frame_path, f"label_{selected_label}_{perspective}_{identifier}.png")
            random_new_labels = list(np.random.choice(20, size=4, replace=False))
            random_new_labels = [int(item) for item in random_new_labels]
            random_new_label = random_new_labels[0]

            new_id2l = {label2id[selected_label]: random_new_label}
            assert new_id2l is not None
            labelframe(
                frame_path=frame_path, perspective="front", save_path=solo_labeled_path,
                query_ids=[label2id[selected_label]], id2l=new_id2l,
                font_scale=1.25, grounding=True, bounding_box=box
            )
            multiple_choice_options = create_options(random_new_labels, 4, random_new_label, random_new_labels)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            answer = answer2label[random_new_label]
            question = f"What is the numerical label of the white-contoured object? Choose the best answer from option (A) through (D): {multiple_choice_string}"
            explanation = ""
            records[idx] = dict(
                question=question, answer=answer, explanation=explanation, type="grounding", objects=[],
                world=[frame_path], obs=[solo_labeled_path], options=option2answer
            )
            idx += 1
    return records


import re


def replace_substrs(original_string, mapping):
    # Define the pattern to match <n> where n is an integer
    pattern = r'<(\d+)>'

    # Function to use for substitution
    def replacer(match):
        # Extract the number n from <n>
        n = int(match.group(1))
        # Check if n is in the mapping and replace with its mapped value
        return f"<{str(mapping.get(n, match.group(0)))}>"

    # Substitute using re.sub with the replacer function
    replaced_string = re.sub(pattern, replacer, original_string)

    return replaced_string


import glob

import tqdm


def batch_generate_static(world_paths, save_path="./", verbose=False, perspective="front", labeled=False, proc_id=0,
                          box=False):
    frame_paths = [os.path.dirname(world_path) for world_path in world_paths]
    template_path = os.path.join(current_directory, "questions_templates.json")
    static_templates = TEMPLATES["static"]  #json.load(open(template_path, "r"))["static"]
    records = {}
    current_type = ""
    current_frame = ""
    try:
        count = 0
        frame_paths = frame_paths
        for frame_path in tqdm.tqdm(frame_paths, desc=f"Processing {proc_id}", unit="frame"):
            current_frame = frame_path
            frame_records = {}
            frame_id = 0
            identifier = os.path.basename(frame_path)
            static_id2label_path = os.path.join(frame_path, f"static_id2label_{perspective}_{identifier}.json")
            static_labeled_path = os.path.join(frame_path, f"static_labeled_{perspective}_{identifier}.png")
            if not labeled or not os.path.exists(static_labeled_path):
                if verbose:
                    print(f"Creating single-frame-consistent labelling for {frame_path}")
                result = static_id2label(frame_path, perspective)
                static_id2l = result  #json.load(open(static_id2label_path, "r"))
                if static_id2l is None:
                    print("why>>>???")
                assert static_id2l is not None
                labelframe(frame_path=frame_path, perspective=perspective, save_path=static_labeled_path,
                           id2l=static_id2l,
                           font_scale=1.25, bounding_box=box)
            else:
                #raise Exception
                if verbose:
                    print(f"Already have labelled version FOR {frame_path}")
                static_id2l = json.load(open(static_id2label_path, "r"))
            queried_ids = set()
            for question_type in tqdm.tqdm(static_templates.keys(), desc="Generating Questions", unit="type"):
                current_type = question_type
                if question_type not in ["describe_sector", "describe_distance"]:
                    question, answer, explanation, ids_of_interest, option2answer = generate(frame_path=frame_path,
                                                                                             question_type=question_type,
                                                                                             perspective=perspective,
                                                                                             verbose=verbose,
                                                                                             id2label_path=static_id2label_path)
                    if question is not None and answer is not None and explanation is not None:
                        frame_records[frame_id] = dict(
                            question=question, answer=answer, explanation=explanation,
                            type=question_type, objects=ids_of_interest, world=[frame_path],
                            obs=[static_labeled_path], options=option2answer
                        )
                        frame_id += 1
                        for id in ids_of_interest:
                            queried_ids.add(id)
                else:
                    if question_type == "describe_sector":
                        params = [
                            {"<pos>": pos} for pos in ["lf", "rf", "f"]
                        ]
                    else:
                        params = [
                            {"<dist>": dist} for dist in ["very close", "close", "medium", "far"]
                        ]
                    for param in params:
                        question, answer, explanation, ids_of_interest, option2answer = parameterized_generate(
                            frame_path=frame_path,
                            question_type=question_type,
                            param=param,
                            perspective=perspective,
                            verbose=verbose,
                            id2label_path=static_id2label_path)
                        if question is not None and answer is not None and explanation is not None:
                            frame_records[frame_id] = dict(
                                question=question, answer=answer, explanation=explanation,
                                type=question_type, objects=ids_of_interest, world=[frame_path],
                                obs=[static_labeled_path], options=option2answer
                            )
                            frame_id += 1
            new_id2label = {object_id: i for i, object_id in enumerate(queried_ids)}
            new_labeled_path = os.path.join(frame_path, f"static_qa_labeled_{perspective}_{identifier}.png")
            original_id2label = static_id2l
            assert new_id2label is not None
            labelframe(
                frame_path=frame_path, perspective="front", save_path=new_labeled_path,
                query_ids=list(queried_ids), id2l=new_id2label, font_scale=1.25, bounding_box=box
            )
            for qid, record in frame_records.items():
                if record["type"] not in ["identify_closest", "identify_leftmost", "identify_rightmost",
                                          "identify_frontmost", "identify_backmost", "describe_scenario",
                                          "describe_sector", "describe_distance"]:
                    old2new = {
                        original_id2label[object_id]: new_id2label[object_id] for object_id in record["objects"]
                    }
                    record["question"] = replace_substrs(record["question"], old2new)
                    record["explanation"] = replace_substrs(record["explanation"], old2new)
                    for opt in record["options"].keys():
                        record["options"][opt] = replace_substrs(record["options"][opt], old2new)
                    record["obs"] = [new_labeled_path]
                records[qid + count] = record
            count += frame_id
            grounding_records = verify_grounding(frame_path=frame_path, perspective=perspective, verbose=verbose,
                                                 id2label_path=static_id2label_path, box=box)
            for g_id, record in grounding_records.items():
                records[g_id + count] = record
            count += len(grounding_records)
    except Exception as e:
        print("Something Wrong! save partial results")
        print(f"Encountered issue at {current_frame},{current_type}")
        print(e)
        var = traceback.format_exc()
        debug_path = os.path.join(
            os.path.dirname(save_path),
            f"{proc_id}_debug.json"
        )
        json.dump(
            {"proc_id": proc_id, "end_frame": current_frame, "end_quetion": current_type, "error": str(e),
             "generated": len(records), "trace": str(var)},
            open(debug_path, "w"),
            indent=2
        )
        raise (e)
    finally:
        json.dump(records, open(save_path, "w"), indent=2)


def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


import math

import multiprocessing as multp


def multiprocess_generate_static(session_path, save_path="./", verbose=False, perspective="front", labeled=False,
                                 num_proc=1, box=False):
    def find_worlds(session_path):
        pattern = os.path.join(session_path, "**", "**", "world**.json")
        matching_frames = glob.glob(pattern)
        return matching_frames

    world_paths = find_worlds(session_path)
    world_paths = world_paths[::5]
    print(f"Working on {len(world_paths)} frames.")
    chunk_size = math.ceil(len(world_paths) / num_proc)
    job_chunks = split_list(world_paths, chunk_size)
    print(f"{len(world_paths)} frames distributed across {num_proc} processes, {chunk_size} MAX each process")
    processes = []
    name_root = os.path.basename(save_path)
    save_dir = os.path.dirname(save_path)
    for proc_id in range(num_proc):
        print(f"Sending job {proc_id}")
        name = f"{proc_id}_{name_root}"
        p = multp.Process(
            target=batch_generate_static,
            args=(
                job_chunks[proc_id],
                os.path.join(save_dir, name),
                verbose,
                perspective,
                labeled, proc_id, box
            )
        )
        print(f"Successfully sent {proc_id}")
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


if __name__ == "__main__":
    #frame_path = "/bigdata/weizhen/repo/qa_platform/public/test_wide/scene-0061_91_100/0_91"
    #"/bigdata/weizhen/repo/qa_platform/public/test_wide/scene-0061_46_55/0_48/"
    #episode_path = os.path.dirname(frame_path)
    #asset_path = "/test_wide/scene-0061_91_100/0_91"
    #obs = os.path.join(asset_path, "labeled_front_{}.png".format(os.path.basename(frame_path)))
    #batch_generate_static("/bigdata/weizhen/repo/qa_platform/public/test", verbose=False,
    #                      save_path="/bigdata/weizhen/repo/qa_platform/public/data_small.json", labeled=True)
    NUSC = True
    if NUSC:
        #NAMED_MAPPING = {
        #    val: dict(
        #        singular=val,
        #        plural=val
        #    ) for val in ALL_TYPE.values()
        #}
        TEMPLATES["static"] = {
            key: val for key, val in TEMPLATES["static"].items() if key not in ["identify_color"]
        }

    multiprocess_generate_static(
        session_path="/bigdata/weizhen/metavqa_iclr/scenarios/nusc_real",
        save_path="/home/weizhen/main_meta/test.json",
        num_proc=1,
        labeled=True,
        box=NUSC
    )
    "/bigdata/weizhen/metavqa_iclr/scenarios/nusc_real/scene-0596_0_40/26_17"
    #generate(frame_path, "describe_scenario", "front", verbose=True)
    #parameterized_generate(frame_path, "describe_distance", {"<dist>":"medium"}, "front", True)
