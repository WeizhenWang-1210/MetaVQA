import os, json, random, itertools, traceback, glob, tqdm, math, copy, argparse
import numpy as np
import multiprocessing as multp
from som.grounding_question import generate_grounding, grounding_ablations, SETTINGS
from som.parameterized_questions import parameterized_generate
from som.qa_utils import create_options, create_multiple_choice, split_list, find_label, replace_substrs
from som.utils import enumerate_frame_labels, get, fill_in_label
from som.masking import labelframe, static_id2label
from vqa.object_node import nodify, extrapolate_bounding_boxes, box_trajectories_overlap, box_trajectories_intersect
from vqa.scene_graph import SceneGraph
from vqa.dataset_utils import get_distance
from vqa.dataset_utils import transform_heading
from vqa.configs.NAMESPACE import NAMESPACE, POSITION2CHOICE



current_directory = os.path.dirname(os.path.abspath(__file__))
NAMED_MAPPING = dict(
                nil=dict(singular="traffic object", plural="traffic objects"),
                Bus=dict(singular="bus", plural="buses"),
                Caravan=dict(singular="caravan", plural="caravans"),
                Coupe=dict(singular="coupe", plural="coupes"),
                FireTruck=dict(singular="fire engine", plural="fire engines"),
                Jeep=dict(singular="jeep", plural="jeeps"),
                Pickup=dict(singular="pickup", plural="pickups"),
                Policecar=dict(singular="police car", plural="police cars"),
                SUV=dict(singular="SUV", plural="SUVs"),
                SchoolBus=dict(singular="school bus", plural="school buses"),
                Sedan=dict(singular="sedan", plural="sedans"),
                SportCar=dict(singular="sports car", plural="sports cars"),
                Truck=dict(singular="truck", plural="trucks"),
                Hatchback=dict(singular="hatchback", plural="hatchbacks"),
                Pedestrian=dict(singular="pedestrian", plural="pedestrians"),
                vehicle=dict(singular="vehicle", plural="vehicles"),
                Bike=dict(singular="bike", plural="bikes"),
                Barrier=dict(singular="traffic barrier", plural="traffic barriers"),
                Warning=dict(singular="warning sign", plural="warning signs"),
                Cone=dict(singular="traffic cone", plural="traffic cones"),
                #nusc additions
                Wheelchair=dict(singular="wheel chair", plural="wheel chairs"),
                Police_officer=dict(singular="police officer", plural="police officers"),
                Construction_worker=dict(singular="construction worker", plural="construction workers"),
                Animal=dict(singular="animal", plural="animals"),
                Car=dict(singular="car", plural="cars"),
                Motorcycle=dict(singular="motorcycle", plural="motorcycles"),
                Construction_vehicle=dict(singular="construction vehicle", plural="construction vehicles"),
                Ambulance=dict(singular="ambulance", plural="ambulances"),
                Trailer=dict(singular="trailer", plural="trailers"),
                Stroller=dict(singular="stroller", plural="strollers")
            )
FONT_SCALE=1
BACKGROUND=(0,0,0)
USEBOX=True
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
CLOCK_TO_SECTOR = {1:"rf", 2:"rf", 3:"r", 4:"rb", 5:"rb", 6:"b", 7:"lb", 8:"lb",9:"l", 10:"lf", 11:"lf", 12:"f"}
SECTORS = ["rf", "rb", "lb", "lf", "f", "r", "b", "l"]



#{
# l->2,3,4, lf->4,5, f->5,6,7, rf->7,8 -> r->8,9,10, rb->10 11 b->11 12 1 lb->1 2, m-determine if rays intersect with my line
# }
#
# 12->f, 1->rf, 2->rf, 3->r, 4->rb, 5->rb, 6 ->b, 7->lf, 8->lb 9->l, 10 lf, 11lf


# 345, 15 -> front; 15-75 right-front; 75->105 right; 105-165->right back; 165-195->back; 195->255->left back; 255->285 left; 285, 345->left front
#

#TODO refactor this ugly code.

def identify_angle(origin_pos, origin_heading):
    def helper(search_spaces):
        import numpy as np
        result = {}
        for search_space in search_spaces:
            for object in search_space:
                angle_rotated = transform_heading(
                    object.heading, origin_pos, origin_heading
                )
                angle = 2 * np.pi - angle_rotated  #so the range is now 0 to 360.
                angle = angle % (2 * np.pi)
                angle = np.degrees(angle)
                result[object.id] = angle
        return result
    return helper


def angle2sector(degree):
    assert 0<=degree<=360
    if degree < 15 or degree > 345:
        return "f"
    elif 15<degree<75:
        return "rf"
    elif 75<degree<105:
        return "r"
    elif 105<degree<165:
        return "rb"
    elif 165<degree<195:
        return "b"
    elif 195<degree<255:
        return "lb"
    elif 255<degree<285:
        return "l"
    elif 285<degree<345:
        return "lf"


def select_not_from(space, forbidden, population=1):
    unique_types = set(space)
    unique_forbiddens = set(forbidden)
    assert len(space)>len(forbidden)
    diff = unique_types.difference(unique_forbiddens)
    assert len(diff) >= population
    return random.sample(list(diff), population)


def generate(frame_path: str, question_type: str, perspective: str = "front", verbose: bool = False,
             id2label_path: str = None):
    identifier = os.path.basename(frame_path)
    world_path = os.path.join(frame_path, "world_{}.json".format(identifier))
    world = json.load(open(world_path, "r"))
    label2id, invalid_ids = enumerate_frame_labels(frame_path, perspective, id2label_path)
    labels = list(label2id.keys())
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
            #Remove synonyms
            if "vehicle" in multiple_choice_options and "car" in multiple_choice_options:
                if NAMED_MAPPING[type]["singular"] == "vehicle":
                    index = multiple_choice_options.index("car")
                else:
                    index = multiple_choice_options.index("vehicle")
                multiple_choice_options[index] = select_not_from(type_space,multiple_choice_options)[-1]
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
            explanation = f"The type of this{color_str}object (<{label_str}>) {pos_str} of us is {type_str}."

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
            explanation = f"The{color_str}{type_str} (<{label_str}>) is {round(distance)} meters {pos_str} us. Therefore, it belongs to \"{option2answer[answer]}\"."
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
            explanation = "The {} {} (<{}>) is {} us.".format(color.lower(), NAMED_MAPPING[type]["singular"],
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
            heading = identify_angle(ego_node.pos, ego_node.heading)([[object]])
            heading = heading[object.id]
            #heading_clock = heading
            #heading = CLOCK_TO_SECTOR[heading]
            heading_angle = heading
            heading = angle2sector(heading)

            heading_idx = SECTORS.index(heading)
            if heading_idx < 4:
                options = SECTORS[:4]
            else:
                options = SECTORS[4:]
            assert len(options) == 4 and heading in options
            transform = lambda x: "{}".format(POSITION2CHOICE[x])
            multiple_choice_options = create_options(options, 4, heading, SECTORS, transform)
            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            color, type = object.color, object.type
            pos = ego_node.compute_relation_string(node=graph.get_node(object.id),
                                                   ref_heading=ego_node.heading)
            answer = answer2label[transform(heading)]
            question = " ".join(
                [question, "Choose the best answer from option (A) through (D): {}".format(multiple_choice_string)])
            explanation = "The {} {} (<{}>) {} us is facing our {} direction.".format(color.lower(),
                                                                                             NAMED_MAPPING[type][
                                                                                                 "singular"],
                                                                                             selected_label,
                                                                                             DIRECTION_MAPPING[pos],
                                                                                             POSITION2CHOICE[heading])
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
                explanation = ("Object <{}>, a {} {} located to our {}, is about the same distance from us as object <{}>, a {} {} located to our {}.").format(
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
            heading = identify_angle(ego_node.pos, ego_node.heading)([[object]])
            heading = heading[object.id]
            heading = angle2sector(heading)
            heading = POSITION2CHOICE[heading]

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
                explanation = "Yes, this {} {} (<{}>) {} us and heading toward our {} direction will run into us if it drives along its current heading.".format(
                    color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)
            else:
                explanation = "No, this {} {} (<{}>) {} us and heading toward our {} direction will not run into us if it drives along its current heading.".format(
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
            heading = identify_angle(ego_node.pos, ego_node.heading)([[object]])
            heading = heading[object.id]
            heading = angle2sector(heading)
            heading = POSITION2CHOICE[heading]
            pos = ego_node.compute_relation_string(node=graph.get_node(object.id),
                                                   ref_heading=ego_node.heading)
            if crash:
                explanation = "Yes, this {} {} (<{}>) {} us and heading toward our {} direction will run into us if it proceed along its current heading.".format(
                    color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)
            else:
                if intersect:
                    explanation = "No, this {} {} (<{}>) {} us and heading toward our {} direction will not run into us even though our paths will intersect.".format(
                        color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)
                else:
                    explanation = "No, this {} {} (<{}>) {} us and heading toward our {} direction will not run into us.".format(
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
            explanation = "The {} {} (<{}>) is closest to us, and the {} {}(<{}>) is furthest away from us. The {} {}(<{}>) and the {} {}(<{}>) are in between.".format(
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
            explanation = "The {} {} (<{}>) is at the far left, and the {} {}(<{}>) is at the far right. The {} {}(<{}>) and the {} {}(<{}>) are in between.".format(
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
            explanation = "The {} {} (<{}>) is at the far right, and the {} {}(<{}>) is at the far left. The {} {}(<{}>) and the {} {}(<{}>) are in between.".format(
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
            explanation = "The {} {} (<{}>) is at the furthest along our heading direction, and the {} {}(<{}>) is the closest. The {} {}(<{}>) and the {} {}(<{}>) are in between.".format(
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
            explanation = "The {} {} (<{}>) is at the closest along our heading direction, and the {} {}(<{}>) is the furthest. The {} {}(<{}>) and the {} {}(<{}>) are in between.".format(
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
            explanation = "The {} {} ({}) is the closest labeled object from us.".format(
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
            explanation = "The {} {} ({}) is the leftmost labeled object from us.".format(
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
            explanation = "The {} {} ({}) is the rightest labeled object from us.".format(
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
            explanation = "The {} {} ({}) is the object furthest along front direction us.".format(
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
            explanation = "The {} {} ({}) is the object closest along the front direction of us.".format(
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
                heading = identify_angle(ego_node.pos, ego_node.heading)([[object]])[object.id]
                heading = POSITION2CHOICE[angle2sector(heading)]
                if type not in TYPES_WITHOUT_HEADINGS:
                    description_string = "{} {} {} positioned in our {} sector at {} distance. It heads toward our {} direction".format(
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
        print('Unknown Question Type')
    if verbose:
        print(question)
        print(answer, explanation)
    return question, answer, explanation, ids_of_interest, option2answer


def batch_generate_static(world_paths, save_path="./", verbose=False, perspective="front", labeled=False, proc_id=0,
                          box=USEBOX, domain="sim"):
    frame_paths = [os.path.dirname(world_path) for world_path in world_paths]
    #template_path = os.path.join(current_directory, "questions_templates.json")
    static_templates = TEMPLATES["static"]  #json.load(open(template_path, "r"))["static"]
    records = {}
    current_type = ""
    current_frame = ""
    try:
        count = 0
        frame_paths = frame_paths
        for frame_path in tqdm.tqdm(frame_paths, desc=f"Proc-{proc_id}", unit="frame"):
            # First, create som-annotated images.
            current_frame = frame_path
            frame_records = {}
            frame_id = 0
            identifier = os.path.basename(frame_path)
            static_id2label_path = os.path.join(frame_path, f"static_id2label_{perspective}_{identifier}.json")
            static_labeled_path = os.path.join(frame_path, f"static_labeled_{perspective}_{identifier}.png")
            if domain == "real":
                id2corners = json.load(open(os.path.join(frame_path, f"id2corners_{identifier}.json"),"r"))
            else:
                id2corners = None
            if not labeled or not os.path.exists(static_labeled_path):
                if verbose:
                    print(f"Creating single-frame-consistent labelling for {frame_path}")
                result = static_id2label(frame_path, perspective)
                static_id2l = result  #json.load(open(static_id2label_path, "r"))
                assert static_id2l is not None, "static_id2l is None"
                labelframe(frame_path=frame_path, perspective=perspective, save_path=static_labeled_path,
                           id2l=static_id2l, id2corners=id2corners,
                           font_scale=FONT_SCALE, bounding_box=USEBOX, background_color=BACKGROUND)
            else:
                if verbose:
                    print(f"Already have labelled version FOR {frame_path}")
                static_id2l = json.load(open(static_id2label_path, "r"))
            queried_ids = set()
            for question_type in tqdm.tqdm(static_templates.keys(), desc="Generating Questions", unit="type"):
                current_type = question_type
                if question_type not in ["describe_sector", "describe_distance", "embodied_distance", "embodied_sideness", "embodied_collision"]:
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
                    elif question_type == "describe_distance":
                        params = [
                            {"<dist>": dist} for dist in ["very close", "close", "medium", "far"]
                        ]
                    elif question_type == "embodied_distance" :
                        speeds= [2,5,15,25]  #0-10 mph, 10-30 mph,  30-50 mph, 50mph+
                                              #0-4.47 m/s, 4.47-13.41 m/s, 13.41-22.35 m/s, 22.35m/s +
                        actions = [0,1,2,3,4]#[0,1,2,3,4]
                        durations = [5,10,15,20]#[5]#,10,15,20]
                        configs = list(itertools.product(speeds, actions, durations))
                        configs = random.sample(configs, k=4)
                        params = [
                            {"<speed>": speed, "<action>":action, "<duration>": duration} for (speed, action, duration) in configs
                        ]
                    elif question_type == "embodied_sideness":
                        speeds = [2, 5, 15, 25]  # 0-10 mph, 10-30 mph,  30-50 mph, 50mph+
                        # 0-4.47 m/s, 4.47-13.41 m/s, 13.41-22.35 m/s, 22.35m/s +
                        actions = [0, 1, 2, 3, 4]  # [0,1,2,3,4]
                        durations = [5, 10, 15, 20]  # [5]#,10,15,20]
                        configs = list(itertools.product(speeds, actions, durations))
                        configs = random.sample(configs, k=4)
                        params = [
                            {"<speed>": speed, "<action>": action, "<duration>": duration} for (speed, action, duration)
                            in configs
                        ]
                    elif question_type == "embodied_collision":
                        speeds = [2, 5, 15, 25]  # 0-10 mph, 10-30 mph,  30-50 mph, 50mph+
                        # 0-4.47 m/s, 4.47-13.41 m/s, 13.41-22.35 m/s, 22.35m/s +
                        actions = [0, 1, 2, 3, 4]  # [0,1,2,3,4]
                        durations = [5, 10, 15, 20]  # [5]#,10,15,20]
                        configs = list(itertools.product(speeds, actions, durations))
                        configs = random.sample(configs, k=4)
                        params = [
                            {"<speed>": speed, "<action>": action, "<duration>": duration} for (speed, action, duration)
                            in configs
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
                        for id in ids_of_interest:
                            queried_ids.add(id)
            new_id2label = {object_id: i for i, object_id in enumerate(queried_ids)}
            new_labeled_path = os.path.join(frame_path, f"static_qa_labeled_{perspective}_{identifier}.png")
            original_id2label = static_id2l
            assert new_id2label is not None
            #New labelling to leave out unferred objects.
            labelframe(
                frame_path=frame_path, perspective="front", save_path=new_labeled_path,  query_ids=list(queried_ids),
                id2l=new_id2label, id2corners=id2corners, font_scale=FONT_SCALE, bounding_box=USEBOX, background_color=BACKGROUND
            )
            #Determine which questions should have labelling replaced.
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
            #Seperate function to generate grounding questions
            grounding_records = generate_grounding(frame_path=frame_path, perspective=perspective, verbose=verbose,
                                                 id2label_path=static_id2label_path, box=box, font_scale=FONT_SCALE, domain=domain)
            for g_id, record in grounding_records.items():
                records[g_id + count] = record
                records[g_id + count]["domain"] = domain
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


def batch_generate_grounding_ablation(world_paths, save_path="./", verbose=False, perspective="front", labeled=False, proc_id=0,
                          box=False, domain="sim"):
    frame_paths = [os.path.dirname(world_path) for world_path in world_paths]
    records = {}
    current_type = "grounding"
    current_frame = ""
    try:
        count = 0
        for frame_path in tqdm.tqdm(frame_paths, desc=f"Processing {proc_id}", unit="frame"):
            current_frame = frame_path
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
                if verbose:
                    print(f"Already have labelled version FOR {frame_path}")
                static_id2l = json.load(open(static_id2label_path, "r"))
                assert static_id2l is not None
            grounding_records = grounding_ablations(
                frame_path=frame_path, perspective=perspective, verbose=verbose,
                id2label_path=static_id2label_path, configs=SETTINGS)
            for g_id, record in grounding_records.items():
                records[g_id + count] = record
                records[g_id + count]["group"] = count
                records[g_id + count]["variant"] = g_id
                records[g_id + count]["domain"] = domain
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


def batch_generate_general_ablation(world_paths, save_path="./", verbose=False, perspective="front", labeled=False, proc_id=0,
                          configs=None, domain="sim"):
    frame_paths = [os.path.dirname(world_path) for world_path in world_paths]
    static_templates = TEMPLATES["static"]
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
            if not labeled or not os.path.exists(static_labeled_path):
                if verbose:
                    print(f"Creating single-frame-consistent labelling for {frame_path}")
                result = static_id2label(frame_path, perspective)
                static_id2l = result  #json.load(open(static_id2label_path, "r"))
            else:
                if verbose:
                    print(f"Already have labelled version FOR {frame_path}")
                static_id2l = json.load(open(static_id2label_path, "r"))
            assert static_id2l is not None
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
                            obs=None, options=option2answer
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
                                obs=None, options=option2answer
                            )
                            frame_id += 1
                        for id in ids_of_interest:
                            queried_ids.add(id)
            new_id2label = {object_id: i for i, object_id in enumerate(queried_ids)}
            original_id2label = static_id2l
            assert new_id2label is not None

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



            for config_idx, config in enumerate(configs):
                print(config_idx)
                font_scale, background_color, form = config["font_scale"], config["background_color"], config["form"]
                static_labeled_path = os.path.join(frame_path, f"{config_idx}_static_labeled_{perspective}_{identifier}.png")
                labelframe(frame_path=frame_path, perspective=perspective, save_path=static_labeled_path,
                           id2l=static_id2l,
                           font_scale=font_scale, bounding_box=form == "box", masking=form == "mask",
                           background_color=background_color)
                new_labeled_path = os.path.join(frame_path, f"{config_idx}_static_qa_labeled_{perspective}_{identifier}.png")
                labelframe(
                    frame_path=frame_path, perspective="front", save_path=new_labeled_path,
                    query_ids=list(queried_ids), id2l=new_id2label, font_scale=font_scale, bounding_box=form == "box", masking=form == "mask",
                           background_color=background_color
                )
                for qid, record in frame_records.items():
                    copied = copy.deepcopy(record)
                    if copied["type"] not in ["identify_closest", "identify_leftmost", "identify_rightmost",
                                              "identify_frontmost", "identify_backmost", "describe_scenario",
                                              "describe_sector", "describe_distance"]:
                        copied["obs"] = [new_labeled_path]
                    else:
                        copied["obs"] = [static_labeled_path]
                    copied["variant"] = config_idx
                    copied["domain"] = domain
                    records[qid + count] = copied
                count += len(frame_records)

            grounding_records = grounding_ablations(frame_path=frame_path, perspective=perspective, verbose=verbose,
                                                   id2label_path=static_id2label_path, configs=configs)
            for g_id, record in grounding_records.items():
                records[g_id + count] = record
                records[g_id + count]["variant"] = g_id
                records[g_id + count]["domain"] = domain
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


def multiprocess_generate_static(session_path, save_path="./", verbose=False, perspective="front", labeled=False,
                                 num_proc=1, box=False, domain="sim"):
    def find_worlds(session_path):
        print("Searching")
        pattern = os.path.join(session_path, "**", "**", "world**.json")
        matching_frames = glob.glob(pattern)
        return matching_frames
    world_paths = find_worlds(session_path)
    print(f"Working on {len(world_paths)} frames.")
    job_chunks = split_list(world_paths, num_proc)
    print(f"{len(world_paths)} frames distributed across {num_proc} processes, {math.ceil(len(world_paths)/num_proc)} MAX each process")
    processes = []
    name_root = os.path.basename(save_path)
    save_dir = os.path.dirname(save_path)
    for proc_id in range(num_proc):
        print(f"Sending job {proc_id}")
        name = f"{proc_id}_{name_root}"
        p = multp.Process(
            target= batch_generate_static,
            args=(
                job_chunks[proc_id],
                os.path.join(save_dir, name),
                verbose,
                perspective,
                labeled, proc_id, USEBOX, domain
            ),
        )
        print(f"Successfully sent {proc_id}")
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="/bigdata/weizhen/metavqa_cvpr/scenarios/nusc_sim", help="The path to stored episodes.")
    parser.add_argument("--save_path", default="/bigdata/weizhen/metavqa_cvpr/vqas/scratch/qa.json", help="Name template for qas. Each process will have its proc_id as prefix to the basename.")
    parser.add_argument("--num_proc", default=1, type=int, help="Number of process generating qas.")
    parser.add_argument("--use_existing_labels", action="store_true", help="If set, will generate new frame-level labelling regardless of if such labelling already exists.")
    parser.add_argument("--nusc_real", action="store_true", help="If set, will load additional corner files for generating som for nuscenes data with real observations. Also will skip color-identification problems.")
    parser.add_argument("--verbose", action="store_true", help="Set verbosity for debugging")
    parser.add_argument("--domain", default="sim", help="Specify the observation domains. Either \"sim\" or \"real\" ")
    args = parser.parse_args()
    print("Running with the following parameters")
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))
    if args.nusc_real:
        TEMPLATES["static"] = {
            key: val for key, val in TEMPLATES["static"].items() if key not in ["identify_color"]
        }
    multiprocess_generate_static(
        session_path=args.scenarios,
        save_path=args.save_path,
        verbose=args.verbose,
        num_proc=args.num_proc,
        labeled=args.use_existing_labels,
        box=USEBOX,
        domain=args.domain
    )
