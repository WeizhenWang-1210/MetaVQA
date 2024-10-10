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

current_directory = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = json.load(open(os.path.join(current_directory, "questions_templates.json"), "r"))

DIRECTION_MAPPING = {
    "l": "to the left of",
    "r": "to the right of",
    "f": "directly in front of",
    "b": "directly behind",
    "lf": "to the left and in front of",
    "rf": "to the right and in front of",
    "lb": "to the left and behind",
    "rb": "to the right and behind"
}


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
            result=[transform(o) for o in result]
        elif isinstance(transform,dict):
            result = [transform[o] for o in result]
    return result


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


def generate(frame_path, question_type, perspective="front", verbose=True):
    identifier = os.path.basename(frame_path)
    world_path = os.path.join(frame_path, "world_{}.json".format(identifier))
    world = json.load(open(world_path, "r"))
    label2id, invalid_ids = enumerate_frame_labels(frame_path, perspective)
    labels = list(label2id.keys())
    ego_id, nodelist = nodify(world, multiview=False)
    graph = SceneGraph(ego_id, nodelist, frame_path)
    ego_node = graph.get_ego_node()
    if question_type == "identify_color":
        # randomly choose a non-ego labelled object
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        # Fill the question template's <id...> with the chosen label.
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
        # Getting the answer from SceneGraph. In addition, grab extra information from the scene graph for explanation string.
        object = get(world, label2id[selected_label])
        color, type = object["color"], object["type"]
        # First, get a pool of sensible answers for the multiple-choice setup from the scenario. If no sufficient number
        # of options present, we grab from the answer space
        multiple_choice_options = create_options(graph.statistics["<p>"], 4, color, NAMESPACE["color"])
        multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
        answer = answer2label[color]
        # Postpend the multiple-choice string.
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])
        pos = ego_node.compute_relation_string(node=graph.get_node(object["id"]), ref_heading=ego_node.heading)
        explanation = f"The color of this {type.lower()} (<{selected_label}>) {DIRECTION_MAPPING[pos]} us is {color.lower()}."#.format(, , , )
        if verbose:
            print(question_type, question)
            print(answer, explanation)
    #todo maybe size?
    elif question_type == "identify_type":
        type_space = [NAMED_MAPPING[obj]["singular"] for obj in NAMED_MAPPING.keys()]
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
        object = get(world, label2id[selected_label])
        color, type = object["color"], object["type"]
        multiple_choice_options = create_options(graph.statistics["<t>"], 4, type, type_space)
        multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
        for key, value in NAMED_MAPPING.items():
            multiple_choice_string = multiple_choice_string.replace(key, value["singular"].capitalize())
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])
        answer = answer2label[type]
        pos = ego_node.compute_relation_string(node=graph.get_node(object["id"]), ref_heading=ego_node.heading)
        explanation = "The type of this {} object(<{}>) {} of us is {}.".format(color.lower(), selected_label,
                                                                          DIRECTION_MAPPING[pos],
                                                                          NAMED_MAPPING[type]["singular"])
        print(question)
        print(answer, explanation)
    elif question_type == "identify_distance":
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
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
        pos = ego_node.compute_relation_string(node=graph.get_node(object.id), ref_heading=ego_node.heading)
        explanation = "The {} {}(<{}>) is {} meters {} us.".format(color.lower(), NAMED_MAPPING[type]["singular"],
                                                                  selected_label, round(distance),
                                                                  DIRECTION_MAPPING[pos])
        print(question)
        print(answer, explanation)
    elif question_type == "identify_position":
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
        object = graph.get_node(label2id[selected_label])
        color, type = object.color, object.type
        pos = ego_node.compute_relation_string(node=graph.get_node(object.id), ref_heading=ego_node.heading)
        multiple_choice_options = create_options(list(POSITION2CHOICE.keys()), 4, pos, list(POSITION2CHOICE.keys()),
                                                 POSITION2CHOICE)
        multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])
        answer = answer2label[POSITION2CHOICE[pos]]
        explanation = "The {} {}(<{}>) is {} us.".format(color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos])
        print(question)
        print(answer, explanation)
    elif question_type == "identify_heading":
        #TODO select cars only
        labels= [label for label in labels if graph.get_node(label2id[label]).type not in ["Cone", "Barrier", "Warning", "TrafficLight"]]
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        selected_label = 24
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
        object = graph.get_node(label2id[selected_label])
        heading = identify_heading(ego_node.pos, ego_node.heading)([[object]])
        heading = heading[object.id]

        options = [((heading - 1 + i * 3) % 12) + 1 for i in range(4)]
        transform = lambda x: "{} o'clock".format(x)

        multiple_choice_options = create_options(options, 4, heading, list(range(1,13)), transform)
        multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)



        color, type = object.color, object.type
        pos = ego_node.compute_relation_string(node=graph.get_node(object.id),
                                               ref_heading=ego_node.heading)

        answer = answer2label[transform(heading)]
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])

        explanation = "The {} {}(<{}>) {} us is facing the {} o'clock direction.".format(color.lower(), NAMED_MAPPING[type]["singular"], selected_label,
                                                                          DIRECTION_MAPPING[pos], heading)
        print(question)
        print(answer, explanation)
    elif question_type == "pick_closer":
        selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        while -1 in selected_labels:
            selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        id1, id2 = selected_labels

        options = [f"<{id1}> and <{id2}> are about the same distance", f"<{id1}> is closer", f"<{id2}> is closer"]


        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(id1), "<id2>": str(id2)})
        object1, object2 = graph.get_node(label2id[id1]), graph.get_node(label2id[id2])
        distance1, distance2 = get_distance(object1.pos, ego_node.pos), get_distance(object2.pos, ego_node.pos)
        color1, type1 = object1.color, object1.type
        pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),ref_heading=ego_node.heading)
        color2, type2 = object2.color, object2.type
        pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id), ref_heading=ego_node.heading)
        if abs(distance1 - distance2) <= 2:
            index = 0

            explanation = ("Object <{}>, a {} {} located to our {}, is about the same distance from us as object <{"
                             "}>, a {} {} located to our {}.").format(
                id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], id2, color2.lower(), NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2]
            )
        else:
            if distance1 < distance2:
                index = 1
                explanation = ("Object <{}>, a {} {} {} us , is closer to us than object <{}>, "
                                 "a {} {} {} us.").format(
                    id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], id2, color2.lower(), NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2]
                )
            else:
                index = 2
                explanation = ("Object <{}>, a {} {} {} us, is closer to us than object <{}>, "
                                 "a {} {} {} us.").format(
                    id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], id2, color2.lower(), NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2]
                )

        multiple_choice_options = create_options(options, 3, options[index], options)
        multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
        answer = answer2label[options[index]]
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])
        print(question)
        print(answer, explanation)
    elif question_type == "predict_crash_ego_still":
        #TODO examine the trajectories. via visualization
        labels = [label for label in labels if
                  graph.get_node(label2id[label]).type not in ["Cone", "Barrier", "Warning", "TrafficLight"]]
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        object = graph.get_node(label2id[selected_label])
        init_center = object.pos
        selected_label = 1
        extrapolated_centers = [list(np.array(object.heading) * i + np.array(init_center)) for i in range(50)]
        extrapolated_boxes = extrapolate_bounding_boxes(extrapolated_centers,
                                                        np.arctan2(object.heading[1], object.heading[0]), object.bbox)
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
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])
        answer = answer2label["Yes" if crash else "No"]
        if crash:
            explanation = "Yes, this {} {}(<{}>) {} us and heading at {} o\' clock will run into us if we both drive along our current heading.".format(
                color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)
        else:
            explanation = "No, this {} {}(<{}>) {} us and heading at {} o\' clock will not run into us if we both drive along our current heading.".format(
                color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)

        print(question)
        print(answer, explanation)
    elif question_type == "predict_crash_ego_dynamic":
        #TODO examine the trajectories. via visualization
        #TODO make sure the selectd objects ar movable
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        selected_label = 1
        object = graph.get_node(label2id[selected_label])
        init_center = object.pos
        extrapolated_centers = [list(np.array(object.heading) * i + np.array(init_center)) for i in range(50)]
        extrapolated_boxes = extrapolate_bounding_boxes(extrapolated_centers,
                                                        np.arctan2(object.heading[1], object.heading[0]), object.bbox)
        ego_centers = [list(np.array(ego_node.heading) * i + np.array(ego_node.pos)) for i in range(50)]
        ego_boxes = extrapolate_bounding_boxes(ego_centers,
                                               np.arctan2(ego_node.heading[1], ego_node.heading[0]), ego_node.bbox)

        crash = box_trajectories_overlap(extrapolated_boxes, ego_boxes)
        intersect = box_trajectories_intersect(extrapolated_boxes, ego_boxes)
        options = ["Yes", "No"]

        multiple_choice_options = create_options(options, 2, "Yes" if crash else "No", options)
        multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])
        answer = answer2label["Yes" if crash else "No"]

        object = graph.get_node(label2id[selected_label])
        color, type = object.color, object.type
        heading = identify_heading(ego_node.pos, ego_node.heading)([[object]])
        heading = heading[object.id]
        pos = ego_node.compute_relation_string(node=graph.get_node(object.id),
                                               ref_heading=ego_node.heading)
        if crash:
            explanation = "Yes, this {} {}<{}> {} us and heading at {} o\' clock will run into us if it proceed along its current heading.".format(
                color.lower(), NAMED_MAPPING[type]["singular"], selected_label, DIRECTION_MAPPING[pos], heading)
        else:
            if intersect:
                explanation = "No, this {} {}<{}> {} us and heading at {} o\' clock will not run into us even though our paths will intersect.".format(
                    color.lower(), NAMED_MAPPING[type]["singular"], selected_label,DIRECTION_MAPPING[pos], heading)
            else:
                explanation = "No, this {} {}<{}> {} us and heading at {} o\' clock will not run into us.".format(
                    color.lower(), NAMED_MAPPING[type]["singular"], selected_label,DIRECTION_MAPPING[pos], heading)

        print(question)
        print(answer, explanation)
    elif question_type == "relative_distance":
        selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        while -1 in selected_labels:
            selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        id1, id2 = selected_labels
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(id1), "<id2>": str(id2)})
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
        pos2to1 = object2.compute_relation_string(node=object1, ref_heading=ego_node.heading)
        color1, type1 = object1.color, object1.type
        pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),
                                                ref_heading=ego_node.heading)
        color2, type2 = object2.color, object2.type
        pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id),
                                                ref_heading=ego_node.heading)
        explanation = "Object <{}>, a {} {} {} us, is {} object <{}>, a {} {} {} us, at a {} distance.".format(
            id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], DIRECTION_MAPPING[pos2to1], id2, color2.lower(), NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2], relative_dist)
        print(question)
        print(answer, explanation)
    elif question_type == "relative_position":
        selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        while -1 in selected_labels:
            selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        id1, id2 = selected_labels
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(id1), "<id2>": str(id2)})
        object1, object2 = graph.get_node(label2id[id1]), graph.get_node(label2id[id2])
        #distance1, distance2 = get_distance(object1.pos, ego_node.pos), get_distance(object2.pos, ego_node.pos)
        pos2to1 = object2.compute_relation_string(node=object1, ref_heading=ego_node.heading)
        color1, type1 = object1.color, object1.type
        pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),
                                                ref_heading=ego_node.heading)
        color2, type2 = object2.color, object2.type
        pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id),
                                                ref_heading=ego_node.heading)

        multiple_choice_options = create_options(list(POSITION2CHOICE.keys()), 4, pos2to1, list(POSITION2CHOICE.keys()),
                                                 POSITION2CHOICE)
        multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])
        answer = answer2label[POSITION2CHOICE[pos2to1]]







        explanation = "Object <{}>, a {} {} {} us, is {} object <{}>, a {} {} {} us.".format(
            id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], DIRECTION_MAPPING[pos2to1], id2, color2.lower(), NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2])
        print(question)
        print(answer, explanation)
    elif question_type == "relative_heading":
        labels = [label for label in labels if
                  graph.get_node(label2id[label]).type not in ["Cone", "Barrier", "Warning", "TrafficLight"]]
        #TODO select cars only
        selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        while -1 in selected_labels:
            selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        id1, id2 = 1, 0
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(id1), "<id2>": str(id2)})
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
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])


        if abs(angle1to2) < 20:
            explanation = "Yes. Object <{}>, a {} {} {} us, is heading toward roughly the same direction as object <{}>, a {} {} {} us.".format(
                id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], id2, color2.lower(), NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2])
        else:
            explanation = "No. Object <{}>, a {} {} {} us, is not heading toward the same direction as object <{}>, a {} {} located {} us. In particular, object <{}>'s heading differs by {} degrees counterclockwise from that of object <{}>.".format(
                id1, color1.lower(), NAMED_MAPPING[type1]["singular"], DIRECTION_MAPPING[pos1], id2, color2.lower(), NAMED_MAPPING[type2]["singular"], DIRECTION_MAPPING[pos2], id2, round(angle1to2), id1)
        print(question)
        print(answer, explanation)
    elif question_type == "relative_predict_crash_still":
        #TODO examine the trajectories. via visualization
        label1_pool = [label for label in labels if
                  graph.get_node(label2id[label]).type not in ["Cone", "Barrier", "Warning", "TrafficLight"] and label!=-1]
        id1 = random.choice(label1_pool)
        id2 = random.choice(labels)
        while id2 == -1 or id2 == id1:
            id2 = random.choice(labels)
        object1, object2 = graph.get_node(label2id[id1]), graph.get_node(label2id[id2])
        init_center = object1.pos
        extrapolated_centers = [list(np.array(object1.heading) * i + np.array(init_center)) for i in range(50)]
        extrapolated_boxes = extrapolate_bounding_boxes(extrapolated_centers,
                                                        np.arctan2(object1.heading[1], object1.heading[0]),
                                                        object1.bbox)
        object2_boxes = [object2.bbox for i in range(50)]
        crash = box_trajectories_overlap(extrapolated_boxes, object2_boxes)
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(id1), "<id2>": str(id2)})
        color1, type1 = object1.color, object1.type
        pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),
                                                ref_heading=ego_node.heading)
        color2, type2 = object2.color, object2.type
        pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id),
                                                ref_heading=ego_node.heading)

        options = ["Yes", "No"]
        multiple_choice_options = create_options(options, 2, "Yes" if crash else "No", options)
        multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])
        answer = answer2label["Yes" if crash else "No"]


        if crash:
            explanation = "Yes, object <{}> will run into object <{}>.".format(
                id1, id2)
        else:
            explanation = "No, object <{}> will not run into object <{}>.".format(
                id1, id2)
        print(question)
        print(answer, explanation)

    elif question_type == "relative_predict_crash_dynamic":
        #TODO examine the trajectories. via visualization
        labels = [label for label in labels if
                  graph.get_node(label2id[label]).type not in ["Cone", "Barrier", "Warning", "TrafficLight"] and label!=-1]
        selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        while -1 in selected_labels:
            selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
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
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(id1), "<id2>": str(id2)})
        color1, type1 = object1.color, object1.type
        pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),
                                                ref_heading=ego_node.heading)
        color2, type2 = object2.color, object2.type
        pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id),
                                                ref_heading=ego_node.heading)
        options = ["Yes", "No"]
        multiple_choice_options = create_options(options, 2, "Yes" if crash else "No", options)
        multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
        question = " ".join([question, "Choose the best answer from: {}".format(multiple_choice_string)])
        answer = answer2label["Yes" if crash else "No"]
        if crash:
            explanation = "Yes, object <{}> will run into object <{}>.".format(
                id1, id2)
        else:
            explanation = "No, object <{}> will not run into object <{}>.".format(
                id1, id2)
        print(question)
        print(answer, explanation)
    else:
        print('Something wrong!')
    return question, answer, explanation


if __name__ == "__main__":
    frame_path = "/bigdata/weizhen/repo/qa_platform/public/test_wide/scene-0061_91_100/0_91"
    asset_path = "/test_wide/scene-0061_91_100/0_91"
    obs = os.path.join(asset_path, "labeled_front_{}.png".format(os.path.basename(frame_path)))
    generate(frame_path, "relative_predict_crash_dynamic", "front")
    results = {}
    idx = 0
    for question_type in TEMPLATES["static"].keys():
        question, answer, explanation = generate(frame_path, question_type, "front")
        results[idx] = dict(
            question=question, answer=answer, obs=obs, type=question_type, explanation = explanation
        )
        idx += 1
    json.dump(results, open("/bigdata/weizhen/repo/qa_platform/public/data.json", "w"), indent=2)
