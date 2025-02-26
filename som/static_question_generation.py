from som.utils import enumerate_frame_labels, get, fill_in_label
import os
import json
import random
from vqa.object_node import nodify, extrapolate_bounding_boxes
from vqa.scene_graph import SceneGraph
from vqa.dataset_utils import get_distance
from vqa.functionals import identify_heading
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = json.load(open(os.path.join(current_directory, "questions_templates.json"), "r"))


def generate(frame_path, question_type, perspective="front"):
    identifier = os.path.basename(frame_path)
    world_path = os.path.join(frame_path, "world_{}.json".format(identifier))
    world = json.load(open(world_path, "r"))
    label2id, invalid_ids = enumerate_frame_labels(frame_path, perspective)
    labels = list(label2id.keys())

    ego_id, nodelist = nodify(world, multiview=False)
    graph = SceneGraph(ego_id, nodelist, frame_path)




    ego_node = graph.get_ego_node()
    if question_type == "identify_color":
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
        object = get(world, label2id[selected_label])
        color, type = object["color"], object["type"]
        pos = ego_node.compute_relation_string(node=graph.get_node(object["id"]),
                                               ref_heading=ego_node.heading)
        answer_string = "The color of this {} located at our {} is {}.".format(type.lower(), pos, color.lower())
        print(question, answer_string)
    elif question_type == "identify_type":
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
        object = get(world, label2id[selected_label])
        color, type = object["color"], object["type"]
        pos = ego_node.compute_relation_string(node=graph.get_node(object["id"]),
                                               ref_heading=ego_node.heading)
        answer_string = "The type of this {} thing located at our {} is {}.".format(color.lower(), pos, type.lower())
        print(question, answer_string)
    elif question_type == "identify_distance":
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
        object = graph.get_node(label2id[selected_label])
        distance = get_distance(object.pos, ego_node.pos)
        color, type = object.color, object.type
        if 0 < distance <= 2:
            relative_dist = "very close"
        elif 2 < distance <= 10:
            relative_dist = "close"
        elif 10 < distance <= 30:
            relative_dist = "medium"
        else:
            relative_dist = "far"
        pos = ego_node.compute_relation_string(node=graph.get_node(object.id),
                                               ref_heading=ego_node.heading)
        answer_string = "The {} {} located at {} is at {} distance from me {}.".format(color.lower(), type.lower(), pos,
                                                                                       relative_dist, distance)
        print(question, answer_string)
    elif question_type == "identify_position":
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        selected_label = 7
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
        object = graph.get_node(label2id[selected_label])
        color, type = object.color, object.type
        pos = ego_node.compute_relation_string(node=graph.get_node(object.id),
                                               ref_heading=ego_node.heading)
        answer_string = "The {} {} is located at {} from me.".format(color.lower(), type.lower(), pos)
        print(question, answer_string)
    elif question_type == "identify_heading":
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        selected_label = 24
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(selected_label)})
        object = graph.get_node(label2id[selected_label])
        heading = identify_heading(ego_node.pos, ego_node.heading)([[object]])
        heading = heading[object.id]
        color, type = object.color, object.type
        pos = ego_node.compute_relation_string(node=graph.get_node(object.id),
                                               ref_heading=ego_node.heading)
        answer_string = "The {} {} to our {} is oriented toward {} o'clock direction.".format(color.lower(),
                                                                                              type.lower(), pos,
                                                                                              heading)
        print(question, answer_string)

    elif question_type == "pick_closer":
        selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        while -1 in selected_labels:
            selected_labels = np.random.choice(np.array(labels), size=2, replace=False)
        id1, id2 = selected_labels
        question = fill_in_label(TEMPLATES["static"][question_type]["text"][0], {"<id1>": str(id1), "<id2>": str(id2)})
        object1, object2 = graph.get_node(label2id[id1]), graph.get_node(label2id[id2])
        distance1, distance2 = get_distance(object1.pos, ego_node.pos), get_distance(object2.pos, ego_node.pos)
        color1, type1 = object1.color, object1.type
        pos1 = ego_node.compute_relation_string(node=graph.get_node(object1.id),
                                                ref_heading=ego_node.heading)
        color2, type2 = object2.color, object2.type
        pos2 = ego_node.compute_relation_string(node=graph.get_node(object2.id),
                                                ref_heading=ego_node.heading)
        if abs(distance1 - distance2) <= 2:
            answer_string = ("Object <{}>, a {} {} located to our {}, is about the same distance from us as object <{"
                             "}>, a {} {} located to our {}.").format(
                id1, color1.lower(), type1.lower(), pos1, id2, color2.lower(), type2.lower(), pos2
            )
        else:
            if distance1 < distance2:
                answer_string = ("Object <{}>, a {} {} located to our {}, is closer to us than object <{}>, "
                                 "a {} {} located to our {}.").format(
                    id1, color1.lower(), type1.lower(), pos1, id2, color2.lower(), type2.lower(), pos2
                )
            else:
                answer_string = ("Object <{}>, a {} {} located to our {}, is closer to us than object <{}>, "
                                 "a {} {} located to our {}.").format(
                    id2, color2.lower(), type2.lower(), pos2, id1, color1.lower(), type1.lower(), pos1
                )
        print(question, answer_string)

    elif question_type == "predict_crash":
        selected_label = random.choice(labels)
        while selected_label == -1:
            selected_label = random.choice(labels)
        object = graph.get_node(label2id[selected_label])
        init_center = object.pos

        extrapolated_centers = [np.array(object.heading)*i + np.array(init_center) for i in range(50)]
        extrapolated_boxes = extrapolate_bounding_boxes(extrapolated_centers, object.heading, object.bbox)
        #print(extrapolated_boxes)





if __name__ == "__main__":
    frame_path = "E:/Bolei/scene-0061_76_85/0_76"#"/bigdata/weizhen/metavqa_iclr/scenarios/test_wide/scene-0061_91_100/0_91"
    generate(frame_path, "predict_crash", "front")
