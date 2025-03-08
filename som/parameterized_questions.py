import itertools
import json
import os
import random
from collections import defaultdict
from copy import deepcopy
from som.qa_utils import create_options, create_multiple_choice
from som.utils import enumerate_frame_labels, fill_in_label
from vqa.configs.NAMESPACE import POSITION2CHOICE
from vqa.dataset_utils import get_distance, transform_to_world
from vqa.object_node import nodify, transform
from vqa.scene_graph import SceneGraph
from som.embodied_utils import ACTION, classify_speed, get_end_sector, determine_collisions
current_directory = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = json.load(open(os.path.join(current_directory, "questions_templates.json"), "r"))






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
    elif question_type == "embodied_distance":
        criteria = {
            "slow" : "(0-10 mph)",
            "moderate": "(10-30 mph)",
            "fast": "(30-50 mph)",
            "very fast": "(50+ mph)",
        }
        speed, action, duration = param["<speed>"], param["<action>"], param["<duration>"]
        speed_class = classify_speed(speed)
        action_class = ACTION.get_action(action)
        end_distance, _ = get_end_sector(action=action, speed=speed, duration=duration)
        available_options = ["very close", "close", "medium", "far"]
        labels = ["A","B","C","D"]
        question = fill_in_label(
                    template_str=TEMPLATES["static"][question_type]["text"][0],
                    replacement={
                        "<speed>": f"{speed_class}{criteria[speed_class]}",
                        "<action>": action_class,
                        "<duration>": f"{str(round(duration/10,1))} seconds"
                    }
                )
        explanation = ""
        answer = labels[available_options.index(end_distance)]
        option2answer = {
            label: available_option for label, available_option in zip(labels, available_options)
        }
    elif question_type == "embodied_sideness":
        criteria = {
            "slow": "(0-10 mph)",
            "moderate": "(10-30 mph)",
            "fast": "(30-50 mph)",
            "very fast": "(50+ mph)",
        }
        speed, action, duration = param["<speed>"], param["<action>"], param["<duration>"]
        speed_class = classify_speed(speed)
        action_class = ACTION.get_action(action)
        _,end_side = get_end_sector(action=action, speed=speed, duration=duration)
        available_options = ["left-front", "front", "right-front"]
        labels = ["A","B","C"]
        question = fill_in_label(
                    template_str=TEMPLATES["static"][question_type]["text"][0],
                    replacement={
                        "<speed>": f"{speed_class}{criteria[speed_class]}",
                        "<action>": action_class,
                        "<duration>": f"{str(round(duration/10,1))} seconds"
                    }
                )
        explanation = ""
        if end_side == "m":
            answer = "B"
        else:
            answer = labels[available_options.index(POSITION2CHOICE[end_side])]
        option2answer = {
            label: available_option for label, available_option in zip(labels, available_options)
        }
    elif question_type == "embodied_collision":
        criteria = {
            "slow": "(0-10 mph)",
            "moderate": "(10-30 mph)",
            "fast": "(30-50 mph)",
            "very fast": "(50+ mph)",
        }
        non_ego_labels = [label for label in labels if label != -1]
        if len(non_ego_labels) <= 0 :
            print(f"Not enough things to generate {question_type}")
        else:
            chosen_label = random.choice(non_ego_labels)
            speed, action, duration = param["<speed>"], param["<action>"], param["<duration>"]
            object = graph.get_node(label2id[chosen_label])
            object_box = object.bbox
            object_box_ego = transform(ego_node, object_box)
            will_collide, collision_time = determine_collisions(obj_box=object_box_ego, action=action, speed=speed, duration=duration)
            speed_class = classify_speed(speed)
            action_class = ACTION.get_action(action)
            question = fill_in_label(
                        template_str=TEMPLATES["static"][question_type]["text"][0],
                        replacement={
                            "<speed>": f"{speed_class}{criteria[speed_class]}",
                            "<action>": action_class,
                            "<duration>": f"{str(round(duration/10,1))} seconds",
                            "<id1>": str(chosen_label)
                        }
                    )
            obj_pos = ego_node.compute_relation_string(node=graph.get_node(object.id), ref_heading=ego_node.heading)
            _, end_pos = get_end_sector(action=action, speed=speed, duration=duration)
            if end_pos == "m":
                end_pos = "f"
            if will_collide:
                explanation = f"We will run into object <{chosen_label}> currently in {POSITION2CHOICE[obj_pos]} sector after {round(collision_time,1)} seconds."
            else:
                if obj_pos == end_pos:
                    explanation = f"We will not run into object <{chosen_label}>, even though we both end in our {POSITION2CHOICE[obj_pos]} sector."
                else:
                    explanation = f"We will not run into object <{chosen_label}>. Object <{chosen_label}> is located in the {POSITION2CHOICE[obj_pos]} sector, but we will end in the {POSITION2CHOICE[end_pos]} sector."
            labels = ["A", "B"]
            available_options = ["Yes", "No"]
            answer = labels[available_options.index("Yes" if will_collide else "No")]
            option2answer = {
                label: available_option for label, available_option in zip(labels, available_options)
            }
            ids_of_interest = [label2id[chosen_label]]
    else:
        print("Not yet implemented")
        exit()
    if verbose:
        print(question)
        print(answer)
        print(explanation)
    return question, answer, explanation, ids_of_interest, option2answer
