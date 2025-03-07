# find_all_episodes
# for each episode:
#   for each question type:
#       try for x times, get at most N valid questions
#       sample 4 out of N valid questions
#
import json
import os
import re
import glob
from vqa.scene_graph import TemporalGraph
from relic.dynamic_question_generator import DynamicQuerySpecifier
from vqa.static_question_generator import NAMED_MAPPING
from vqa.object_node import transform, transform_vec
import random
from collections import defaultdict
import math


def find_episodes(session_folder):
    """
    Return subfolders representing an episode under a session folder.
    """
    pattern = re.compile(r'\d+_\d+_\d+')
    folders = [f for f in os.listdir(session_folder) if os.path.isdir(os.path.join(session_folder, f))]
    matched_folders = [os.path.join(session_folder, folder) for folder in folders if pattern.match(folder)]
    return matched_folders

def load_valid_episodes(session_folder):
    import json
    statistics = json.load(open(os.path.join(session_folder, "session_statistics.json")))
    valid_episodes = statistics["valid_episodes"]
    return [os.path.join(session_folder, valid_episode) for valid_episode in valid_episodes]


def extract_frames(episode, debug=False):
    sep = os.sep
    annotation_path_template = f"{episode}{sep}**{sep}world*.json"
    #annotation_path_template = f"{episode}/**/world*.json"
    files = glob.glob(annotation_path_template, recursive=True)
    def extract_numbers(filename):
        identifier = filename.split(sep)[-2]
        x, y = identifier.split('_')
        return int(x), int(y)
    sorted_files = sorted(files, key=extract_numbers)
    return sorted_files


def select_key_frames(root_dir, frame_per_episode=3):
    key_frames = []
    for content in os.listdir(root_dir):
        path = os.path.join(root_dir, content)  # episode_folder
        if os.path.isdir(path):
            frame_files = extract_frames(path)
            if len(frame_files) <= 1:
                key_frames += frame_files
            else:
                freq = math.floor(len(frame_files) / frame_per_episode)
                selected = frame_files[::freq]
                key_frames += selected
    return key_frames



def find_nuscene_frames(root_dir):
    result = []
    root_dir_content = os.listdir(root_dir)
    episodes_folder = [content for content in root_dir_content if os.path.isdir(os.path.join(root_dir, content))]
    for content in episodes_folder:
        path = os.path.join(root_dir, content)
        path_template = os.path.join(path, "**/real*.png")
        frame_files = glob.glob(path_template, recursive=True)
        nuscenes_folders = list(set([os.path.dirname(frame_file) for frame_file in frame_files]))
        time_stamps = [os.path.basename(nuscenes_folder)  for nuscenes_folder in nuscenes_folders]
        nuscenes_annotation = [os.path.join(nuscenes_folder, f"world_{time_stamp}.json") for nuscenes_folder, time_stamp in zip(nuscenes_folders,time_stamps)]
        result += nuscenes_annotation
    return result


def extract_observations(episode, debug=False):
    def extract_numbers(filename):
        sep = os.sep
        identifier = filename.split(sep)[-2]
        x, y = identifier.split('_')
        return int(x), int(y)
    observations = {}
    perspectives = ["front", "leftb", "leftf", "rightb", "rightf", "back"]
    for perspective in perspectives:
        rgb_path_template = f"{episode}/**/rgb_{perspective}**.png"
        sorted_rgb = sorted(glob.glob(rgb_path_template, recursive=True), key=extract_numbers)
        observations[perspective] = sorted_rgb
    lidar_template = f"{episode}/**/lidar_**.pkl"
    sorted_lidar = sorted(glob.glob(lidar_template, recursive=True), key=extract_numbers)
    observations["lidar"] = sorted_lidar
    if debug:
        length = len(observations["lidar"])
        for value in observations.values():
            assert len(value) == length
    return observations


def extract_real_observations(episode, debug=False):
    def extract_numbers(filename):
        sep = os.sep
        identifier = filename.split(sep)[-2]
        x, y = identifier.split('_')
        return int(x), int(y)
    observations = {}
    perspectives = ["front", "leftb", "leftf", "rightb", "rightf", "back"]
    for perspective in perspectives:
        rgb_path_template = f"{episode}/**/real_{perspective}**.png"
        sorted_rgb = sorted(glob.glob(rgb_path_template, recursive=True), key=extract_numbers)
        observations[perspective] = sorted_rgb
    return observations


def generate_trimmed_grammar(graph, tense, template):
    from vqa.grammar import NO_STATE_CFG, NO_COLOR_STATIC, NO_TYPE_STATIC, STATIC_GRAMMAR, CFG_GRAMMAR, \
        NO_COLOR_CFG, NO_TYPE_CFG
    import copy
    # type, color, action, interaction
    statistics = graph.statistics
    # in the first static settng: type, color
    # in the dynamic setting: type, color, action, interaction.
    if tense == "static":
        if "no_color" in template["constraint"]:
            grammar = copy.deepcopy(NO_COLOR_STATIC)
            for lhs, rhs in statistics.items():
                if lhs == "<p>":
                    grammar["<px>"] = [[item] for item in rhs + ["nil"]]
                else:
                    grammar[lhs] = [[item] for item in rhs + ["vehicle"]]
        elif "no_type" in template["constraint"]:
            grammar = copy.deepcopy(NO_TYPE_STATIC)
            for lhs, rhs in statistics.items():
                if lhs == "<t>":
                    grammar["<tx>"] = [[item] for item in rhs + ["vehicle"]]
                else:
                    grammar[lhs] = [[item] for item in rhs + ["nil"]]
        else:
            grammar = copy.deepcopy(STATIC_GRAMMAR)
            for lhs, rhs in statistics.items():
                if lhs == "<p>":
                    grammar[lhs] = [[item] for item in rhs + ["nil"]]
                else:
                    grammar[lhs] = [[item] for item in rhs + ["vehicle"]]
    else:
        if "no_state" in template["constraint"]:
            grammar = copy.deepcopy(NO_STATE_CFG)
            for lhs, rhs in statistics.items():
                if lhs == "<t>":
                    grammar[lhs] = [[item] for item in rhs + ["vehicle"]]
                elif lhs == "<active_deed>" or lhs == "<passive_deed>":
                    grammar[lhs] = [[item] for item in rhs]
                elif lhs != "<s>":
                    grammar[lhs] = [[item] for item in rhs + ["nil"]]
        elif "no_color" in template["constraint"]:
            grammar = copy.deepcopy(NO_COLOR_CFG)
            for lhs, rhs in statistics.items():
                if lhs == "<t>":
                    grammar[lhs] = [[item] for item in rhs + ["vehicle"]]
                elif lhs == "<active_deed>" or lhs == "<passive_deed>":
                    grammar[lhs] = [[item] for item in rhs]
                elif lhs == "<p>":
                    grammar["<px>"] = [[item] for item in rhs + ["nil"]]
                else:
                    grammar[lhs] = [[item] for item in rhs + ["nil"]]
        elif "no_type" in template["constraint"]:
            grammar = copy.deepcopy(NO_TYPE_CFG)
            for lhs, rhs in statistics.items():
                if lhs == "<t>":
                    grammar["<tx>"] = [[item] for item in rhs + ["vehicle"]]
                elif lhs == "<active_deed>" or lhs == "<passive_deed>":
                    grammar[lhs] = [[item] for item in rhs]
                else:
                    grammar[lhs] = [[item] for item in rhs + ["nil"]]
        else:
            grammar = copy.deepcopy(CFG_GRAMMAR)
            for lhs, rhs in statistics.items():
                if lhs == "<t>":
                    grammar[lhs] = [[item] for item in rhs + ["vehicle"]]
                elif lhs == "<active_deed>" or lhs == "<passive_deed>":
                    grammar[lhs] = [[item] for item in rhs]
                else:
                    grammar[lhs] = [[item] for item in rhs + ["nil"]]
    # finally, remove non-terminal token that can'tbe grounded in our particular scene.
    remove_key = set()
    for lhs, rhs in grammar.items():
        if len(rhs) == 0:
            remove_key.add(lhs)
    new_grammar = copy.deepcopy(grammar)
    for token in remove_key:
        new_grammar.pop(token)
    for token in remove_key:
        for lhs, rules in new_grammar.items():
            new_rule = []
            for rhs in rules:
                if token in rhs:
                    continue
                new_rule.append(rhs)
            new_grammar[lhs] = new_rule
    return new_grammar


def not_degenerate(question_type, q, result):
    def degenerate_compare(q):
        degenerate_refer = True
        for param, info in q.parameters.items():
            if len(q.parameters[param]["answer"]) > 0:
                degenerate_refer = False
                break
        return degenerate_refer

    func_map = dict(
        localization=lambda x: len(x[0][1][1]) > 0,
        counting=lambda x: x[0][1][1][0] > 0,
        count_equal_binary=lambda x: not degenerate_compare(q),
        count_more_binary=lambda x: not degenerate_compare(q),
        color_identification=lambda x: len(x[0][1][1]) > 0,
        type_identification=lambda x: len(x[0][1][1]) > 0,
        color_identification_unique=lambda x: len(x) > 0,
        type_identification_unique=lambda x: len(x) > 0,
        identify_stationary=lambda x: len(x) > 0,
        identify_turning=lambda x: len(x) > 0,
        identify_acceleration=lambda x: len(x) > 0,
        identify_speed=lambda x: len(x) > 0,
        identify_heading=lambda x: len(x) > 0,
        identify_head_toward=lambda x: len(x) > 0,
        predict_trajectory=lambda x: len(x) > 0
    )
    return func_map[question_type](result)


def add_to_record(question_type, q, result, candidates):
    if question_type in ["color_identification_unique", "type_identification_unique"]:
        for question, answer_record in result:
            obj_id, answer = answer_record
            if question_type == "color_identification_unique":
                explanation = "The color of that {} is {}".format \
                    (NAMED_MAPPING[q.graph.get_node(obj_id).type]["singular"], answer.lower())
            else:
                explanation = "The type of that {} thing is {}.".format \
                    (q.graph.get_node(obj_id).color.lower(),
                     NAMED_MAPPING[q.graph.get_node(obj_id).type]["singular"])

            obj_node = q.graph.get_node(obj_id)
            data_point = dict(
                question=question, answer=answer, explanation=explanation,
                answer_form="str", question_type=question_type,
                type_statistics=[obj_node.type],
                pos_statistics=transform(q.graph.get_ego_node(), [obj_node.pos]),
                color_statistics=[obj_node.color], action_statistics=[obj_node.actions],
                interaction_statistics=[obj_node.get_all_interactions()], ids=[obj_id],
                key_frame=q.graph.idx_key_frame  # key_frame will be used to determine the observation.
            )
            candidates[question_type].append(data_point)

    elif question_type in ["identify_stationary", "identify_turning", "identify_acceleration", "identify_speed",
                           "identify_heading", "identify_head_toward", "predict_trajectory"]:
        for question, answer_record in result:
            obj_id, answer = answer_record
            obj_node = q.graph.get_node(obj_id)
            if question_type == "identify_stationary":
                color_string = obj_node.color
                type_string = NAMED_MAPPING[obj_node.type]["singular"]
                explanation = "The {} {} has been staying at {}(in our current reference system) for the past period.".format(
                    color_string, type_string, transform(q.graph.get_ego_node(), [obj_node.pos])
                )
            elif question_type == "identify_turning":
                if answer == True:
                    turn_string = "turned left" if "turn_left" in obj_node.actions else "turned right"
                    explanation= "The {} {} {} over the past period.".format(color_string, type_string, turn_string)
                else:
                    explanation= "The {} {} remained roughly on the same direction."
            elif question_type == "identify_acceleration":
                explanation = "The {} {}'s speed changd from {} to {}.".format(
                    color_string, type_string, obj_node.speeds[0], obj_node.speed
                )
            elif question_type == "identify_speed":
                explanation = "The current speed of the {} {} is {}.".format(
                    color_string, type_string, answer
                )
            elif question_type == "identify_heading":
                explanation = "The {} {} is currently heading toward our {} o'clock direction.".format(
                    color_string, type_string, answer
                )
            elif question_type == "identify_head_toward":
                ego_node = q.graph.get_node()
                explanation = "The {} {} is heading toward us from our {}." \
                               .format(color_string,type_string, ego_node.compute_relation_string(obj_node, ego_node.heading))
            elif question_type == "predict_trajectory":
                explanation = "The {} {} will move along this trajectory for the next second.".format(color_string, type_string)
            else:
                print("Encountered unseen question type! {}".format(question_type))
                explanation = ""


            data_point = dict(
                question=question, answer=answer, explanation=explanation,
                answer_form="str", question_type=question_type,
                type_statistics=[obj_node.type],
                pos_statistics=transform(q.graph.get_ego_node(), [obj_node.pos]),
                color_statistics=[obj_node.color], action_statistics=[obj_node.actions],
                interaction_statistics=[obj_node.get_all_interactions()], ids=[obj_id],
                key_frame=q.graph.idx_key_frame  # key_frame will be used to determine the observation.
            )
            candidates[question_type].append(data_point)

    else:
        question, answer = result[0]
        ids, answer = answer
        if question_type=="counting":
            referral_string = q.parameters["<o>"]["en"]
            if answer[0] == 0:
                explanation = "There is no {}.".format(referral_string)
            elif answer[0] == 1:
                explanation = "There is 1 {}.".format(referral_string)
            else:
                explanation = "There are {} {}.".format(answer[0], referral_string)
        elif question_type=="localization":
            referral_string = q.parameters["<o>"]["en"]
            if len(answer) == 0:
                explanation = "There is no {} at this moment.".format(referral_string)
            elif len(answer) == 1:
                explanation = "There is 1 {} located here at this moment.".format(referral_string)
            else:
                explanation = ("There are {} {} located at the aforementioned positions at this moment."
                               .format(len(answer), referral_string))
        elif question_type == "count_equal_binary" or question_type == "count_more_binary":
            #print(ids, answer)
            #exit()
            referral_string0, referral_string1 = q.parameters["<o1>"]["en"], q.parameters["<o2>"]["en"]
            count0, count1 = len(set([obj.id for obj in q.parameters["<o1>"]["answer"]])), len(
                set([obj.id for obj in q.parameters["<o2>"]["answer"]]))
            answer_string = "\"True\"" if answer else "\"False\""
            explanation = "The number of {} is {}, and there {} {} {}. Therefore, the answer is {}".format(
                referral_string0, count0, "is" if count1 <= 1 else "are", "no" if count1 == 0 else count1,
                referral_string1,
                answer_string
            )
        elif question_type == "color_identification" or question_type=="type_identification":
            referral_string = q.parameters["<o>"]["en"]
            if len(answer) > 1:
                if question_type == "color_identification":
                    distro_strings = ["{} in {}".format(count, color.lower()) for color, count in
                                      q.statistics["colors"].items()]
                    if len(distro_strings) > 2:
                        distro_string_prefix = ", ".join(distro_strings[:-1])
                        distro_string_suffix = distro_strings[-1]
                        distro_string = ", and ".join([distro_string_prefix, distro_string_suffix])
                    elif len(distro_strings) > 1:
                        distro_string = " and ".join(distro_strings)
                    else:
                        distro_string = distro_strings[0]

                    first_string = "are" if list(q.statistics["colors"].values())[0] > 1 else "is"
                    explanation = "These are the colors seen on {}. To be more specific, there {} {}." \
                        .format(referral_string, first_string, distro_string)
                else:
                    distro_strings = ["{} {}".format(count,
                                                     NAMED_MAPPING[typee]["plural"]
                                                     if count > 1 else NAMED_MAPPING[typee]["singular"])
                                      for typee, count in q.statistics["types"].items()]
                    if len(distro_strings) > 2:
                        distro_string_prefix = ", ".join(distro_strings[:-1])
                        distro_string_suffix = distro_strings[-1]
                        distro_string = ", and ".join([distro_string_prefix, distro_string_suffix])
                    elif len(distro_strings) > 1:
                        distro_string = " and ".join(distro_strings)
                    else:
                        distro_string = distro_strings[0]
                    first_string = "are" if list(q.statistics["types"].values())[0] > 1 else "is"
                    explanation = "These are the types of {}. To be more specific, there {} {}." \
                        .format(referral_string, first_string, distro_string)
            elif len(answer) > 0:
                if question_type == "color_identification":
                    explanation = "This is the only color seen on all {} {}.".format(len(ids), referral_string)
                else:
                    explanation = "This is the type of thing that all {} {} belong to.".format(len(ids),
                                                                                               referral_string)
            else:
                if question_type == "color_identification":
                    explanation = "No color can be identified for {} because they don't exist.".format(
                        referral_string)
                else:
                    explanation = "No type can be identified for {} because no such thing is observable.".format(
                        referral_string)
        data_point = dict(
            question=question, answer=answer, explanation=explanation,
            answer_form="str", question_type=question_type,
            type_statistics=q.statistics["types"], pos_statistics=q.statistics["pos"],
            color_statistics=q.statistics["colors"], action_statistics=q.statistics["actions"],
            interaction_statistics=q.statistics["interactions"], ids=ids,
            key_frame=q.graph.idx_key_frame  # key_frame will be used to determine the observation.
        )
        candidates[question_type].append(data_point)
    return candidates


def generate_context_string(graph, end=False):
    if end:
        frame_idx = len(graph.frames) - 1
    else:
        frame_idx = graph.idx_key_frame
    ego_node = graph.get_ego_node()
    start_pos, end_pos, speed = ego_node.positions[0], ego_node.positions[frame_idx], ego_node.speed
    if "accelerating" in ego_node.actions:
        dv_string = "accelerated"
    elif "decelerating" in ego_node.actions:
        dv_string = "decelerated"
    else:
        dv_string = "maintained speed"
    if "turn_left" in ego_node.actions:
        action_string = "turned left"
    elif "turn_right" in ego_node.actions:
        action_string = "turned right"
    else:
        action_string = "maintained direction"
    avg_speed = round(sum(ego_node.speeds[:frame_idx + 1]) / frame_idx + 1, 1)
    cur_speed = round(ego_node.speed, 1)
    if end:
        start_pos = transform_vec(end_pos, ego_node.headings[frame_idx], [start_pos])[0]
    else:
        start_pos = transform(ego_node, [start_pos])[0]
    start_pos = [round(start_pos[0], 1), round(start_pos[1], 1)]
    context = (
        f"For the past period, we moved from {start_pos} to where we are now with an average speed of {avg_speed}."
        f" We {dv_string} and {action_string}, and our current speed is {cur_speed}.")
    return context


def generate_dynamic_questions(episode, templates, max_per_type=5, choose=3, attempts_per_type=100, verbose=False):
    frame_files = extract_frames(episode)
    graph = TemporalGraph(frame_files)
    print(f"Generating dynamic questions for {episode}...")
    print(f"KEY FRAME at{graph.framepaths[graph.idx_key_frame]}")
    print(f"Key frame is {graph.idx_key_frame}")
    print(f"Total frame number {len(graph.frames)}")
    context_string = generate_context_string(graph)
    if verbose:
        print(f"Context: {context_string}")
    candidates, counts, valid_questions = defaultdict(list), 0, set()
    degenerate_allowance = dict(
        localization=2,
        counting=2,
        count_equal_binary=2,
        count_more_binary=2,
        color_identification=2,
        type_identification=2,
        color_identification_unique=0,
        type_identification_unique=0,
        identify_stationary=0,
        identify_turning=0,
        identify_acceleration=0,
        identify_speed=0,
        identify_heading=0,
        identify_head_toward=0,
        predict_trajectory=0
    )


    for question_type, question_template in templates.items():
        grammar = generate_trimmed_grammar(graph, "dynamic", question_template)
        countdown, generated = attempts_per_type, 0
        while countdown > 0 and generated < max_per_type:
            if verbose:
                print("Attempt {} of {} for {}".format(attempts_per_type - countdown, attempts_per_type, question_type))
            q = DynamicQuerySpecifier(
                type=question_type, template=question_template, parameters=None, graph=graph,
                grammar=grammar, debug=False, stats=True,
            )
            if q.signature in valid_questions:
                if verbose:
                    print("Skip <{}> since the equivalent question has been asked before".format(q.translate()))
                continue
            result = q.export_qa()
            if verbose:
                print(result)
            if not_degenerate(question_type, q, result) or degenerate_allowance[question_type] > 0:
                candidates = add_to_record(question_type, q, result, candidates)
                # print(candidates)
                valid_questions.add(q.signature)
                generated += len(result)
                if not not_degenerate(question_type,q,result):
                    degenerate_allowance[question_type] -= 1
            countdown -= 1
        if generated > choose:
            candidate = candidates[question_type]
            final_selected = random.sample(candidate, choose)
            candidates[question_type] = final_selected
            counts += len(final_selected)
        else:
            counts += len(candidates[question_type])
    return candidates, counts, context_string


from vqa.static_question_generation import find_supertype

import copy
def  generate_trimmed_grammar_nuscenes(graph, template):
    from vqa.grammar import NO_COLOR_NO_TYPE_NO_STATE, NO_COLOR_NO_TYPE
    statistics = graph.statistics
    if "no_state" in template["constraint"]:
        grammar = copy.deepcopy(NO_COLOR_NO_TYPE_NO_STATE)
        for lhs, rhs in statistics.items():
            if lhs == "<t>":
                names = list(set([find_supertype(stuff) for stuff in rhs]))
                grammar[lhs] = [[name] for name in names]
            elif lhs == "<active_deed>" or lhs == "<passive_deed>":
                grammar[lhs] = [[item] for item in rhs]
    else:
        grammar = copy.deepcopy(NO_COLOR_NO_TYPE)
        for lhs, rhs in statistics.items():
            if lhs == "<t>":
                names = list(set([find_supertype(stuff) for stuff in rhs]))
                grammar[lhs] = [[name] for name in names]
            elif lhs == "<active_deed>" or lhs == "<passive_deed>":
                grammar[lhs] = [[item] for item in rhs]
            elif lhs != "<p>":
                grammar[lhs] = [[item] for item in rhs + ["nil"]]
    # finally, remove non-terminal token that can'tbe grounded in our particular scene.
    remove_key = set()
    for lhs, rhs in grammar.items():
        if len(rhs) == 0:
            remove_key.add(lhs)
    new_grammar = copy.deepcopy(grammar)
    for token in remove_key:
        new_grammar.pop(token)
    for token in remove_key:
        for lhs, rules in new_grammar.items():
            new_rule = []
            for rhs in rules:
                if token in rhs:
                    continue
                new_rule.append(rhs)
            new_grammar[lhs] = new_rule
    return new_grammar
def generate_dynamic_questions_nuscene(episode, templates, max_per_type=5, choose=3, attempts_per_type=100, verbose=False):
    frame_files = extract_frames(episode)
    graph = TemporalGraph(frame_files)
    print(f"Generating dynamic questions for {episode}...")
    print(f"KEY FRAME at{graph.framepaths[graph.idx_key_frame]}")
    print(f"Key frame is {graph.idx_key_frame}")
    print(f"Total frame number {len(graph.frames)}")
    context_string = generate_context_string(graph)
    if verbose:
        print(f"Context: {context_string}")
    candidates, counts, valid_questions = defaultdict(list), 0, set()
    for question_type, question_template in templates.items():
        if question_type in ["color_identification", "color_identification_unique", "type_identification",
                             "type_identification_unique"]:
            continue
        grammar = generate_trimmed_grammar_nuscenes(graph, question_template)
        countdown, generated = attempts_per_type, 0
        while countdown > 0 and generated < max_per_type:
            if verbose:
                print("Attempt {} of {} for {}".format(attempts_per_type - countdown, attempts_per_type, question_type))
            q = DynamicQuerySpecifier(
                type=question_type, template=question_template, parameters=None, graph=graph,
                grammar=grammar, debug=False, stats=True,
            )
            if q.signature in valid_questions:
                if verbose:
                    print("Skip <{}> since the equivalent question has been asked before".format(q.translate()))
                continue
            result = q.export_qa()
            if verbose:
                print(result)
            if not_degenerate(question_type, q, result):
                candidates = add_to_record(question_type, q, result, candidates)
                # print(candidates)
                valid_questions.add(q.signature)
                generated += len(result)
            countdown -= 1
        if generated > choose:
            candidate = candidates[question_type]
            final_selected = random.sample(candidate, choose)
            candidates[question_type] = final_selected
            counts += len(final_selected)
        else:
            counts += len(candidates[question_type])
    return candidates, counts, context_string

def generate():
    session_name = "/bigdata/weizhen/metavqa_iclr/scenarios/nuscenes/"  # "test_collision"
    episode_folders = load_valid_episodes(session_name)
    print(len(episode_folders))
    #exit()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    storage_folder = "test_temporal"
    abs_storage_folder = os.path.join(current_directory, storage_folder)
    os.makedirs(abs_storage_folder, exist_ok=True)
    source = "CAT"
    name = "dynamic_all"
    templates_path = os.path.join(current_directory, "question_templates.json")
    templates = json.load(open(templates_path, "r"))
    templates = templates["dynamic"]
    templates = {
        # "localization": templates["localization"]
        # "counting": templates["counting"],
        # "count_equal_binary": templates["count_equal_binary"]
        # "count_more_binary": templates["count_more_binary"],
        # "color_identification": templates["color_identification"]
        # "type_identification": templates["type_identification"],
        # "color_identification_unique": templates["color_identification_unique"],
        "type_identification_unique": templates["type_identification_unique"]
        # "identify_stationary": templates["identify_stationary"]
        # "identify_heading": templates["identify_heading"]
        #"predict_trajectory": templates["predict_trajectory"]
    }
    qa_tuples = {}
    idx = 0
    for episode in episode_folders[:2]:
        assert len(DynamicQuerySpecifier.CACHE) == 0, f"Non empty cache for {episode}"
        observations = extract_observations(episode)
        records, num_questions, context = generate_dynamic_questions(
            episode, templates, max_per_type=3, choose=2, attempts_per_type=100, verbose=True)
        for question_type, record_list in records.items():
            for record in record_list:
                qa_tuples[idx] = dict(
                    context=context, #question=" ".join([record["question"], context])
                    question=record["question"], answer=record["answer"], explanation=record["explanation"],
                    question_type="|".join(["dynamic", question_type]), answer_form=record["answer_form"],
                    type_statistics=record["type_statistics"], pos_statistics=record["pos_statistics"],
                    color_statistics=record["color_statistics"], action_statistics=record["action_statistics"],
                    interaction_statistics=record["interaction_statistics"], ids=record["ids"],
                    rgb={
                        perspective: observations[perspective][:record["key_frame"] + 1] for perspective in
                        ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                    },
                    lidar=observations["lidar"][:record["key_frame"] + 1],
                    metadrive_scene=extract_frames(episode),
                    multiview=True,
                    source=source,
                )
                idx += 1
        DynamicQuerySpecifier.CACHE.clear()
    json.dump(qa_tuples, open(os.path.join(abs_storage_folder, f"{name}.json"), "w"), indent=2)


if __name__ == "__main__":
    generate()
    #print(find_nuscene_frames("C:/Users/arnoe/Downloads/real"))
