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
from vqa.dynamic_question_generator import DynamicQuerySpecifier
from vqa.object_node import transform
import random
from collections import defaultdict


def find_episodes(session_folder):
    """
    Return subfolders representing an episode under a session folder.
    """
    pattern = re.compile(r'\d+_\d+_\d+')
    folders = [f for f in os.listdir(session_folder) if os.path.isdir(os.path.join(session_folder, f))]
    matched_folders = [os.path.join(session_folder, folder) for folder in folders if pattern.match(folder)]
    return matched_folders


def extract_frames(episode, debug=False):
    annotation_path_template = f"{episode}/**/world*.json"
    sorted_rgb = sorted(glob.glob(annotation_path_template, recursive=True))
    return sorted_rgb


def extract_observations(episode, debug=False):
    observations = {}
    perspectives = ["front", "leftb", "leftf", "rightb", "rightf", "back"]
    for perspective in perspectives:
        rgb_path_template = f"{episode}/**/rgb_{perspective}**.png"
        sorted_rgb = sorted(glob.glob(rgb_path_template, recursive=True))
        observations[perspective] = sorted_rgb
    lidar_template = f"{episode}/**/lidar_**.json"
    sorted_lidar = sorted(glob.glob(lidar_template, recursive=True))
    observations["lidar"] = sorted_lidar
    if debug:
        length = len(observations["lidar"])
        for value in observations.values():
            assert len(value) == length
    return observations


def generate_trimmed_grammar(graph, tense, template):
    from vqa.grammar import NO_STATE_CFG, NO_COLOR_STATIC, NO_TYPE_STATIC, STATIC_GRAMMAR, CFG_GRAMMAR,\
        NO_COLOR_CFG,NO_TYPE_CFG
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
            obj_node = q.graph.get_node(obj_id)
            data_point = dict(
                question=question, answer=answer, answer_form="str", question_type=question_type,
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
            data_point = dict(
                question=question, answer=answer, answer_form="str", question_type=question_type,
                type_statistics=[obj_node.type],
                pos_statistics=transform(q.graph.get_ego_node(), [obj_node.pos]),
                color_statistics=[obj_node.color], action_statistics=[obj_node.actions],
                interaction_statistics=[obj_node.get_all_interactions()], ids=[obj_id],
                key_frame=q.graph.idx_key_frame # key_frame will be used to determine the observation.
            )
            candidates[question_type].append(data_point)

    else:
        question, answer = result[0]
        ids, answer = answer
        data_point = dict(
            question=question, answer=answer, answer_form="str", question_type=question_type,
            type_statistics=q.statistics["types"], pos_statistics=q.statistics["pos"],
            color_statistics=q.statistics["colors"], action_statistics=q.statistics["actions"],
            interaction_statistics=q.statistics["interactions"], ids=ids,
            key_frame=q.graph.idx_key_frame # key_frame will be used to determine the observation.
        )
        candidates[question_type].append(data_point)
    return candidates


def generate_dynamic_questions(episode, templates, max_per_type=5, choose=3, attempts_per_type=100, verbose=False):
    annotation_template = f"{episode}/**/world*.json"
    frame_files = sorted(glob.glob(annotation_template, recursive=True))
    graph = TemporalGraph(frame_files)
    print(f"Generating dynamic questions for {episode}...")
    print(f"KEY FRAME at{graph.framepaths[graph.idx_key_frame]}")
    print(f"Key frame is {graph.idx_key_frame}")
    print(f"Total frame number {len(graph.frames)}")
    candidates, counts, valid_questions = defaultdict(list), 0, set()

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
            if not_degenerate(question_type, q, result):
                candidates = add_to_record(question_type, q, result, candidates)
                #print(candidates)
                valid_questions.add(q.signature)
                generated += 1
            countdown -= 1
        if generated > choose:
            candidate = candidates[question_type]
            final_selected = random.sample(candidate, choose)
            candidates[question_type] = final_selected
            counts += len(final_selected)
        else:
            counts += len(candidates[question_type])
    return candidates, counts


def generate():
    session_name = "test_collision"
    episode_folders = find_episodes(session_name)
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
        #"localization": templates["localization"]
        #"counting": templates["counting"],
        #"count_equal_binary": templates["count_equal_binary"]
        #"count_more_binary": templates["count_more_binary"],
        #"color_identification": templates["color_identification"]
        #"type_identification": templates["type_identification"],
        #"color_identification_unique": templates["color_identification_unique"]
        #"identify_stationary": templates["identify_stationary"]

    }
    qa_tuples = {}
    idx = 0
    for episode in episode_folders:
        assert len(DynamicQuerySpecifier.CACHE) == 0, f"Non empty cache for {episode}"
        observations = extract_observations(episode)
        records, num_questions = generate_dynamic_questions(
            episode, templates, max_per_type=3, choose=2, attempts_per_type=100, verbose=True)
        for question_type, record_list in records.items():
            for record in record_list:
                qa_tuples[idx] = dict(
                    question=record["question"], answer=record["answer"],
                    question_type="|".join(["dynamic",question_type]), answer_form=record["answer_form"],
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
