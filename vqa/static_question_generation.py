import json
import os
from vqa.grammar import NO_COLOR_STATIC, NO_TYPE_STATIC
from vqa.grammar import STATIC_GRAMMAR
from vqa.object_node import nodify
from vqa.object_node import transform
from vqa.scene_graph import SceneGraph
from vqa.static_question_generator import QuerySpecifier, NAMED_MAPPING


def generate_all_frame(templates, frame: str, attempts: int, max: int, id_start: int, verbose: bool = False,
                       multiview: bool = True) -> dict:
    '''
    Take in a path to a world.json file(a single frame), generate all
    static questions
    '''
    try:
        with open(frame, 'r') as file:
            scene_dict = json.load(file)
    except Exception as e:
        raise e

    #templates = {
    #"count_equal_binary": templates["count_equal_binary"],
    #"count_more_binary": templates["count_more_binary"]
    #"color_identification": templates["color_identification"],
    #"type_identification": templates["type_identification"]
    #    "color_identification_unique": templates["color_identification_unique"],
    #    "type_identification_unique": templates["type_identification_unique"]
    #}

    print("Working on scene {}".format(frame))
    ego_id, nodelist = nodify(scene_dict, multiview=multiview)
    graph = SceneGraph(ego_id, nodelist, frame)
    # Based on the objects/colors that actually exist in this frame, reduce the size of the CFG
    record = {}
    counts = 0
    valid_questions = set()

    degenerate_allowance = dict(
        counting=2,
        localization=2,
        count_equal_binary=2,
        count_more_binary=2,
        color_identification=2,
        type_identification=2,
        color_identification_unique=0,
        type_identification_unique=0
    )

    for question_type, specification in templates.items():
        type_count = 0
        if question_type == "color_identification" or question_type == "color_identification_unique":
            grammar = NO_COLOR_STATIC
            for lhs, rhs in graph.statistics.items():
                if lhs == "<p>":
                    grammar["<px>"] = [[item] for item in rhs + ["nil"]]
                else:
                    grammar[lhs] = [[item] for item in rhs + ["vehicle"]]
        elif question_type == "type_identification" or question_type == "type_identification_unique":
            grammar = NO_TYPE_STATIC
            for lhs, rhs in graph.statistics.items():
                if lhs == "<t>":
                    grammar["<tx>"] = [[item] for item in rhs + ["vehicle"]]
                else:
                    grammar[lhs] = [[item] for item in rhs + ["nil"]]
        else:
            grammar = STATIC_GRAMMAR
            for lhs, rhs in graph.statistics.items():
                if lhs == "<p>":
                    grammar[lhs] = [[item] for item in rhs + ["nil"]]
                else:
                    grammar[lhs] = [[item] for item in rhs + ["vehicle"]]
        for idx in range(attempts):
            if verbose:
                print("Attempt {} of {} for {}".format(idx, attempts, question_type))
            q = QuerySpecifier(type=question_type, template=specification, parameters=None, graph=graph,
                               grammar=grammar, debug=False, stats=True)
            if q.signature in valid_questions:
                if verbose:
                    print("Skip <{}> since the equivalent question has been asked before".format(q.translate()))
                continue
            result = q.export_qa()
            if verbose:
                print(result)
            for question, answer in result:
                if question_type == "counting":
                    referral_string = q.parameters["<o>"]["en"]
                    ids, answer = answer
                    if answer[0] == 0:
                        explanation = "There is no {}.".format(referral_string)
                    elif answer[0] == 1:
                        explanation = "There is 1 {}.".format(referral_string)
                    else:
                        explanation = "There are {} {}.".format(answer[0], referral_string)
                    if answer[0] > 0:
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                    elif degenerate_allowance[question_type] > 0:
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                        degenerate_allowance[question_type] -= 1
                    if type_count >= max:
                        break
                        # return record
                elif question_type == "localization":
                    ids, answer = answer
                    referral_string = q.parameters["<o>"]["en"]
                    if len(answer) == 0 and degenerate_allowance[question_type] > 0:
                        explanation = "There is no {} at this moment.".format(referral_string)
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                        degenerate_allowance[question_type] -= 1

                    if len(answer) == 1:
                        explanation = "There is 1 {} located here at this moment.".format(referral_string)
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                    if len(answer) > 1:
                        explanation = ("There are {} {} located at the aforementioned positions at this moment."
                                       .format(len(answer), referral_string))
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                    if type_count >= max:
                        break
                        # return record
                elif question_type == "count_equal_binary" or question_type == "count_more_binary":
                    referral_string0, referral_string1 = q.parameters["<o1>"]["en"], q.parameters["<o2>"]["en"]
                    count0, count1 = len(set([obj.id for obj in q.parameters["<o1>"]["answer"]])), len(
                        set([obj.id for obj in q.parameters["<o2>"]["answer"]]))
                    #print(referral_string0, count0)
                    #print(referral_string1, count1)
                    answer_string = "\"True\"" if answer[1] else "\"False\""
                    explanation = "The number of {} is {}, and there {} {} {}. Therefore, the answer is {}".format(
                        referral_string0, count0, "is" if count1 <= 1 else "are", "no" if count1 == 0 else count1,
                        referral_string1,
                        answer_string
                    )
                    ids, answer = answer
                    degenerate = True
                    for param, info in q.parameters.items():
                        if len(q.parameters[param]["answer"]) > 0:
                            degenerate = False
                            break
                    if not degenerate or degenerate_allowance[question_type] > 0:
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        if degenerate:
                            degenerate_allowance[question_type] -= 1
                        type_count += 1
                        valid_questions.add(q.signature)
                        if type_count >= max:
                            break
                            # return record
                elif question_type in ["color_identification", "type_identification"]:
                    ids, answer = answer
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
                    if len(answer) > 0 or degenerate_allowance[question_type] > 0:
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids,
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                        if len(answer) == 0:
                            degenerate_allowance[question_type] -= 1
                        if type_count >= max:
                            break
                elif question_type in ["color_identification_unique", "type_identification_unique"]:
                    obj_id, answer = answer
                    if question_type == "color_identification_unique":
                        explanation = "The color of that {} is {}".format \
                            (NAMED_MAPPING[q.graph.get_node(obj_id).type]["singular"], answer.lower())
                    else:
                        explanation = "The type of that {} thing is {}.".format \
                            (q.graph.get_node(obj_id).color.lower(),
                             NAMED_MAPPING[q.graph.get_node(obj_id).type]["singular"])
                    record[id_start + counts + type_count] = dict(
                        question=question,
                        answer=answer,
                        explanation=explanation,
                        answer_form="str",
                        question_type=question_type,
                        type_statistics=[q.graph.get_node(obj_id).type],
                        pos_statistics=transform(q.graph.get_ego_node(), [q.graph.get_node(obj_id).pos]),
                        color_statistics=[q.graph.get_node(obj_id).color],
                        ids=[obj_id]
                        # already in ego's coordinate, with ego's heading as the +y direction
                    )
                    type_count += 1
                    valid_questions.add(q.signature)
                    if type_count >= max:
                        break
                else:
                    print("Unknown question type!")
            if type_count >= max:
                break
        counts += type_count
    if verbose:
        print("{} questions generated for {}".format(counts, frame))
    return record, counts


def find_supertype(subname):
    if subname in ["Hatchback", "Pickup", "Policecar", "SUV", "Sedan", "SportCar"]:
        return "vehicle"
    else:
        return subname


from vqa.grammar import NO_COLOR_NO_TYPE_STATIC
from collections import defaultdict


def generate_all_frame_nuscene(templates, frame: str, attempts: int, max: int, id_start: int, verbose: bool = False,
                               multiview: bool = True) -> dict:
    """
        Take in a path to a world.json file(a single frame), generate all
        static questions
    """
    try:
        with open(frame, 'r') as file:
            scene_dict = json.load(file)
    except Exception as e:
        raise e

    #templates = {
        #"counting": templates["counting"]
        #"localization":templates["localization"]
    #    "count_equal_binary": templates["count_equal_binary"]
    #}

    print("Working on scene {}".format(frame))
    ego_id, nodelist = nodify(scene_dict, multiview=multiview)
    graph = SceneGraph(ego_id, nodelist, frame)
    # Based on the objects/colors that actually exist in this frame, reduce the size of the CFG
    record = {}
    counts = 0
    valid_questions = set()

    degenerate_allowance = defaultdict(
        counting=2,
        localization=2,
        count_equal_binary=2,
        count_more_binary=2,
    )

    for question_type, specification in templates.items():
        if question_type in ["color_identification", "color_identification_unique", "type_identification",
                             "type_identification_unique"]:
            continue
        type_count = 0
        grammar = NO_COLOR_NO_TYPE_STATIC
        for lhs, rhs in graph.statistics.items():
            if lhs == "<p>":
                continue
            else:
                names = list(set([find_supertype(stuff) for stuff in rhs]))
                grammar[lhs] = [[name] for name in names]
        for idx in range(attempts):
            if verbose:
                print("Attempt {} of {} for {}".format(idx, attempts, question_type))
            q = QuerySpecifier(type=question_type, template=specification, parameters=None, graph=graph,
                               grammar=grammar, debug=False, stats=True)
            if q.signature in valid_questions:
                if verbose:
                    print("Skip <{}> since the equivalent question has been asked before".format(q.translate()))
                continue
            result = q.export_qa()
            if verbose:
                print(result)
            for question, answer in result:
                if question_type == "counting":
                    referral_string = q.parameters["<o>"]["en"]
                    ids, answer = answer
                    if answer[0] == 0:
                        explanation = "There is no {}.".format(referral_string)
                    elif answer[0] == 1:
                        explanation = "There is 1 {}.".format(referral_string)
                    else:
                        explanation = "There are {} {}.".format(answer[0], referral_string)
                    if answer[0] > 0 or degenerate_allowance[question_type] > 0:
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                        if not answer[0] > 0:
                            degenerate_allowance[question_type] -= 1
                        if type_count >= max:
                            break
                            # return record
                elif question_type == "localization":
                    ids, answer = answer
                    referral_string = q.parameters["<o>"]["en"]
                    if len(answer) == 0 and degenerate_allowance[question_type] > 0:
                        explanation = "There is no {} at this moment.".format(referral_string)
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                        degenerate_allowance[question_type] -= 1
                    if len(answer) == 1:
                        explanation = "There is 1 {} located here at this moment.".format(referral_string)
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                    if len(answer) > 1:
                        explanation = ("There are {} {} located at the aforementioned positions at this moment."
                                       .format(len(answer), referral_string))
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                    if type_count >= max:
                        break
                        # return record
                elif question_type == "count_equal_binary" or question_type == "count_more_binary":
                    referral_string0, referral_string1 = q.parameters["<o1>"]["en"], q.parameters["<o2>"]["en"]
                    count0, count1 = len(set([obj.id for obj in q.parameters["<o1>"]["answer"]])), len(
                        set([obj.id for obj in q.parameters["<o2>"]["answer"]]))
                    answer_string = "\"True\"" if answer[1] else "\"False\""
                    explanation = "The number of {} is {}, and there {} {} {}. Therefore, the answer is {}".format(
                        referral_string0, count0, "is" if count1 <= 1 else "are", "no" if count1 == 0 else count1,
                        referral_string1,
                        answer_string
                    )
                    ids, answer = answer
                    degenerate = True
                    for param, info in q.parameters.items():
                        if len(q.parameters[param]["answer"]) > 0:
                            degenerate = False
                            break
                    if not degenerate or degenerate_allowance[question_type] > 0:
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            explanation=explanation,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"],
                            color_statistics=q.statistics["colors"],
                            ids=ids
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        if degenerate:
                            degenerate_allowance[question_type] -= 1
                        type_count += 1
                        valid_questions.add(q.signature)
                        if type_count >= max:
                            break
                            # return record
                else:
                    print(f"Unknown question type!:{question_type}")
            if type_count >= max:
                break
        counts += type_count
    if verbose:
        print("{} questions generated for {}".format(counts, frame))
    return record, counts


def static_all(root_folder, source, summary_path, verbose=False, multiview=True):
    def find_world_json_paths(root_dir):
        #count = 0
        world_json_paths = []  # List to hold paths to all world_{id}.json files
        for root, dirs, files in os.walk(root_dir):
            # Extract the last part of the current path, which should be the frame folder's name
            frame_folder_name = os.path.basename(root)
            expected_json_filename = f'world_{frame_folder_name}.json'  # Construct expected filename
            if expected_json_filename in files:
                path = os.path.join(root, expected_json_filename)  # Construct full path
                world_json_paths.append(path)
                #count += 1
                #if count > 2:
                #    break
        return world_json_paths

    current_directory = os.path.dirname(os.path.abspath(__file__))
    paths = find_world_json_paths(root_folder)
    template_path = os.path.join(current_directory, "question_templates.json")
    try:
        with open(template_path, "r") as f:
            templates = json.load(f)
    except Exception as e:
        raise e
    records = {}
    count = 0
    for path in paths:
        assert len(QuerySpecifier.CACHE) == 0, f"Non empty cache for {path}"
        folder_name = os.path.dirname(path)
        identifier = os.path.basename(folder_name)
        perspectives = ["front", "leftb", "leftf", "rightb", "rightf", "back"] if multiview else ["front"]
        lidar = os.path.join(folder_name, f"lidar_{identifier}.pkl")

        record, num_data = generate_all_frame(templates["static"], path, 100, 10, count, verbose=verbose,
                                                      multiview=multiview)
        for id, info in record.items():
            records[id] = dict(
                question=info["question"],
                answer=info["answer"],
                explanation=info["explanation"],
                question_type="|".join(["static", info["question_type"]]),
                answer_form=info["answer_form"],
                type_statistics=info["type_statistics"],
                pos_statistics=info["pos_statistics"],
                color_statistics=info["color_statistics"],
                ids=info["ids"],
                rgb={perspective: [os.path.join(folder_name, f'rgb_{perspective}_{identifier}.png')] for perspective in
                     perspectives},
                lidar=[lidar],
                metadrive_scene=[path],
                multiview=multiview,
                source=source,
            )
        count += num_data
        QuerySpecifier.CACHE.clear()
    try:
        with open(summary_path, "w") as f:
            json.dump(records, f, indent=2),
    except Exception as e:
        raise e


if __name__ == "__main__":
    #static_all("/bigdata/weizhen/metavqa_final/scenarios/NuScenes_Mixed", "NuScenes",
    #           "./test.json", verbose=True,
    #           multiview=True)
    static_all("/bigdata/weizhen/metavqa_iclr/scenarios/nuscenes", "NuScenes",
               "./test.json", verbose=True,
               multiview=True)
