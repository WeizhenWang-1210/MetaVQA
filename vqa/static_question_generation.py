import json
import os

from vqa.grammar import STATIC_GRAMMAR
from vqa.object_node import nodify
from vqa.question_generator import GRAMMAR, QuerySpecifier
from vqa.grammar import NO_COLOR_STATIC, NO_TYPE_STATIC
from vqa.scene_graph import SceneGraph
from vqa.question_generator import CACHE
from vqa.object_node import transform


def generate_all_frame(templates, frame: str, attempts: int, max: int, id_start: int, verbose: bool = False) -> dict:
    '''
    Take in a path to a world.json file(a single frame), generate all
    static questions
    '''
    try:
        with open(frame, 'r') as file:
            scene_dict = json.load(file)
    except Exception as e:
        raise e
    print("Working on scene {}".format(frame))
    ego_id, nodelist = nodify(scene_dict)
    graph = SceneGraph(ego_id, nodelist, frame)
    # Based on the objects/colors that actually exist in this frame, reduce the size of the CFG
    record = {}
    counts = 0
    valid_questions = set()
    """templates = {
        # "color_identification_unique": templates["color_identification_unique"]
        "type_identification_unique": templates["type_identification_unique"]
    }"""
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
            # question = q.translate()
            # answer = q.answer()
            result = q.export_qa()
            if verbose:
                print(result)
            for question, answer in result:
                """                if verbose:
                                    print(question, answer)"""
                if question_type == "counting":

                    if answer[0] > 0:
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"]
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                        if type_count >= max:
                            break
                            # return record
                elif question_type == "localization":
                    if len(answer) != 0:
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            answer_form="bboxs",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"]
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                        if type_count >= max:
                            break
                            # return record
                elif question_type == "count_equal_binary" or question_type == "count_more_binary":
                    degenerate = True
                    for param, info in q.parameters.items():
                        if len(q.parameters[param]["answer"]) > 0:
                            degenerate = False
                            break
                    if not degenerate:
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"]
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                        if type_count >= max:
                            break
                            # return record
                elif question_type in ["color_identification", "type_identification"]:
                    if len(answer) > 0:
                        record[id_start + counts + type_count] = dict(
                            question=question,
                            answer=answer,
                            answer_form="str",
                            question_type=question_type,
                            type_statistics=q.statistics["types"],
                            pos_statistics=q.statistics["pos"]
                            # already in ego's coordinate, with ego's heading as the +y direction
                        )
                        type_count += 1
                        valid_questions.add(q.signature)
                        if type_count >= max:
                            break
                elif question_type in ["color_identification_unique", "type_identification_unique"]:
                    obj_id, answer = answer
                    record[id_start + counts + type_count] = dict(
                        question=question,
                        answer=answer,
                        answer_form="str",
                        question_type=question_type,
                        type_statistics=[q.graph.get_node(obj_id).type],
                        pos_statistics=transform(q.graph.get_ego_node(), [q.graph.get_node(obj_id).pos])
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


def static_all(root_folder, source, summary_path, verbose=False):
    # TODO Debug so that all static questions can be generated efficiently
    def find_world_json_paths(root_dir):
        world_json_paths = []  # List to hold paths to all world_{id}.json files
        for root, dirs, files in os.walk(root_dir):
            # Extract the last part of the current path, which should be the frame folder's name
            frame_folder_name = os.path.basename(root)
            expected_json_filename = f'world_{frame_folder_name}.json'  # Construct expected filename
            if expected_json_filename in files:
                path = os.path.join(root, expected_json_filename)  # Construct full path
                world_json_paths.append(path)
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
        assert len(CACHE) == 0, f"Non empty cache for {path}"
        folder_name = os.path.dirname(path)
        identifier = os.path.basename(folder_name)
        perspectives = ["front", "leftb", "leftf", "rightb", "rightf", "back"]
        # rgb = os.path.join(folder_name, f"rgb_{identifier}.png")
        lidar = os.path.join(folder_name, f"lidar_{identifier}.json")
        record, num_data = generate_all_frame(templates["generic"], path, 100, 10, count, verbose=verbose)
        for id, info in record.items():
            records[id] = dict(
                question=info["question"],
                answer=info["answer"],
                question_type=info["question_type"],
                answer_form=info["answer_form"],
                type_statistics=info["type_statistics"],
                pos_statistics=info["pos_statistics"],
                rgb={perspective: [os.path.join(folder_name, f'rgb_{perspective}_{identifier}.png')] for perspective in perspectives},
                lidar=lidar,
                source=source,
            )
        count += num_data
        CACHE.clear()
    try:
        with open(summary_path, "w") as f:
            json.dump(records, f, indent=4),
    except Exception as e:
        raise e


if __name__ == "__main__":
    static_all("verification_multiview_small", "NuScenes", "verification_multiview_small/static.json", verbose=True)
