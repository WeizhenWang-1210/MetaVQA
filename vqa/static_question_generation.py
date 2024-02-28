import json
import os

from vqa.grammar import STATIC_GRAMMAR
from vqa.object_node import nodify
from vqa.question_generator import GRAMMAR, QuerySpecifier
from vqa.scene_graph import SceneGraph


def generate_all_frame(templates, frame: str, attempts: int, max:int) -> dict:
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
    for lhs, rhs in graph.statistics.items():
        GRAMMAR[lhs] = [[item] for item in rhs + ['nil']]
    record = {}
    templates = {"count_more_binary": templates["count_more_binary"]}
    # Cache seen problems. Skip them if resampled
    seen_problems = set()
    counts = 0
    for question_type, specification in templates.items():
        for idx in range(attempts):
            print("Attempt {} of {} for {}".format(idx, attempts, question_type))
            q = QuerySpecifier(template=specification, parameters=None, graph=graph, grammar=GRAMMAR)
            """if q.signature in seen_problems:
                continue
            else:
                seen_problems.add(q.signature)"""
            question = q.translate()
            answer = q.answer()
            if question_type == "counting":
                if answer[0] > 0:
                    counts += 1
                    record[question] = answer
                    if counts >= max:
                        return record
            elif question_type == "localization":
                if len(answer) != 0:
                    counts += 1
                    record[question] = answer
                    if counts >= max:
                        return record
            elif question_type == "count_equal_binary":
                degenerate = True
                for param, info in q.parameters.items():
                    if len(q.parameters[param]["answer"]) > 0:
                        degenerate = False
                        break
                if not degenerate:
                    counts += 1
                    record[question] = answer
                    if counts >= max:
                        return record
            elif question_type == "count_more_binary":
                degenerate = True
                for param, info in q.parameters.items():
                    if len(q.parameters[param]["answer"]) > 0:
                        degenerate
                        break
                if not degenerate:
                    counts += 1
                    record[question] = answer
                    if counts >= max:
                        return record
            else:
                print("Unknown question type!")
    return record


def static_all(root_folder):
    GRAMMAR = STATIC_GRAMMAR

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
    for path in paths:
        record = generate_all_frame(templates["generic"], path, 100,3)
        records[path] = record
        break
    for path, record in records.items():
        folder = os.path.dirname(path)
        try:
            with open(os.path.join(folder, "static_questions.json"), "w") as f:
                json.dump(record, f, indent=4)
        except Exception as e:
            raise e


if __name__ == "__main__":
    static_all("verification")