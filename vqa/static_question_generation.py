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
    #print(graph.statistics)
    #print(GRAMMAR)
    # Based on the objects/colors that actually exist in this frame, reduce the size of the CFG
    for lhs, rhs in graph.statistics.items():
        GRAMMAR[lhs] = [[item] for item in rhs + ['nil']]
    print(rhs)
    #print(GRAMMAR)
    record = {}
    templates = {"localization": templates["localization"]}
    # Cache seen problems. Skip them if resampled
    seen_problems = set()
    counts = 0
    for question_type, specification in templates.items():
        for idx in range(attempts):
            print(idx)
            q = QuerySpecifier(template=specification, parameters=None, graph=graph, grammar=GRAMMAR)
            if q.signature in seen_problems:
                #print(q.signature)
                continue
            else:
                seen_problems.add(q.signature)
                #print(seen_problems)
            question = q.translate()
            answer = q.answer()
            if len(answer) != 0:
                counts += 1
                print(question, answer)
                record[question] = answer
                if counts >= max:
                    return record
            print(len(seen_problems))

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