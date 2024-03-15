import json
import os

from vqa.grammar import STATIC_GRAMMAR
from vqa.object_node import nodify
from vqa.question_generator import GRAMMAR, QuerySpecifier
from vqa.scene_graph import SceneGraph
from vqa.question_generator import CACHE



def generate_all_frame(templates, frame: str, attempts: int, max:int, id_start:int, verbose:bool =False) -> dict:
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
    #templates = {"count_more_binary":templates["count_more_binary"]}
    grammar = GRAMMAR
    for lhs, rhs in graph.statistics.items():
        #GRAMMAR[lhs] = [[item] for item in rhs + ['nil']]
        if lhs == "<p>":
            grammar[lhs] = [[item] for item in rhs+["nil"]]
        else:
            grammar[lhs] = [[item] for item in rhs + ["vehicle"]]
    record = {}
    counts = 0
    for question_type, specification in templates.items():
        for idx in range(attempts):
            if verbose:
                print("Attempt {} of {} for {}".format(idx, attempts, question_type))
            q = QuerySpecifier(template=specification, parameters=None, graph=graph, grammar=grammar, debug = True, stats = True)
            question = q.translate()
            answer = q.answer()
            if verbose:
                print(question, answer)
            if question_type == "counting":
                if answer[0] > 0:
                    record[id_start+counts] = dict(
                        question = question,
                        answer = answer,
                        answer_form = "str",
                        question_type = question_type,
                        type_statistics = q.statistics["types"],
                        pos_statistics =q.statistics["pos"] #already in ego's coordinate, with ego's heading as the +y direction
                    )
                    counts += 1
                    if counts >= max:
                        break
                        #return record
            elif question_type == "localization":
                if len(answer) != 0:
                    record[id_start + counts] = dict(
                        question=question,
                        answer=answer,
                        answer_form="bboxs",
                        question_type=question_type,
                        type_statistics=q.statistics["types"],
                        pos_statistics=q.statistics["pos"]
                        # already in ego's coordinate, with ego's heading as the +y direction
                    )
                    counts += 1
                    if counts >= max:
                        break
                        #return record
            elif question_type == "count_equal_binary" or question_type == "count_more_binary":
                degenerate = True
                for param, info in q.parameters.items():
                    if len(q.parameters[param]["answer"]) > 0:
                        degenerate = False
                        break
                if not degenerate:
                    record[id_start + counts] = dict(
                        question=question,
                        answer=answer,
                        answer_form="str",
                        question_type=question_type,
                        type_statistics=q.statistics["types"],
                        pos_statistics=q.statistics["pos"]
                        # already in ego's coordinate, with ego's heading as the +y direction
                    )
                    counts += 1
                    if counts >= max:
                        break
                        #return record
            else:
                print("Unknown question type!")
    if verbose:
        print("{} questions generated for {}".format(counts, frame))
    return record, counts


def static_all(root_folder, source, summary_path, verbose = False):
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
    count = 0
    for path in paths:
        assert len(CACHE)==0, f"Non empty cache for {path}"
        folder_name = os.path.dirname(path)
        identifider = os.path.basename(folder_name)
        rgb = os.path.join(folder_name,f"rgb_{identifider}.png")
        lidar = os.path.join(folder_name,f"lidar_{identifider}.json")
        record,num_data = generate_all_frame(templates["generic"], path, 100,2, count, verbose = verbose)
        for id, info in record.items():
            records[id] = dict(
                question = info["question"],
                answer = info["answer"],
                question_type = info["question_type"],
                answer_form = info["answer_form"],
                type_statistics = info["type_statistics"],
                pos_statistics = info["pos_statistics"],
                rgb = dict(
                    front = [rgb],
                    left=[],
                    back=[],
                    right=[]
                ),
                lidar = lidar,
                source = source
            )
        count += num_data
        CACHE.clear()
    try:
        with open(summary_path, "w") as f:
            json.dump(records,f,indent=4),
    except Exception as e:
        raise e


if __name__ == "__main__":
    static_all("some", "NuScenes", ".some/static.json", verbose= True)