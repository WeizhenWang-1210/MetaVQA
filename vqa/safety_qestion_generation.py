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
from vqa.object_node import transform
import random
from vqa.dynamic_question_generation import find_episodes, extract_observations
from collections import defaultdict
import numpy as np

from vqa.scene_level_functionals import sample_keypoints, extrapolate_bounding_boxes, counterfactual_trajectory, \
    counterfactual_stop, locate_crash_timestamp, move_around, predict_collision
from vqa.object_node import box_overlap, transform_vec


def choices(question_type, graph):
    if question_type == "counterfactual_trajectory":
        t = random.choice([0, 10])
        tnow = len(graph.frames)
        ego = graph.get_ego_node()
        ego_trajectory = ego.positions[t:]

        sampled_ego_trajectory = sample_keypoints(np.array(ego_trajectory), num_points=2, sqrt_std_max=16)
        sampled_ego_trajectory = [(x, y) for x, y in zip(sampled_ego_trajectory[0], sampled_ego_trajectory[1])]
        sampled_ego_bboxes = extrapolate_bounding_boxes(sampled_ego_trajectory,
                                                        np.arctan2(ego.headings[0][1], ego.headings[0][0]),
                                                        ego.bboxes[0])
        injected_ego_bboxes = ego.bboxes[:t] + sampled_ego_bboxes
        answer = counterfactual_trajectory(graph, injected_ego_bboxes)
        config = {
            "<t>": t, "<tnow>": tnow, "<traj>": sampled_ego_trajectory, "ans": answer
        }
    elif question_type == "stop_safe":
        t = random.choice([0, 10])
        tnow = len(graph.frames)
        answer = counterfactual_stop(graph, t)
        config = {
            "<t>": t, "<tnow>": tnow, "ans": answer
        }
    elif question_type == "move_around":
        tnow = len(graph.frames)
        first_impact_step = locate_crash_timestamp(graph)
        std = 4
        offset = np.random.normal(0, std, (2)).tolist()
        ego = graph.get_ego_node()
        new_box_at_impact = move_around(ego.bboxes, offset, first_impact_step)
        answer = True
        for id, node in graph.get_nodes().items():
            if id != ego.id and box_overlap(new_box_at_impact, node.bboxes[first_impact_step]):
                answer = False
                break
        config = {
            "<d>": offset, "<tnow>": tnow, "answer": answer
        }
    elif question_type == "predict_collision":
        flag, objects = predict_collision(graph)
        tnow = len(graph.frames)
        config = {
            "answer": flag,
            "collided_objects": objects,
            "<tnow>": tnow
        }
    else:
        print("unknown question type")
        exit()
    return config


def postprocess(question_type, template, record, origin, positive_x):
    question_string = template["text"]
    answer = record["answer"]
    tnow = record["<tnow>"]
    if question_type == "predict_collision":
        return question_string, answer
    elif question_type == "counterfactual_trajectory":
        final_trajectory = [transform_vec(origin, positive_x, point) for point in record["<traj>"]]
        t = record["<t>"]
        question_string.replace("<traj>", final_trajectory)
        question_string.replace("<tnow>", tnow)
        question_string.replace("<t>", t)
    elif question_type == "stop_safe":
        t = record["<t>"]
        question_string.replace("<tnow>", tnow)
        question_string.replace("<t>", t)
    elif question_type == "move_around":
        d, tnow = record["<d>"], record["<tnow>"]
        final_displacement = transform_vec(origin, positive_x, d)
        question_string.replace("<d>", final_displacement)
        question_string.replace("<tnow>", tnow)
    else:
        print("Unknown question")
        exit()
    record = {
        "question": question_string, "answer_form": "string",
        "answer": answer,
        "key_frame": tnow
    }
    return record


def postprocess_all(question_type, template, records, graph):
    results = []
    ego_node = graph.get_ego_node()
    for record in records:
        tnow = record["<tnow>"]
        origin = ego_node.positions[tnow]
        positive_x = ego_node.headings[tnow]
        results.append(postprocess(question_type, template, record, origin, positive_x))
    return results


def generate_safety_questions(episode, templates, max_per_type=5, choose=3, attempts_per_type=100, verbose=False):
    """
Every positional information will be transformed to the ego coordinate at the time of the postmortem analysis

    """
    annotation_template = f"{episode}/**/world*.json"
    frame_files = sorted(glob.glob(annotation_template, recursive=True))
    graph = TemporalGraph(frame_files)
    print(f"Generating safety-critical questions for {episode}...")
    print(f"KEY FRAME at{graph.framepaths[graph.idx_key_frame]}")
    print(f"Key frame is {graph.idx_key_frame}")
    print(f"Total frame number {len(graph.frames)}")
    candidates, counts, valid_questions = defaultdict(list), 0, set()
    for question_type, question_template in templates.items():
        countdown, generated = attempts_per_type, 0
        while countdown > 0 and generated < max_per_type:
            if verbose:
                print("Attempt {} of {} for {}".format(attempts_per_type - countdown, attempts_per_type, question_type))
            record = choices(question_template, graph)
            signature = json.dumps(record)
            if signature in valid_questions and verbose:
                print("Skip <{}> since the equivalent question has been asked before".format(signature))
                continue
            valid_questions.add("_".join([question_type, signature]))
            countdown -= 1
        if generated > choose:
            candidate = candidates[question_type]
            final_selected = random.sample(candidate, choose)
            candidates[question_type] = postprocess_all(question_type, question_template, final_selected, graph)
            counts += len(final_selected)
        else:
            counts += len(candidates[question_type])
    return candidates, counts


def generate_safety():
    session_name = "test_collision"
    episode_folders = find_episodes(session_name)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    storage_folder = "test_safety"
    abs_storage_folder = os.path.join(current_directory, storage_folder)
    os.makedirs(abs_storage_folder, exist_ok=True)
    source = "CAT"
    name = "safety_all"
    templates_path = os.path.join(current_directory, "question_templates.json")
    templates = json.load(open(templates_path, "r"))
    templates = templates["safety"]
    qa_tuples = {}
    idx = 0
    for episode in episode_folders[:3]:
        observations = extract_observations(episode)
        records, num_questions = generate_safety_questions(
            episode, templates, max_per_type=100, choose=10, attempts_per_type=200, verbose=True)
        for question_type, record_list in records.items():
            for record in record_list:
                qa_tuples[idx] = dict(
                    question=record["question"], answer=record["answer"],
                    question_type=question_type, answer_form=record["answer_form"],
                    rgb={
                        perspective: observations[perspective][:record["key_frame"] + 1] for perspective in
                        ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                    },
                    lidar=observations["lidar"][:record["key_frame"] + 1],
                    source=source,
                    metadrive_scene=episode,
                )
                idx += 1
    json.dump(qa_tuples, open(os.path.join(abs_storage_folder, f"{name}.json"), "w"), indent=2)


if __name__ == "__main__":
    generate_safety()
