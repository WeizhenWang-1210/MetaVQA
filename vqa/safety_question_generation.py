# find_all_episodes
# for each episode:
#   for each question type:
#       try for x times, get at most N valid questions
#       sample 4 out of N valid questions
#
import json
import os
import glob
from vqa.scene_graph import TemporalGraph
import random
from vqa.dynamic_question_generation import find_episodes, extract_observations
from collections import defaultdict
import numpy as np

from vqa.scene_level_functionals import sample_keypoints, extrapolate_bounding_boxes, counterfactual_trajectory, \
    counterfactual_stop, locate_crash_timestamp, move_around, predict_collision
from vqa.object_node import box_overlap, transform_vec
from vqa.dynamic_question_generation import extract_frames, generate_context_string


def choices(question_type, graph):
    if question_type == "counterfactual_trajectory":
        """
        Our episode contains from [0,19]
        We observe [0,19]
        we start injecting at step [0,10]
        """
        t = 10#random.choice([10])
        tnow = len(graph.frames) - 1
        ego = graph.get_ego_node()
        ego_trajectory = ego.positions[t:]
        sampled_ego_trajectory = sample_keypoints(np.array(ego_trajectory), num_points=2, sqrt_std_max=16)
        sampled_ego_trajectory = [(x, y) for x, y in zip(sampled_ego_trajectory[0], sampled_ego_trajectory[1])]
        sampled_ego_bboxes = extrapolate_bounding_boxes(sampled_ego_trajectory,
                                                        np.arctan2(ego.headings[0][1], ego.headings[0][0]),
                                                        ego.bboxes[0])
        injected_ego_bboxes = ego.bboxes[:t] + sampled_ego_bboxes
        answer = counterfactual_trajectory(graph, injected_ego_bboxes)
        traj_str = [str(point) for point in sampled_ego_trajectory[::2]]
        traj = ",".join(traj_str)
        config = {
            "<t>": t, "<tnow>": tnow, "<traj>": traj, "answer": answer
        }
    elif question_type == "stop_safe":
        t = random.choice([0, 10])
        tnow = len(graph.frames) - 1
        answer = counterfactual_stop(graph, t)
        config = {
            "<t>": t, "<tnow>": tnow, "answer": answer
        }
    elif question_type == "move_around":
        tnow = len(graph.frames) - 1
        first_impact_step = locate_crash_timestamp(graph)
        std = 4
        offset = np.random.normal(0, std, 2).tolist()
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
        flag, objects, collision_step = predict_collision(graph)
        tnow = 19
        config = {
            "answer": flag,
            "collided_objects": objects,
            "<tnow>": tnow,
        }
    else:
        print("unknown question type")
        exit()
    return config


def postprocess(question_type, template, record, origin, positive_x, graph):
    question_string = template["text"][0]
    answer = record["answer"]
    answer_string = "Yes." if answer else "No."
    tnow = record["<tnow>"]
    if question_type == "predict_collision":
        context_string = generate_context_string(graph)
        question_string = " ".join([question_string, context_string])
        #print(record["collided_objects"])
        object_collided = record["collided_objects"][0] if len(record["collided_objects"]) > 0 else []
        if len(object_collided) > 0 and answer:
            collided_object = graph.get_node(object_collided)
            color = collided_object.color
            type = collided_object.type
            current_location = transform_vec(origin, positive_x, [collided_object.positions[tnow]])[0]
            current_location = [round(current_location[0], 1), round(current_location[1], 1)]
            answer_string = (
                "Yes. We could collide with the {} {} that's at {} currently.".format(color.lower(), type.lower(),
                                                                           current_location))
        else:
            answer_string = "No"
    elif question_type == "counterfactual_trajectory":
        final_trajectory = [transform_vec(origin, positive_x, [point])[0] for point in record["<traj>"]]
        final_trajectory = [[round(x, 1), round(y, 1)] for (x, y) in final_trajectory]
        t = record["<t>"]
        question_string = question_string.replace("<traj>", str(final_trajectory))
        question_string = question_string.replace("<tnow>", str(tnow))
        question_string = question_string.replace("<t>", str(t))
        context_string = generate_context_string(graph, end=True)
        question_string = " ".join([question_string, context_string])

    elif question_type == "stop_safe":
        t = record["<t>"]
        question_string = question_string.replace("<tnow>", str(tnow))
        question_string = question_string.replace("<t>", str(t))
        context_string = generate_context_string(graph, end=True)
        question_string = " ".join([question_string, context_string])
    elif question_type == "move_around":
        d, tnow = record["<d>"], record["<tnow>"]
        final_displacement = d #transform_vec(origin, positive_x, [d])[0]
        final_displacement = [round(final_displacement[0], 1), round(final_displacement[1], 1)]
        question_string = question_string.replace("<d>", str(final_displacement))
        question_string = question_string.replace("<tnow>", str(tnow))
        context_string = generate_context_string(graph, end=True)
        question_string = " ".join([question_string, context_string])
    else:
        print("Unknown question")
        exit()
    record = {
        "question": question_string, "answer_form": "str",
        "answer": answer_string,
        "key_frame": tnow
    }
    return record


def postprocess_all(question_type, template, records, graph):
    results = []
    ego_node = graph.get_ego_node()
    for record in records:
        try:
            tnow = record["<tnow>"]
            origin = ego_node.positions[tnow]
            positive_x = ego_node.headings[tnow]
        except:
            print(record["<tnow>"])
        results.append(postprocess(question_type, template, record, origin, positive_x, graph))
    return results


def generate_safety_questions(episode, templates, max_per_type=5, choose=3, attempts_per_type=100, verbose=False,
                              only_predict=False):
    """
    Every positional information will be transformed to the ego coordinate at the time of the postmortem analysis
    """
    annotation_template = f"{episode}/**/world*.json"
    frame_files = extract_frames(episode)#sorted(glob.glob(annotation_template, recursive=True))
    predict_files = frame_files
    postmortem_files = frame_files[5:]
    predict_graph = TemporalGraph(predict_files, tolerance=0.5)
    postmortem_graph = TemporalGraph(postmortem_files, tolerance=0.5)
    print(f"Generating safety-critical questions for {episode}...")
    print(f"KEY FRAME at{predict_graph.framepaths[predict_graph.idx_key_frame]} for predict_graph")
    print(f"KEY frame is {predict_graph.idx_key_frame} for predict_graph")
    print(f"Total frame number {len(predict_graph.frames)} for predict_graph")
    print(f"Total frame number {len(postmortem_graph.frames)} for postmortem_graph")
    default_max_per_type = max_per_type
    candidates, counts, valid_questions = defaultdict(list), 0, set()
    for question_type, question_template in templates.items():
        if only_predict and question_type != "predict_collision":
            continue
        if question_type == "predict_collision":
            max_per_type = 1
            graph = predict_graph
        else:
            max_per_type = default_max_per_type
            graph = postmortem_graph
        countdown, generated = attempts_per_type, 0
        while countdown > 0 and generated < max_per_type:
            if verbose:
                print("Attempt {} of {} for {}".format(attempts_per_type - countdown, attempts_per_type, question_type))
            record = choices(question_type, graph)
            signature = json.dumps(record)
            if signature in valid_questions and verbose:
                print("Skip <{}> since the equivalent question has been asked before".format(signature))
                continue
            valid_questions.add("_".join([question_type, signature]))
            candidates[question_type].append(record)
            generated += 1
            countdown -= 1
        if generated > choose:
            candidate = candidates[question_type]
            final_selected = random.sample(candidate, choose)
            candidates[question_type] = postprocess_all(question_type, question_template, final_selected, graph)
            counts += len(final_selected)
        else:
            candidate = candidates[question_type]
            candidates[question_type] = postprocess_all(question_type, question_template, candidate, graph)
            counts += len(candidates[question_type])
    return candidates, counts

def generate_safety():
    session_name = "collision_scenarios_final"
    episode_folders = find_episodes(session_name)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    storage_folder = "test_safety"
    abs_storage_folder = os.path.join(current_directory, storage_folder)
    os.makedirs(abs_storage_folder, exist_ok=True)
    source = "CAT"
    name = "safety_all_some"
    templates_path = os.path.join(current_directory, "question_templates.json")
    templates = json.load(open(templates_path, "r"))
    templates = templates["safety"]
    qa_tuples = {}
    idx = 0
    for episode in episode_folders:
        observations = extract_observations(episode)
        records, num_questions = generate_safety_questions(
            episode, templates, max_per_type=100, choose=2, attempts_per_type=200, verbose=True)
        for question_type, record_list in records.items():
            if question_type == "predict_collision":
                """
                Only in predict_collision question, we cannot see the full future.
                [0,24] total, [0,19] for observations
                """
                for record in record_list:
                    qa_tuples[idx] = dict(
                        question=record["question"], answer=record["answer"],
                        question_type="|".join(["safety", question_type]), answer_form=record["answer_form"],
                        rgb={
                            perspective: observations[perspective][:record["key_frame"] + 1] for perspective in
                            ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                        },
                        lidar=observations["lidar"][:record["key_frame"] + 1],
                        metadrive_scene=extract_frames(episode),
                        multiview=True,
                        source=source
                    )
                    idx += 1
            else:
                """
                In postmortem questions, we need to see the entire collision happening. Therefore, we cannot start from
                the beginning of the episode. Instead, we start at the frame 10(0-indexed) inclusive till the collision.
                [0,24], [5,24] for observations
                """
                for record in record_list:
                    qa_tuples[idx] = dict(
                        question=record["question"], answer=record["answer"],
                        question_type="|".join(["safety", question_type]), answer_form=record["answer_form"],
                        rgb={
                            perspective: observations[perspective][5:] for perspective in
                            ["front", "leftb", "leftf", "rightb", "rightf", "back"]
                        },
                        lidar=observations["lidar"][5:],
                        metadrive_scene=extract_frames(episode)[5:],
                        multiview=True,
                        source=source
                    )
                    idx += 1
        break
    json.dump(qa_tuples, open(os.path.join(abs_storage_folder, f"{name}.json"), "w"), indent=2)


if __name__ == "__main__":
    generate_safety()
