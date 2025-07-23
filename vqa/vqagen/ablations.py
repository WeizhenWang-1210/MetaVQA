import copy
import json
import os
import random
import traceback

import numpy as np
import tqdm

from vqa.vqagen.grounding_question import SETTINGS
from vqa.vqagen.parameterized_questions import parameterized_generate
from vqa.vqagen.set_of_marks import grounding_labelframe, static_id2label, labelframe
from vqa.vqagen.static_question_generation import TEMPLATES, generate
from vqa.vqagen.utils.qa_utils import create_options, create_multiple_choice, enumerate_frame_labels, replace_substrs


def grounding_ablations(frame_path, perspective, verbose, id2label_path: str = None, configs: list = None):
    identifier = os.path.basename(frame_path)
    label2id, invalid_ids = enumerate_frame_labels(frame_path, perspective, id2label_path)
    labels = list(label2id.keys())
    non_ego_labels = [label for label in labels if label != -1]
    records = {}
    new_id2l = {}
    if len(non_ego_labels) <= 0:
        print("Too few objects for grounding")
    else:
        selected_label = random.choice(non_ego_labels)
        other_labels = [label for label in non_ego_labels if label != selected_label]
        new_random_labels = list(np.random.choice(50, size=len(non_ego_labels), replace=False))
        new_selected_label = new_random_labels[0]
        new_id2l[label2id[selected_label]] = new_selected_label
        for idx, other_label in enumerate(other_labels):
            new_id2l[label2id[other_label]] = new_random_labels[idx + 1]
        multiple_choice_options = create_options(new_random_labels, 4, new_selected_label, list(range(50)))
        multiple_choice_options = [int(option) for option in multiple_choice_options]
        multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
        option2answer = {
            val: key for key, val in answer2label.items()
        }
        answer = answer2label[new_selected_label]
        question = f"What is the numerical label associated with the highlighted area? Choose the best answer from option (A) through (D): {multiple_choice_string}"
        explanation = ""
        for setting_id, config in enumerate(configs):
            print('setting_id')
            font_scale, background_color, form = config["font_scale"], config["background_color"], config["form"]
            solo_labeled_path = os.path.join(frame_path, f"setting_{setting_id}_{perspective}_{identifier}.png")
            grounding_labelframe(
                ground_id=label2id[selected_label],
                frame_path=frame_path, perspective="front", save_path=solo_labeled_path,
                query_ids=list(new_id2l.keys()), id2l=new_id2l,
                font_scale=font_scale, grounding=True, bounding_box=form=="box", masking=form=="mask", background_color=background_color
            )
            records[setting_id] = dict(
                question=question, answer=answer, explanation=explanation, type="grounding", objects=[],
                world=[frame_path], obs=[solo_labeled_path], options=option2answer
            )
    return records


def batch_generate_grounding_ablation(world_paths, save_path="./", verbose=False, perspective="front", labeled=False, proc_id=0, box=False, domain="sim"):
    frame_paths = [os.path.dirname(world_path) for world_path in world_paths]
    records = {}
    current_type = "grounding"
    current_frame = ""
    try:
        count = 0
        for frame_path in tqdm.tqdm(frame_paths, desc=f"Processing {proc_id}", unit="frame"):
            current_frame = frame_path
            identifier = os.path.basename(frame_path)
            static_id2label_path = os.path.join(frame_path, f"static_id2label_{perspective}_{identifier}.json")
            static_labeled_path = os.path.join(frame_path, f"static_labeled_{perspective}_{identifier}.png")
            if not labeled or not os.path.exists(static_labeled_path):
                if verbose:
                    print(f"Creating single-frame-consistent labelling for {frame_path}")
                result = static_id2label(frame_path, perspective)
                static_id2l = result  #json.load(open(static_id2label_path, "r"))
                if static_id2l is None:
                    print("why>>>???")
                assert static_id2l is not None
                labelframe(frame_path=frame_path, perspective=perspective, save_path=static_labeled_path,
                           id2l=static_id2l,
                           font_scale=1.25, bounding_box=box)
            else:
                if verbose:
                    print(f"Already have labelled version FOR {frame_path}")
                static_id2l = json.load(open(static_id2label_path, "r"))
                assert static_id2l is not None
            grounding_records = grounding_ablations(
                frame_path=frame_path, perspective=perspective, verbose=verbose,
                id2label_path=static_id2label_path, configs=SETTINGS)
            for g_id, record in grounding_records.items():
                records[g_id + count] = record
                records[g_id + count]["group"] = count
                records[g_id + count]["variant"] = g_id
                records[g_id + count]["domain"] = domain
            count += len(grounding_records)
    except Exception as e:
        print("Something Wrong! save partial results")
        print(f"Encountered issue at {current_frame},{current_type}")
        print(e)
        var = traceback.format_exc()
        debug_path = os.path.join(
            os.path.dirname(save_path),
            f"{proc_id}_debug.json"
        )
        json.dump(
            {"proc_id": proc_id, "end_frame": current_frame, "end_quetion": current_type, "error": str(e),
             "generated": len(records), "trace": str(var)},
            open(debug_path, "w"),
            indent=2
        )
        raise (e)
    finally:
        json.dump(records, open(save_path, "w"), indent=2)


def batch_generate_general_ablation(world_paths, save_path="./", verbose=False, perspective="front", labeled=False, proc_id=0, configs=None, domain="sim"):
    frame_paths = [os.path.dirname(world_path) for world_path in world_paths]
    static_templates = TEMPLATES["static"]
    records = {}
    current_type = ""
    current_frame = ""
    try:
        count = 0
        frame_paths = frame_paths
        for frame_path in tqdm.tqdm(frame_paths, desc=f"Processing {proc_id}", unit="frame"):
            current_frame = frame_path
            frame_records = {}
            frame_id = 0
            identifier = os.path.basename(frame_path)
            static_id2label_path = os.path.join(frame_path, f"static_id2label_{perspective}_{identifier}.json")
            if not labeled or not os.path.exists(static_labeled_path):
                if verbose:
                    print(f"Creating single-frame-consistent labelling for {frame_path}")
                result = static_id2label(frame_path, perspective)
                static_id2l = result  #json.load(open(static_id2label_path, "r"))
            else:
                if verbose:
                    print(f"Already have labelled version FOR {frame_path}")
                static_id2l = json.load(open(static_id2label_path, "r"))
            assert static_id2l is not None
            queried_ids = set()
            for question_type in tqdm.tqdm(static_templates.keys(), desc="Generating Questions", unit="type"):
                current_type = question_type
                if question_type not in ["describe_sector", "describe_distance"]:
                    question, answer, explanation, ids_of_interest, option2answer = generate(frame_path=frame_path,
                                                                                             question_type=question_type,
                                                                                             perspective=perspective,
                                                                                             verbose=verbose,
                                                                                             id2label_path=static_id2label_path)
                    if question is not None and answer is not None and explanation is not None:
                        frame_records[frame_id] = dict(
                            question=question, answer=answer, explanation=explanation,
                            type=question_type, objects=ids_of_interest, world=[frame_path],
                            obs=None, options=option2answer
                        )
                        frame_id += 1
                        for id in ids_of_interest:
                            queried_ids.add(id)
                else:
                    if question_type == "describe_sector":
                        params = [
                            {"<pos>": pos} for pos in ["lf", "rf", "f"]
                        ]
                    else:
                        params = [
                            {"<dist>": dist} for dist in ["very close", "close", "medium", "far"]
                        ]
                    for param in params:
                        question, answer, explanation, ids_of_interest, option2answer = parameterized_generate(
                            frame_path=frame_path,
                            question_type=question_type,
                            param=param,
                            perspective=perspective,
                            verbose=verbose,
                            id2label_path=static_id2label_path)
                        if question is not None and answer is not None and explanation is not None:
                            frame_records[frame_id] = dict(
                                question=question, answer=answer, explanation=explanation,
                                type=question_type, objects=ids_of_interest, world=[frame_path],
                                obs=None, options=option2answer
                            )
                            frame_id += 1
                        for id in ids_of_interest:
                            queried_ids.add(id)
            new_id2label = {object_id: i for i, object_id in enumerate(queried_ids)}
            original_id2label = static_id2l
            assert new_id2label is not None

            for qid, record in frame_records.items():
                if record["type"] not in ["identify_closest", "identify_leftmost", "identify_rightmost",
                                          "identify_frontmost", "identify_backmost", "describe_scenario",
                                          "describe_sector", "describe_distance"]:
                    old2new = {
                        original_id2label[object_id]: new_id2label[object_id] for object_id in record["objects"]
                    }
                    record["question"] = replace_substrs(record["question"], old2new)
                    record["explanation"] = replace_substrs(record["explanation"], old2new)
                    for opt in record["options"].keys():
                        record["options"][opt] = replace_substrs(record["options"][opt], old2new)



            for config_idx, config in enumerate(configs):
                print(config_idx)
                font_scale, background_color, form = config["font_scale"], config["background_color"], config["form"]
                static_labeled_path = os.path.join(frame_path, f"{config_idx}_static_labeled_{perspective}_{identifier}.png")
                labelframe(frame_path=frame_path, perspective=perspective, save_path=static_labeled_path,
                           id2l=static_id2l,
                           font_scale=font_scale, bounding_box=form == "box", masking=form == "mask",
                           background_color=background_color)
                new_labeled_path = os.path.join(frame_path, f"{config_idx}_static_qa_labeled_{perspective}_{identifier}.png")
                labelframe(
                    frame_path=frame_path, perspective="front", save_path=new_labeled_path,
                    query_ids=list(queried_ids), id2l=new_id2label, font_scale=font_scale, bounding_box=form == "box", masking=form == "mask",
                           background_color=background_color
                )
                for qid, record in frame_records.items():
                    copied = copy.deepcopy(record)
                    if copied["type"] not in ["identify_closest", "identify_leftmost", "identify_rightmost",
                                              "identify_frontmost", "identify_backmost", "describe_scenario",
                                              "describe_sector", "describe_distance"]:
                        copied["obs"] = [new_labeled_path]
                    else:
                        copied["obs"] = [static_labeled_path]
                    copied["variant"] = config_idx
                    copied["domain"] = domain
                    records[qid + count] = copied
                count += len(frame_records)

            grounding_records = grounding_ablations(frame_path=frame_path, perspective=perspective, verbose=verbose,
                                                   id2label_path=static_id2label_path, configs=configs)
            for g_id, record in grounding_records.items():
                records[g_id + count] = record
                records[g_id + count]["variant"] = g_id
                records[g_id + count]["domain"] = domain
            count += len(grounding_records)
    except Exception as e:
        print("Something Wrong! save partial results")
        print(f"Encountered issue at {current_frame},{current_type}")
        print(e)
        var = traceback.format_exc()
        debug_path = os.path.join(
            os.path.dirname(save_path),
            f"{proc_id}_debug.json"
        )
        json.dump(
            {"proc_id": proc_id, "end_frame": current_frame, "end_quetion": current_type, "error": str(e),
             "generated": len(records), "trace": str(var)},
            open(debug_path, "w"),
            indent=2
        )
        raise (e)
    finally:
        json.dump(records, open(save_path, "w"), indent=2)