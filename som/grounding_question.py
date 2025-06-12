import itertools
import json
import os
import random

import numpy as np
import tqdm

from som.qa_utils import create_options, create_multiple_choice
from som.utils import enumerate_frame_labels
from vqa.vqagen.set_of_marks import grounding_labelframe

TEXTSCALES=(0.75,1.25)
FORMS=("box", "mask", "contour")
BACKGROUND=((255,255,255), (0,0,0))

SETTINGS = [
    dict(font_scale=textscale, form=form, background_color=background) for (textscale, form, background) in itertools.product(TEXTSCALES, FORMS, BACKGROUND)
]



def generate_grounding(frame_path, perspective, verbose, id2label_path: str = None, box: bool = False, font_scale:float=0.75, domain="sim"):
    identifier = os.path.basename(frame_path)
    world_path = os.path.join(frame_path, "world_{}.json".format(identifier))
    world = json.load(open(world_path, "r"))
    label2id, invalid_ids = enumerate_frame_labels(frame_path, perspective, id2label_path)
    labels = list(label2id.keys())
    non_ego_labels = [label for label in labels if label != -1]
    records = {}
    if domain == "real":
        id2corners = json.load(open(os.path.join(frame_path,f"id2corners_{identifier}.json"),"r"))
    else:
        id2corners = None
    if len(non_ego_labels) <= 0:
        print("Too few objects for grounding")
    else:
        randomly_selected_labels = random.sample(non_ego_labels, min(len(non_ego_labels), 2))
        idx = 0
        for selected_label in tqdm.tqdm(randomly_selected_labels, desc="Generating Grounding Question", unit="question"):
            other_labels = [label for label in non_ego_labels if label != selected_label]
            solo_labeled_path = os.path.join(frame_path, f"label_{selected_label}_{perspective}_{identifier}.png")
            random_new_labels = list(np.random.choice(50, size=len(non_ego_labels), replace=False))
            random_new_labels = [int(item) for item in random_new_labels]
            assert all([isinstance(label, int) for label in random_new_labels])
            random_new_label = random_new_labels[0]

            new_id2l = {label2id[selected_label]: random_new_label}

            for s_idx, other_label in enumerate(other_labels):
                assert isinstance(random_new_labels[s_idx+1],int)
                new_id2l[label2id[other_label]] = random_new_labels[s_idx+1]
            assert new_id2l is not None
            grounding_labelframe(
                ground_id=label2id[selected_label],
                frame_path=frame_path, perspective="front", save_path=solo_labeled_path,
                query_ids=list(new_id2l.keys()), id2l=new_id2l, id2corners=id2corners,
                font_scale=font_scale, grounding=True, bounding_box=box, background_color=(0,0,0)
            )
            multiple_choice_options = create_options(random_new_labels, 4, random_new_label, list(range(50)))

            multiple_choice_options = [int(option) for option in multiple_choice_options]

            multiple_choice_string, answer2label = create_multiple_choice(multiple_choice_options)
            option2answer = {
                val: key for key, val in answer2label.items()
            }
            answer = answer2label[random_new_label]
            question = f"What is the numerical label associated with the highlighted area? Choose the best answer from option (A) through (D): {multiple_choice_string}"
            explanation = ""
            records[idx] = dict(
                question=question, answer=answer, explanation=explanation, type="grounding", objects=[],
                world=[frame_path], obs=[solo_labeled_path], options=option2answer
            )
            idx += 1
    return records



def grounding_ablations(frame_path, perspective, verbose, id2label_path: str = None, configs:list=None):
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




if __name__ == "__main__":
    frame_path = "/bigdata/weizhen/metavqa_iclr/scenarios/nusc_real/scene-0596_0_40/26_17"

