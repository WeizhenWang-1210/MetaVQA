from omegaconf import OmegaConf, DictConfig
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json
import argparse
import os

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.models import *
from PIL import Image
import torch
device = "cuda"
model_config_path = "/bigdata/chenda/ELM/lavis/projects/blip2/train/eval_metavqa_multiview_mixmultiframe_test.yaml"
model_path = "/bigdata/chenda/output/metavqa_multiview_mixmultiframe_critical_test/20240529222/checkpoint_3.pth"
def load_ckpt(model, file_path):
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path, map_location=device)
    else:
        raise RuntimeError("checkpoint url or path is invalid")
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict=True)

    print("Resume checkpoint from {}".format(file_path))
    return model
def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", help="path to configuration file.", default=model_config_path)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # replace some settings in the used config
    parser.add_argument("--replace_cfg", nargs="+", help="replace some settings in the used config", default=None)
    parser.add_argument("--job_id", default=None, help="job id")
    args = parser.parse_args()

    return args
def load_model():
    args = parse_args()
    cfg = Config(args)
    task = tasks.setup_task(cfg)
    print("Initialize the model...")
    model = task.build_model(cfg)
    print("Load Checkpoint...")
    model = load_ckpt(model, model_path)
    print('Success!')
    # vis_config =  cfg.get("vis_processor").get("eval")
    # text_config = cfg.get("text_processor").get("eval")
    cfg: DictConfig = OmegaConf.create({"name": "blip_image_eval", "image_size": 364})
    vis_config = cfg
    cfg2: DictConfig = OmegaConf.create({"name": "blip_question"})
    text_config = cfg2
    vis_processors = registry.get_processor_class(vis_config.name).from_config(vis_config)
    text_processors = registry.get_processor_class(text_config.name).from_config(text_config)
    return model, vis_processors, text_processors
def eval_model(model, samples, vis_processors, text_processors):
    model.eval()
    # samples:
                    #(batch, time, view, channel, size, size)
    # vfeats: tensor, 8 5 6 3 364 364
    # questions: list of 8 str
    # answers: list of 8 str
    answers = model.predict_answers(
        samples=samples,
        answer_list=None,
        inference_method="generate",
        num_beams=5,
        max_len=60,
        min_len=1,
        num_ans_candidates=128,
        prompt="",
    )
    print("predict finished")
    return answers
def process_image_path(annotation_val):
    image_path = []
    for view in ["front", "leftf", "leftb", "rightf", "rightb", "back"]:
        if view in annotation_val['rgb']:
            curr_view_path = []
            for path in annotation_val['rgb'][view]:
                curr_view_path.append(path)
            image_path.append(curr_view_path)
        else:
            raise ValueError(f"The view {view} is not in the annotation")
    # we should have equal number of images for each view
    for i in range(1, len(image_path)):
        if len(image_path[i]) != len(image_path[0]):
            raise ValueError(f"The number of images in each view should be the same")
    return image_path
def load_and_process_images(images_paths, vis_processor):
        # image paths: 6x20(view x time) list of paths
        # return: 5x6x3x364x364 tensor (5 is the select time)
        # Define the views
        views = ["front", "leftf", "leftb", "rightf", "rightb", "back"]

        # Initialize a list to store processed images for each view
        all_views_images = [[] for _ in views]
        select_index = [0, 5, 10, 15, 19]
        # Iterate through the image paths
        for i, view_paths in enumerate(images_paths):
            if len(view_paths) == 1:
                curr_select_index = [0]
            else:
                curr_select_index = select_index
            for view_index in curr_select_index:
            # for view_index, view_path in enumerate(view_paths):
                view_path = view_paths[view_index]
                # Get the full path of the image
                image_full_path = os.path.join(view_path.replace('./', '').replace('\\', '/'))
                # Load and process the image
                image = Image.open(image_full_path).convert("RGB")
                processed_image = vis_processor(image).to(device)
                all_views_images[i].append(processed_image)

        # Convert the lists of images into NumPy arrays
        all_views_images = [torch.stack(view_images, dim=0) for view_images in all_views_images]

        # Concatenate the processed images into the desired shape (t, views, height, width, channel)
        processed_images_tensor = torch.stack(all_views_images, dim=1)
        t, views, height, width, channel = processed_images_tensor.shape
        # If t is 1, repeat the tensor 20 times along the first dimension
        if t == 1:
            processed_images_tensor = processed_images_tensor.repeat(5, 1, 1, 1, 1)

        return processed_images_tensor
def load_example(vis_processors, text_processors):
    annotations = json.load(
        open("/bigdata/weizhen/metavqa_final/vqa/validation/multi_frame_processed/Waymo/dynamic_qa0.json", "r"))
    annotation_keys = list(annotations.keys())
    id = annotation_keys[5]
    val = annotations[id]
    image_path = process_image_path(val) # 6 x 20 list, 6 is view, 20 is timeframe
    print(image_path)
    print(len(image_path)) # 6
    print(len(image_path[0])) # 20
    image = load_and_process_images(image_path, vis_processors) # 5x6x3x364x364 tensor (5 is the select time)
    question = val['question']
    question = "You are the driver, what is the safest action to do? Anwser in left|right|stop."
    question = text_processors(question)
    answer = val['answer']

    samples = [{
            "question": question,
            "answer": answer,
            "image": image,
        }]
    # merge samples into a list for each key
    questions = [s["question"] for s in samples]
    answers = [s["answer"] for s in samples]
    images = [s["image"] for s in samples]
    images = torch.stack(images, dim=0) #1,5x6x3x364x364
    answers = [item[0] for item in answers]

    return {
        "vfeats": images,
        "questions": questions,
        "answers": answers,
    }
def demo():
    model, vis_processors, text_processors = load_model()
    model.to(device)
    samples = load_example(vis_processors, text_processors)
    print("Question:")
    print(samples['questions'])
    print("Answers:")
    print(samples["answers"])
    print("Predict: ")
    print(eval_model(model, samples, vis_processors, text_processors))
if __name__ == "__main__":
    demo()
