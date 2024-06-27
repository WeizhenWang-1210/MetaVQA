#!/bin/bash

python -m vqa.multiprocess_question_generation --job static_nusc --config "/home/weizhen/MetaVQA/vqa/configs/data_gen/demo_static_nusc.yaml";
python -m vqa.multiprocess_question_generation --job dynamic_nusc --config "/home/weizhen/MetaVQA/vqa/configs/data_gen/demo_dynamic_nusc.yaml";
python -m vqa.multiprocess_question_generation --job safety_nusc --config "/home/weizhen/MetaVQA/vqa/configs/data_gen/demo_safety_nusc.yaml";