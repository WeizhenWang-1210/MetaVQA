#!/bin/bash

python -m vqa.multiprocess_question_generation --job static --config "/home/weizhen/MetaVQA/vqa/configs/data_gen/demo_static.yaml";
python -m vqa.multiprocess_question_generation --job dynamic --config "/home/weizhen/MetaVQA/vqa/configs/data_gen/demo_dynamic.yaml";
python -m vqa.multiprocess_question_generation --job safety --config "/home/weizhen/MetaVQA/vqa/configs/data_gen/demo_safety.yaml";

