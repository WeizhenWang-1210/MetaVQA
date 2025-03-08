#!/bin/bash

python -m vqa.multiprocess_question_generation --num_proc 16 --src "NuScenes" --root_directory \
"/bigdata/weizhen/metavqa_final/scenarios/NuScenes_Mixed" --output_base \
"/bigdata/weizhen/metavqa_final/vqa/NuScenes_Mixed/single_frame";