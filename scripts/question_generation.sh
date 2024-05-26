#!/bin/bash

python -m vqa.multiprocess_question_generation --num_proc 32 --src "NuScenes" --root_directory \
"/bigdata/weizhen/metavqa_final/scenarios/testing/normal/sc_nusc_trainval_6" --output_base \
"/bigdata/weizhen/metavqa_final/vqa/testing/single_frame/NuScenes/"