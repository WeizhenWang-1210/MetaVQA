#!/bin/bash

python -m vqa.multiprocess_question_generation --num_proc 32 --src "Waymo_CAT" --root_directory \
"/bigdata/weizhen/metavqa_final/scenarios/testing/safety_critical" --output_base \
"/bigdata/weizhen/metavqa_final/vqa/testing/safety_critical/collision"