#!/bin/bash
# 400 nusc real scenario annotations.
SAVE_DIR="/bigdata/weizhen/metavqa_cvpr/scenarios/nusc_real"
START_IDX=0
END_IDX=400
NUM_PROC=16
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
cd $METAVQA_DIR;
python -m vqa.nusc_devkit_annotation --start $START_IDX --end $END_IDX --num_proc $NUM_PROC --store_dir $SAVE_DIR
