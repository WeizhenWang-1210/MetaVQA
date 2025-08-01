#!/bin/bash
# 400 nusc real scenario annotations.
SAVE_DIR="/bigdata/weizhen/metavqa_cvpr/scenarios/nusc_real" # Replace with you desired scenario save path
START_IDX=0
END_IDX=400 # Sequentially load the first 400 scenarios.
NUM_PROC=16 # Number of processes to use for parallel scenario aggregation.
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
cd $METAVQA_DIR;
python -m vqa.scenegen.nusc_devkit_annotation --start $START_IDX --end $END_IDX --num_proc $NUM_PROC --store_dir $SAVE_DIR
