#!/bin/bash

# 1 nusc real scenario annotations.
# This is the root directory at which the generated scenarios are saved.
SAVE_DIR="/bigdata/weizhen/metavqa_release/scenarios/nusc_real"
START_IDX=400
END_IDX=401
# Number of independent processed to use. This program is trivially parallelizable as no inter-process communication is needed.
# The number of processes should be equal to the number of CPU cores on your machine.
# The scenarios are saved independently, so you can run this script multiple times with different start and end indices.
NUM_PROC=1
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
cd $METAVQA_DIR;
python -m vqa.nusc_devkit_annotation --start $START_IDX --end $END_IDX --num_proc $NUM_PROC --store_dir $SAVE_DIR

# # 2 nusc sim scenario annotations.
# # SAVE_DIR is specified in config.
# DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# METAVQA_DIR="$(dirname "$DIR")"
# CONFIG="${METAVQA_DIR}/scripts/scenegen_config/nusc_sim.yaml"
# START_IDX=400
# END_IDX=401
# NUM_PROC=1
# cd $METAVQA_DIR;
# python -m vqa.find_nuscenes_observation --headless --num_proc $NUM_PROC --config $CONFIG --start $START_IDX --end $END_IDX


# 3 waymo sim scenario annotations.
# paths=(
#     "/bigdata/datasets/scenarionet/waymo/training/training_1/"
# )
# DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# METAVQA_DIR="$(dirname "$DIR")"
# CONFIG="${METAVQA_DIR}/scripts/scenegen_config/waymo_sim.yaml"
# SETTING="NORMAL"
# START_IDX=0
# END_IDX=1
# NUM_PROC=1
# export SETTING
# 
# cd $METAVQA_DIR;
# for path in "${paths[@]}"; do
#     if [ -d "$path" ]; then
#         echo "Working on $path."
#         python -m vqa.multiprocess_episodes_generation --headless --num_proc $NUM_PROC \
#         --scenarios --data_directory "$path" --config $CONFIG --source "waymo" --split ""\
#         --start $START_IDX --end $END_IDX
#     else
#         echo "File $path does not exist."
#     fi
# done

