#!/bin/bash
SAVE_DIR="/bigdata/weizhen/metavqa_cvpr/scenarios/nusc_sim"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CONFIG="${METAVQA_DIR}/som/configs/nusc_sim.yaml"
START_IDX=0
END_IDX=400
NUM_PROC=8
cd $METAVQA_DIR;
python -m vqa.find_nuscenes_observation --headless --num_proc $NUM_PROC --config $CONFIG --start $START_IDX --end $END_IDX

