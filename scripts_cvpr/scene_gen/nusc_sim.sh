#!/bin/bash
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CONFIG="${METAVQA_DIR}/vqa/scenegen/configs/nusc_sim.yaml"
START_IDX=0
END_IDX=400 # Sequentially load the first 400 scenarios.
NUM_PROC=8 # Number of processes to use for parallel scenario aggregation.
cd $METAVQA_DIR;
python -m vqa.scenegen.nusc_metadrive_annotation --headless --num_proc $NUM_PROC --config $CONFIG --start $START_IDX --end $END_IDX

