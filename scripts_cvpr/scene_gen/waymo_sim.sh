#!/bin/bash
paths=(
    "/bigdata/datasets/scenarionet/waymo/training/training_0/" # Replace with your preprocessed Waymo pkl files
)
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CONFIG="${METAVQA_DIR%/}/vqa/scenegen/configs/waymo_sim.yaml"
SETTING="NORMAL"
START_IDX=0
END_IDX=7000
NUM_PROC=16
export SETTING

cd $METAVQA_DIR;
for path in "${paths[@]}"; do
    if [ -d "$path" ]; then
        echo "Working on $path."
        python -m vqa.scenegen.multiprocess_metadrive_annotation --headless --num_proc $NUM_PROC \
        --scenarios --data_directory "$path" --config $CONFIG --source "waymo" --split ""\
        --start $START_IDX --end $END_IDX
    else
        echo "File $path does not exist."
    fi
done