#!/bin/bash
paths=(
    "/bigdata/datasets/scenarionet/nuscenes/trainval/sc_nusc_trainval_7"
)



for path in "${paths[@]}"; do
    if [ -d "$path" ]; then
        echo "Working on $path."
        python -m vqa.multiprocess_episodes_generation --headless --num_proc 16 \
        --scenarios --data_directory "$path" --source "NuScenes" --split "validation"
    else
        echo "File $path does not exist."
    fi
done