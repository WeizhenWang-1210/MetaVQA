#!/bin/bash
paths=(
    "/bigdata/datasets/scenarionet/nuscenes/test"
)



for path in "${paths[@]}"; do
    if [ -d "$path" ]; then
        echo "Working on $path."
        python -m vqa.multiprocess_episodes_generation --headless --num_proc 32 \
        --scenarios --data_directory "$path" --source "NuScenes" --split "test"
    else
        echo "File $path does not exist."
    fi
done