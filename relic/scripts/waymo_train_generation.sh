#!/bin/bash
paths=(
    "/bigdata/datasets/scenarionet/waymo/training/training_0/"
)



for path in "${paths[@]}"; do
    if [ -d "$path" ]; then
        echo "Working on $path."
        python -m vqa.multiprocess_episodes_generation --headless --num_proc 32 \
        --scenarios --data_directory "$path" --config "./vqa/configs/waymo_scene.yaml" --source "Waymo" --split "train"
    else
        echo "File $path does not exist."
    fi
done