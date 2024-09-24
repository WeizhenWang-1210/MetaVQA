#!/bin/bash
paths=(
    "/bigdata/yuxin/cat_reconstructed/train/subdir_4"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_7"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_11"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_12"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_13"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_14"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_15"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_17"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_18"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_19"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_20"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_21"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_22"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_23"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_24"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_25"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_26"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_27"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_28"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_29"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_30"
)



for path in "${paths[@]}"; do
    if [ -d "$path" ]; then
        echo "Working on $path."
        python -m vqa.multiprocess_episodes_generation --headless --num_proc 1 \
        --scenarios --data_directory "$path" --source "Waymo_CAT" --split "train" --config "./vqa/configs/cat_scene.yaml"

    else
        echo "File $path does not exist."
    fi
done