#!/bin/bash
paths=(
    "/bigdata/yuxin/cat_reconstructed/train/subdir_21"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_22/"
    "/bigdata/yuxin/cat_reconstructed/train/subdir_23/"
    #"/bigdata/yuxin/cat_reconstructed/train/subdir_24/",
    #"/bigdata/yuxin/cat_reconstructed/train/subdir_25/",
    #"/bigdata/yuxin/cat_reconstructed/train/subdir_26/",
    #"/bigdata/yuxin/cat_reconstructed/train/subdir_27/",
)


for path in "${paths[@]}"; do
    if [ -d "$path" ]; then
        echo "Working on $path$."
        python -m vqa.multiprocess_episodes_generation --headless --num_proc 32 \
        --scenarios --data_directory "$path" --source "Waymo_CAT" --split "train"
    else
        echo "File $path does not exist."
    fi
done