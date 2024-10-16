#!/bin/bash
#850 in total, then, 4 invocation of [0,200] [200,400] [400,600] [600 850], each with 8 process

CUDA_DEVICE=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m vqa.find_nuscenes_observation --headless --num_proc 8 --config "./vqa/configs/mixed_up_scene.yaml" --start 0 --end 200