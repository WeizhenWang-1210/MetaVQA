#!/bin/bash
#850 in total, then, 4 invocation of [0,200] [200,400] [400,600] [600 850], each with 8 process
export CUDA_VISIBLE_DEVICES=0,1;
python -m vqa.find_nuscenes_observation --headless --num_proc 1 --config "./vqa/configs/mixed_up_scene.yaml" --start 100 --end 200