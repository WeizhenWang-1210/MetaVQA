#!/bin/bash
NUMSCENARIOS=120
DATA="/data_weizhen/scenarios"
MODELPATH="llava-hf/llava-onevision-qwen2-7b-ov-hf"    #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/llava-onevision_zeroshot"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/llava-onevision_zeroshot/llava-onevisio_zeroshot.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=4
cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH