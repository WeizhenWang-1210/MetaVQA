#!/bin/bash
NUMSCENARIOS=50
DATA="/home/weizhen/cat"
MODELPATH="llava-hf/llava-onevision-qwen2-7b-ov-hf"
RECORDPATH="/home/weizhen/closed_loops/llava-onevision"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/llava-onevision/llava-onevision_direct_zeroshot.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=1

cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed_loop_evaluations --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH