#!/bin/bash
NUMSCENARIOS=50
DATA="/home/weizhen/cat"
MODELPATH="Qwen/Qwen2-VL-7B-Instruct"
RECORDPATH="/home/weizhen/closed_loops/qwen2/"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/qwen2/qwen2_direct_zeroshot.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=2

cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed_loop_evaluations --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH