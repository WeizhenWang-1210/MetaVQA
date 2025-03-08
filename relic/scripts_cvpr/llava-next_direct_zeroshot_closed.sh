#!/bin/bash
NUMSCENARIOS=50
DATA="/home/weizhen/cat"
MODELPATH="llava-hf/llava-v1.6-vicuna-7b-hf"
RECORDPATH="/home/weizhen/closed_loops/llava-next"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/llava-next/llava-next_direct_zeroshot.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=2

cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed_loop_evaluations --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH