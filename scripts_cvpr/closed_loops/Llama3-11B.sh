#!/bin/bash
NUMSCENARIOS=120
DATA="/data_weizhen/scenarios"
MODELPATH="meta-llama/Llama-3.2-11B-Vision-Instruct"#Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/llama_zeroshot/"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/llama_zeroshot/llama_zeroshot.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=6
cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH