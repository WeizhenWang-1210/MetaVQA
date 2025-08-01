#!/bin/bash
NUMSCENARIOS=120
DATA="/data_weizhen/scenarios" # Where the test scenarios are located
MODELPATH="meta-llama/Llama-3.2-11B-Vision-Instruct" #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/llama_zeroshot/" # Where you store the closed-loop results and visualizations
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/llama_zeroshot/llama_zeroshot.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=6
cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed_loop_cvpr --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH