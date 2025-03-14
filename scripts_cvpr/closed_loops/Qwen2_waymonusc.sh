#!/bin/bash
NUMSCENARIOS=120
DATA="/data_weizhen/scenarios"
MODELPATH="/data_weizhen/chenda_data/LLaMA-Factory/models/qwen2_vl_lora_sft_waymonusc" #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/qwen2_waymonusc/"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/qwen2_waymonusc/qwen2_waymonusc.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=0
cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH