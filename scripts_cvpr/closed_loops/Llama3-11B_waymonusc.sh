#!/bin/bash
NUMSCENARIOS=120
DATA="/data_weizhen/scenarios"
MODELPATH="/data_weizhen/chenda_data/LLaMA-Factory/models/llama-3.2-11B-Vision-Instruct_lora_sft_waymonusc"  #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/llama_waymonusc/"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/llama_waymonusc/llama_waymonusc.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=6
cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH