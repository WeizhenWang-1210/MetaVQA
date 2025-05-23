#!/bin/bash
NUMSCENARIOS=120
DATA="/data_weizhen/scenarios"
MODELPATH="/home/chenda/huggingface_ckpt/InternVL2-8B"   #Where the ckpt is stored
RECORDPATH="/home/weizhen/hope/8b_zeroshot"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/hope/8b_zeroshot/result.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=4
export INTERNVL=true
#export INTERNVLZEROSHOT=true
cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH