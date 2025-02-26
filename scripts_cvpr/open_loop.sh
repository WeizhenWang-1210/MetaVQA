#!/bin/bash
NUMSCENARIOS=120
DATA="/data_weizhen/scenarios"
MODELPATH="/home/chenda/ckpt/internvl_drive_trainval_merge"   #Where the ckpt is stored
RECORDPATH="/home/weizhen/hope/finetuned"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/hope/finetuned/result.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=1
export INTERNVL=true
#export INTERNVLZEROSHOT=true
cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH