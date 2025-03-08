#!/bin/bash
NUMSCENARIOS=10
DATA="/data_weizhen/small_scenarios/"
MODELPATH="/home/chenda/ckpt/internvl_finetuned_nusc_merge"   #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/waymonusc/"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/waymonusc/result.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=2
export INTERNVL=true
#export INTERNVLZEROSHOT=true
cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH
