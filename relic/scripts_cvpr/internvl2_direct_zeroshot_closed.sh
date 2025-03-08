#!/bin/bash
NUMSCENARIOS=1
DATA="/data_weizhen/scenarios"
MODELPATH="/home/chenda/huggingface_ckpt/InternVL2-8B"   #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/internvl2_zeroshot"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/debug/debug.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=0
export INTERNVL=true
#export INTERNVLZEROSHOT=true
cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed_loop_evaluations --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH