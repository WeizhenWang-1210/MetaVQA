#!/bin/bash
START=0
END=40
DATA="/data_weizhen/scenarios"
MODELPATH="/home/chenda/ckpt/internvl_finetuned_waymonusc_merge"   #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/CoT"
PROMPTSCHEMA="CoT"
RESULTPATH="/home/weizhen/closed_loops/CoT/0.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=0
export INTERNVL=true

cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.CoT --headless --start $START --end $END --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH