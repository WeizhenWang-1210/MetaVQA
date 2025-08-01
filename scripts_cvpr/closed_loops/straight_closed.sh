#!/bin/bash
NUMSCENARIOS=120
DATA="/data_weizhen/scenarios" # Where the test scenarios are located
MODELPATH="always_straight"   #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/straight" # Where you store the closed-loop results and visualizations
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/straight/straight.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=0

cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed_loop_cvpr --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH