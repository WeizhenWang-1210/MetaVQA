#!/bin/bash
NUMSCENARIOS=50
DATA="/home/weizhen/cat"
MODELPATH="random"   #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/random"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/random/random.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=0

cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed_loop_evaluations --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH