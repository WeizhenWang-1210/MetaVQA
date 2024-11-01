#!/bin/bash
NUMSCENARIOS=50
DATA="/home/weizhen/cat"
MODELPATH="/home/chenda/huggingface_ckpt/InternVL2-4B"   #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/internvl2_zeroshot"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/internvl2_zeroshot/internvl2_direct_zeroshot.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=4
export INTERNVL=true
export INTERNVLZEROSHOT=true

cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed_loop_evaluations --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH