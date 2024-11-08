#!/bin/bash
NUMSCENARIOS=120
DATA="/data_weizhen/scenarios"
MODELPATH="/home/chenda/ckpt/internvl_finetuned_waymonusc_merge"   #Where the ckpt is stored
RECORDPATH="/home/weizhen/closed_loops/internvl2_finetuned_som"
PROMPTSCHEMA="direct"
RESULTPATH="/home/weizhen/closed_loops/internvl2_finetuned_som/internvl2_direct_finetuned.json"
#DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
#METAVQA_DIR="$(dirname "$DIR")"
CUDA_DEVICES=2
export INTERNVL=true

cd $METAVQA_DIR;
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m som.closed_loop_evaluations --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH