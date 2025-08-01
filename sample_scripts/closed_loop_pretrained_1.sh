#!/bin/bash
NUMSCENARIOS=1

MODELPATH="Weizhen011210/InternVL2-8B_MetaVQA-Closed-Loop"   #Where the ckpt is stored
RECORDPATH="/home/weizhen/test_close" # Where you store the closed-loop results and visualizations
PROMPTSCHEMA="direct"
RESULTPATH="$RECORDPATH/result.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"

DATA="$METAVQA_DIR/closed_loop/assets/scenarios" # Where the test scenarios are located
CUDA_DEVICES=0
export INTERNVL=true
cd $METAVQA_DIR;

# Here, we demonstrate using the CVPR version of closed-loop code. InternVL2 demands this different setup.
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m closed_loop.closed_loop_cvpr --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH