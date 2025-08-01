#!/bin/bash
NUMSCENARIOS=1
MODELPATH="Weizhen011210/Llama3.2_MetaVQA-Closed-Loop" #Where the ckpt is stored
RECORDPATH="/home/weizhen/test_close" # Where you store the closed-loop results and visualizations
PROMPTSCHEMA="direct"
RESULTPATH="$RECORDPATH/result.json"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"

DATA="$METAVQA_DIR/closed_loop/assets/scenarios" # Where the test scenarios are located
CUDA_DEVICES=0
export INTERNVL=true
cd $METAVQA_DIR;

# You should use this script for benchmarking your own models.
# Here, we demonstrate using the cleaned-up version of closed-loop code. Qwen and Llama3.2 are compatible with this setup.
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m closed_loop.closed_loop_benchmark --headless --num_scenarios $NUMSCENARIOS --data_directory $DATA --model_path $MODELPATH \
  --prompt_schema $PROMPTSCHEMA --record_path $RECORDPATH --result_path $RESULTPATH