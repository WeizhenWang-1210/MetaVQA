#!/bin/bash
SCENARIOS="/bigdata/weizhen/metavqa_cvpr/scenarios/nusc_sim" # Replace with your scenarios root dir.
SAVEPATH="/bigdata/weizhen/metavqa_cvpr/vqas/nusc_sim/nusc_sim.json" # Replace with you desired VQA save path
NUMPROC=32
DOMAIN="sim"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
cd $METAVQA_DIR;
python -m som.static_question_generation --scenarios $SCENARIOS \
  --save_path $SAVEPATH --num_proc $NUMPROC \
  --domain $DOMAIN