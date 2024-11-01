#!/bin/bash
SCENARIOS="/bigdata/weizhen/metavqa_cvpr/scenarios/nusc_sim"
SAVEPATH="/bigdata/weizhen/metavqa_cvpr/vqas/scratch/nusc_sim.json"
NUMPROC=1
DOMAIN="sim"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
cd $METAVQA_DIR;
python -m som.static_question_generation --scenarios $SCENARIOS \
  --save_path $SAVEPATH --num_proc $NUMPROC \
  --verbose --domain $DOMAIN