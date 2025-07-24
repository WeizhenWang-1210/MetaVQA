#!/bin/bash
# (1) Sample code for generating VQA on the nuScenes real scenarios
SCENARIOS="/bigdata/weizhen/metavqa_release/scenarios/nusc_real"
SAVEPATH="/bigdata/weizhen/metavqa_release/vqas/nusc_real/nusc_real.json"
NUMPROC=1
DOMAIN="real"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
cd $METAVQA_DIR;
python -m som.static_question_generation --scenarios $SCENARIOS \
  --save_path $SAVEPATH --num_proc $NUMPROC \
  --nusc_real --domain $DOMAIN

# (2) Sample code for generating VQA on the WAYMO/nuScenes sim scenarios
#!/bin/bash
# SCENARIOS="/bigdata/weizhen/metavqa_release/scenarios/waymo_sim"
# SAVEPATH="/bigdata/weizhen/metavqa_release/vqas/waymo_sim/waymo_sim.json"
# NUMPROC=1
# DOMAIN="sim"
# DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# METAVQA_DIR="$(dirname "$DIR")"
# cd $METAVQA_DIR;
# python -m closed_loop.static_question_generation --scenarios $SCENARIOS \
#   --save_path $SAVEPATH --num_proc $NUMPROC \
#   --domain $DOMAIN