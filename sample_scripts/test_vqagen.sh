#!/bin/bash
# (1) Sample code for generating VQA on the nuScenes real scenarios
SCENARIOS="/bigdata/weizhen/releases/MetaVQA_migrated/scenarios/nusc_real"
SAVEPATH="/bigdata/weizhen/releases/MetaVQA_migrated/vqas/nusc_real.json"
NUMPROC=16
DOMAIN="real"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
cd $METAVQA_DIR;
python -m som.static_question_generation --scenarios $SCENARIOS \
  --save_path $SAVEPATH --num_proc $NUMPROC \
  --nusc_real --domain $DOMAIN

# (2) Sample code for generating VQA on the WAYMO/nuScenes sim scenarios
#!/bin/bash
SCENARIOS="/bigdata/weizhen/releases/MetaVQA_migrated/scenarios/waymo_sim"
SAVEPATH="/bigdata/weizhen/releases/MetaVQA_migrated/vqas/waymo_sim.json"
NUMPROC=16
DOMAIN="sim"
DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
METAVQA_DIR="$(dirname "$DIR")"
cd $METAVQA_DIR;
python -m closed_loop.static_question_generation --scenarios $SCENARIOS \
  --save_path $SAVEPATH --num_proc $NUMPROC \
  --domain $DOMAIN