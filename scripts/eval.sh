export HF_HOME=/bigdata/chenda/huggingface_ckpt/
MODEL_PATH=OpenGVLab/InternVL2-4B
OUTPUT_FILE=/bigdata/chenda/InternVL/internvl_chat/chenda_output/grounding_result.json
QUESTION_FILE=/bigdata/weizhen/metavqa_iclr/vqa/grounding.json
CUDA_DEVICE=0,1,3

cd /bigdata/chenda/InternVL/internvl_chat/
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE /bigdata/chenda/miniconda3/envs/internvl2/bin/python chenda_scripts/zero_shot.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --output-file $OUTPUT_FILE