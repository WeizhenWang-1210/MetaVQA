export HF_HOME=/bigdata/chenda/huggingface_ckpt/
MODEL_PATH=OpenGVLab/InternVL2-4B
OUTPUT_FILE=/bigdata/weizhen/repo/qa_platform/public/data_verification_result.json
QUESTION_FILE=/bigdata/weizhen/repo/qa_platform/public/test/0_data_verification.json
CUDA_DEVICE=4,5,6,7

cd /bigdata/chenda/InternVL/internvl_chat/
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE /bigdata/chenda/miniconda3/envs/internvl2/bin/python chenda_scripts/zero_shot.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --output-file $OUTPUT_FILE