import json
import random

random.seed(0)

load_path = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/gpt_annotated_processed.json"
save_path = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/gpt_rebuttal_processed.json"
processed_responses = json.load(open(load_path, "r"))

print(len(processed_responses))

SELECT = 5423

candidates = {}


for qid, vqa in processed_responses.items():
    assert "obs" in vqa.keys()
    assert "domain" in vqa.keys()
    assert "world" in vqa.keys()
    assert "question" in vqa.keys()
    assert "answer" in vqa.keys()
    if not (len(vqa['answer']) == 1 and vqa['answer'].isalpha() and vqa["answer"].isupper() and vqa['answer'] in vqa[
        'question']):
        continue
    candidates[qid] = vqa


ids = list(candidates.keys())
selected_ids = random.sample(ids, SELECT)

final_vqa = {}
final_id = 0

for idx in selected_ids:
    final_vqa[final_id] = candidates[idx]
    final_id += 1

json.dump(
    final_vqa,
    open(save_path,"w"),
    indent=2
)