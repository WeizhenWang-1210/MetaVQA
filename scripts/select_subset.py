import json
import os
from pprint import pprint
import copy
IMMAX = 4000

vqa_trainval = json.load(open("/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/trainval_processed.json", "r"))
vqa_selected = {}
new_qid = 0
for qid, vqa in vqa_trainval.items():
    abs_image_path = vqa["obs"][-1]
    image_idx = int(os.path.basename(abs_image_path).split(".")[0])
    if image_idx < 4000:
        print("Imaged being used:", abs_image_path)
        vqa_selected[new_qid] = copy.deepcopy(vqa)
        vqa_selected[new_qid]["old_qid"]=qid
        new_qid += 1
print(new_qid)
pprint(vqa_selected[344])

save_path = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/rebuttal_processed.json"
json.dump(
    vqa_selected,
    open(save_path, "w")
)
