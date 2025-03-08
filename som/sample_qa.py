import json, os
import random


path1 = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/37500_trainval_processed.json"
path2 = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/9375_trainval_processed.json"
qa1 = json.load(open(path1,"r"))
new_idx = 0
final = dict()
keys = list(qa1.keys())
selected_keys = random.sample(keys, k=9375)
for key in selected_keys:
    final[new_idx] = qa1[key]
    new_idx += 1
print(len(final))
json.dump(final, open(path2, "w"), indent=1)

