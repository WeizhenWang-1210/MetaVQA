import json, os
def append_prefix(paths, prefix):
    return [os.path.join(prefix, p) for p in paths]

path1 = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/trainval_processed.json"
path2 = "/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/filtered.json"
merged_path = "/data_weizhen/metavqa_cvpr/datasets/trainval/driving/gts/drive_trainval.json"


qa1 = json.load(open(path1,"r"))
qa2 = json.load(open(path2,"r"))


new_idx = 0


final = dict()

for key, val in qa1.items():
    final[new_idx] = val
    new_idx += 1
for key, val in qa2.items():
    final[new_idx] = val
    new_idx += 1
print(len(final))
json.dump(final, open(merged_path, "w"), indent=1)

