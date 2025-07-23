import json, os

path = "/data_weizhen/metavqa_cvpr/static_medium_export/data.json"
base_dir = os.path.dirname(path)
split_path = "/data_weizhen/metavqa_cvpr/vqa_merged/static_medium_split.json"
qas = json.load(open(path, "r"))
split = json.load(open(split_path, "r"))
print(len(split["train"]))
print(len(split["val"]))
train_path = "/data_weizhen/metavqa_cvpr/static_medium_export/train.json"
val_path = "/data_weizhen/metavqa_cvpr/static_medium_export/val.json"

train_qas, val_qas = dict(), dict()
local_idx = 0


def append_prefix(paths, prefix):
    return [os.path.join(prefix, p) for p in paths]


for idx in split["train"]:
    train_qas[local_idx] = qas[idx]
    train_qas[local_idx]["obs"] = append_prefix(qas[idx]["obs"], base_dir)
    local_idx += 1
local_idx = 0
for idx in split["val"]:
    val_qas[local_idx] = qas[idx]
    val_qas[local_idx]["obs"] = append_prefix(qas[idx]["obs"], base_dir)
    local_idx += 1

print(train_qas[0]["obs"])

json.dump(train_qas, open(train_path, "w"), indent=2)
json.dump(val_qas, open(val_path, "w"), indent=2)
