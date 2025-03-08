import json, os
def append_prefix(paths, prefix):
    return [os.path.join(prefix, p) for p in paths]

path = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/waymonusc.json"
val_path = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/waymonusc_processed.json"
base_dir = os.path.dirname(path)
qas = json.load(open(path, "r"))
for key, val in qas.items():
    qas[key]["obs"] = append_prefix(qas[key]["obs"], base_dir)
print(qas["0"]["obs"])
json.dump(qas, open(val_path, "w"), indent=2)

