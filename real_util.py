import json, os

path = "/data_weizhen/metavqa_cvpr/datasets/test/test/test_real.json"
val_path = "/data_weizhen/metavqa_cvpr/datasets/test/test/test_real_processed.json"

def append_prefix(paths, prefix):
    return [os.path.join(prefix, p) for p in paths]


path = "/data_weizhen/metavqa_cvpr/datasets/test/test/test.json"
val_path = "/data_weizhen/metavqa_cvpr/datasets/test/test/test_processed.json"
base_dir = os.path.dirname(path)
qas = json.load(open(path, "r"))
for key, val in qas.items():
    qas[key]["obs"] = append_prefix(qas[key]["obs"], base_dir)
print(qas["0"]["obs"])
json.dump(qas, open(val_path, "w"), indent=2)

