import json
import re

raw_path = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/gpt_annotated.json"
save_path = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/gpt_annotated_processed.json"
raw_responses = json.load(open(raw_path, "r"))


def preprocess(raw_response):
    pattern = r"```(json|python)\n(.*?)```"
    matches = re.findall(pattern, raw_response, re.DOTALL)
    # Process and print the extracted content
    for language, content in matches:
        try:
            result = json.loads(content)
        except:
            print(f"Language: {language}")
            print(f"Content:\n{content}")
            print("-" * 40)
            result = []
        return result
    return []


qid = 0
result = {}
for old_id, data in raw_responses.items():
    im = data["obs"]
    domain = data["domain"]
    world = data["world"]
    qas = preprocess(data["model_response"])
    if isinstance(qas, dict):
        continue
    for qa in qas:
        result[qid] = qa
        result[qid]["obs"] = im
        result[qid]["domain"] = domain
        result[qid]["world"] = world
        qid += 1

json.dump(
    result,
    open(save_path, "w")
)

