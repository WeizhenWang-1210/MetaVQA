import os
import json
stats = [
    "/home/weizhen/experiments/main/random_test_results_stats.json",
    "/home/weizhen/experiments/main/LLavaNext_test_results_stats.json",
    "/home/weizhen/experiments/main/LLavaOne_test_results_stats.json",
    "/home/weizhen/experiments/main/Qwen2_test_results_stats.json",
    "/home/weizhen/experiments/main/GPT4o_test_results_stats.json",
    "/home/weizhen/experiments/main/Llama_test_results_stats.json",
    "/home/weizhen/experiments/main/InternVL_test_results_stats.json",
    "/home/weizhen/experiments/main/InternVL_trainval_test_results_stats.json",
]





ids = [
    os.path.basename(stat).split("test")[0][:-1] for stat in stats
]


results = [
    json.load(open(stat,"r")) for stat in stats
]


final = dict()
for modelid, result in zip(ids, results):
    for q_id, record in result["stats"].items():
        splitted = q_id.split("_")
        doman = splitted[0]
        q_type = "_".join(splitted[1:])
        identifier = f"{modelid},{doman},{q_type}"
        accu = record["accuracy"]
        #print(identifier, accu)
        final[identifier] = accu

json.dump(
    final,
    open("./aggregated_stats.json","w"),
    indent=2
)

