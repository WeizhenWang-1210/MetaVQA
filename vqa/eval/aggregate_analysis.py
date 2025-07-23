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
#extras
    "/home/weizhen/experiments/extra/Qwen2_trainval_test_results_stats.json",
    "/home/weizhen/experiments/extra/Llama_trainval_test_results_stats.json"
]


super_categories = dict(
    identify_color="spatial",
    identify_type="spatial",
    identify_distance="spatial",
    identify_position="spatial",
    identify_heading="spatial",
    pick_closer="spatial",
    predict_crash_ego_still="embodied",
    predict_crash_ego_dynamic="embodied",
    relative_distance="spatial",
    relative_position="spatial",
    relative_heading="spatial",
    relative_predict_crash_still="spatial",
    relative_predict_crash_dynamic="spatial",
    identify_closest="spatial",
    identify_leftmost="spatial",
    identify_rightmost="spatial",
    identify_frontmost="spatial",
    identify_backmost="spatial",
    order_closest="spatial",
    order_leftmost="spatial",
    order_rightmost="spatial",
    order_frontmost="spatial",
    order_backmost="spatial",
    describe_sector="spatial",
    describe_distance="spatial",
    embodied_distance="embodied",
    embodied_sideness="embodied",
    embodied_collision="embodied",
    grounding="grounding"
)




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
        super_type = super_categories[q_type]
        identifier = f"{modelid},{doman},{super_type}, {q_type}"
        accu = record["accuracy"]
        final[identifier] = accu

json.dump(
    final,
    open("/home/weizhen/final/aggregated_stats.json","w"),
    indent=2
)

