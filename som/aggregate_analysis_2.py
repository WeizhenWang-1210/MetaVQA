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


DOMAIN="real"

ids = [
    os.path.basename(stat).split("test")[0][:-1] for stat in stats
]


results = [
    json.load(open(stat,"r")) for stat in stats
]
from pprint import pprint
from collections import defaultdict



aggregated = dict()
for modelid, result in zip(ids, results):
    questions = defaultdict(
        lambda: dict(
            total=0, correct=0
        )
    )
    super_types = defaultdict(
        lambda: dict(
            total=0, correct=0
        )
    )
    total_num = total_correct = 0
    for q_id, record in result["stats"].items():
        splitted = q_id.split("_")
        doman = splitted[0]
        if doman != DOMAIN:
            continue
        q_type = "_".join(splitted[1:])
        super_type = super_categories[q_type]
        #print(super_type, q_type)
        identifier = f"{modelid},{doman},{super_type},{q_type}"
        #print(identifier)
        total_num += record["total"]
        total_correct += record["correct"]
        super_types[super_type]["total"] += record["total"]
        super_types[super_type]["correct"] += record["correct"]
        questions[q_type]["total"] += record["total"]
        questions[q_type]["correct"] += record["correct"]


    question_accuracy = {
        key: value["correct"]/value["total"] for key, value in questions.items()
    }

    super_accuracy = {
        key: value["correct"]/value["total"] for key, value in super_types.items()
    }
    stat = dict(
        category=question_accuracy,
        super_category=super_accuracy,
        overall = total_correct/total_num
    )
    aggregated[modelid] = stat


json.dump(
    aggregated,
    open(f"/home/weizhen/final/overall_stats_{DOMAIN}.json","w"),
    indent=2
)

