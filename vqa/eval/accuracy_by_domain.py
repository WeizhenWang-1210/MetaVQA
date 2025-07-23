import json
import glob
import re

merged_path = "/home/weizhen/experiments/main/random_test_results.json"
stat_path = "/home/weizhen/experiments/main/random_test_results_stats.json"

NOT_YET_PARSED = ["describe_scenario"]
domained_path = "/data_weizhen/metavqa_cvpr/datasets/test/test/test.json"
original_qa = json.load(open(domained_path, "r"))


final = json.load(open(merged_path,"r"))

def accuracy_by_domain(qa_records, zero_shot=False):
    statistics = dict()
    total_correct = total = 0
    for qid, record in qa_records.items():
        if record["type"] in NOT_YET_PARSED:
            continue
        identifier = "_".join([record["domain"], record["type"]])

        if identifier not in statistics.keys():
            statistics[identifier] = dict(
                total=0, correct=0
            )
        statistics[identifier]["total"] += 1
        if zero_shot:
            statistics[identifier]["correct"] += 1 if record["final_choice"].upper() == record[
                "answer"].upper() else 0
        else:
            statistics[identifier]["correct"] += 1 if record["final_choice"].upper() == record[
                "answer"].upper() else 0
    for type, stat in statistics.items():
        stat["accuracy"] = stat["correct"] / stat["total"]
        total += stat["total"]
        total_correct += stat["correct"]
    return statistics, total, total_correct

statistics, total, total_correct = accuracy_by_domain(final, True)

sim_correct = real_correct = sim_total = real_total = 0

for key, record in statistics.items():
    if key.startswith("sim"):
        sim_total += record["total"]
        sim_correct += record["correct"]
    elif key.startswith("real"):
        real_total += record["total"]
        real_correct += record["correct"]

stats = dict(
    total_questions=total, total_correct=total_correct, total_sims=sim_total, total_reals=real_total,
    sim_correct=sim_correct, real_correct=real_correct, stats=statistics
)
json.dump(final, open(merged_path, "w"), indent=1)
json.dump(stats, open(stat_path, "w"), indent=1)
