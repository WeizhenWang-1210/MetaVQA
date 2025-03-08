import json
import glob
import re
from som.parse_responses import parse_response, parse_gpt

NOT_YET_PARSED = ["describe_scenario"]
path_template = "/home/chenda/evaluations/rebuttal/InternVL_rebuttal_gpt/*_test_results.json"
merged_path = "/home/weizhen/experiments/rebuttal/InternVL_rebuttal_gpt_test_results.json"
stat_path = "/home/weizhen/experiments/rebuttal/InternVL_rebuttal_gpt_test_results_stats.json"
domained_path = "/home/weizhen/data_weizhen/metavqa_cvpr/datasets/test/test/test_processed.json"
original_qa = json.load(open(domained_path, "r"))


def merge_responses(response_paths):
    results = glob.glob(response_paths)
    print(f"Found {len(results)} files:\n", results)
    final = dict()
    for result in results:
        qas = json.load(open(result, "r"))
        for key, value in qas.items():
            assert key not in final.keys(), "Repetitive keys found!"
            #assert "domain" in original_qa.keys(), "You should have domain information"
            final[key] = value
            final[key]["options"]= original_qa[key]["options"]
            final[key]["domain"] = original_qa[key]["domain"]
            #print(final[key])
    return final


def accuracy_by_domain(qa_records):
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
        statistics[identifier]["correct"] += 1 if record["final_choice"].upper() == record["answer"].upper() else 0
    for type, stat in statistics.items():
        stat["accuracy"] = stat["correct"] / stat["total"]
        total += stat["total"]
        total_correct += stat["correct"]
    return statistics, total, total_correct


def analyze(basepath, mergepath, statpath):
    final = merge_responses(basepath)
    parsefail = 0
    for qid in final.keys():
        answer2opt = {str(val): str(key) for key, val in final[qid]["options"].items()} if final[qid]["options"] is not None \
            else {}
        choice = parse_response(final[qid]["model_response"], answer2opt) #parse_gpt(final[qid]["model_response"])#parse_response(final[qid]["model_response"], answer2opt)
        if choice == "" and final[qid]["type"] not in NOT_YET_PARSED:
            parsefail += 1
        print(final[qid]["model_response"])
        print(f"Parsed choice: {choice}")
        print("______")
        final[qid]["final_choice"] = choice
    statistics, total, total_correct = accuracy_by_domain(final)
    sim_correct = real_correct = sim_total = real_total = 0
    for key, record in statistics.items():
        if key.startswith("sim"):
            sim_total += record["total"]
            sim_correct += record["correct"]
        elif key.startswith("real"):
            real_total += record["total"]
            real_correct += record["correct"]
    stats = dict(
        total_questions=total, fail_rate=parsefail / total, total_correct=total_correct, total_sims=sim_total,
        total_reals=real_total,
        sim_correct=sim_correct, real_correct=real_correct, stats=statistics
    )
    json.dump(final, open(mergepath, "w"), indent=1)
    json.dump(stats, open(statpath, "w"), indent=1)


if __name__ == "__main__":
    analyze(path_template, merged_path, stat_path)
