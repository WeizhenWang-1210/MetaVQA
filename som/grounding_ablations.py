import json, os
import glob
import re


original_qas = "/home/weizhen/data_weizhen/metavqa_cvpr/general_ablations/val.json"

path_template = "/home/chenda/evaluations/lmms/Qwen2/*_Qwen2-VL-7B_zeroshot_general_ablations.json"
merged_path = "/home/weizhen/eval/Qwen2/Qwen2-VL-7B_zeroshot_general_ablations.json"
stat_path = "/home/weizhen/eval/Qwen2/Qwen2-VL-7B_zeroshot_general_ablations_stats.json"
results = glob.glob(path_template)

final = dict()

NOT_YET_PARSED = ["describe_scenario", "identify_color"]

for result in results:
    qas = json.load(open(result, "r"))
    for key, value in qas.items():
        assert key not in qas.items(), "Repetitive keys found!"
        final[key] = value

original = json.load(open(original_qas, "r"))



for result in results:
    qas = json.load(open(result, "r"))
    for key, value in qas.items():
        assert key not in qas.items(), "Repetitive keys found!"
        final[key] = value
        final[key]["variant"] = original[key]["variant"]



def parse_response_llava(all_response, zero_shot=False):
    pattern = r'\t"ANSWER":(.*?)"(.*?)"\n'
    response = all_response.split("assistant")[-1]
    #print(response)
    if zero_shot:
        matches = re.findall(pattern, response)
        if len(matches) > 0:
            #print(f"Match:{matches[-1]}")
            #print("___________")
            for choice in ["(A)", "(B)", "(C)", "(D)", "(a)", "(b)", "(c)", "(d)"]:
                #print(choice, matches[-1][-1]])
                if choice in matches[-1][-1]:
                    #print("here")
                    return matches[-1][-1][1]
            return matches[-1][-1]
        else:
            #print(f"No Match")
            #print("___________")
            return ""
    else:
        return ""




def accuracy_analysis(qa_records, zero_shot=False):
    statistics = dict()
    total_correct = total = 0
    for qid, record in qa_records.items():
        if record["type"] in NOT_YET_PARSED:
            print("here")
            continue
        type_variant = "{}_{}".format(str(record["variant"]), record["type"])
        if type_variant not in statistics.keys():
            statistics[type_variant] = dict(
                total=0, correct=0
            )
        statistics[type_variant]["total"] += 1
        if zero_shot:
            statistics[type_variant]["correct"] += 1 if record["final_choice"].upper() == "({})".format(
                record["answer"]).upper() else 0
        else:
            statistics[type_variant]["correct"] += 1 if record["final_choice"].upper() == record[
                "answer"].upper() else 0
    for type, stat in statistics.items():
        stat["accuracy"] = stat["correct"] / stat["total"]
        total += stat["total"]
        total_correct += stat["correct"]
    return statistics, total, total_correct


def accuracy_analysis_llava(qa_records, zero_shot=False):
    statistics = dict()
    total_correct = total = 0
    for qid, record in qa_records.items():
        if record["type"] in NOT_YET_PARSED or record["type"]!="grounding":
            print("here")
            continue
        type_variant = "{}".format(str(record["variant"]), record["type"])
        if type_variant not in statistics.keys():
            statistics[type_variant] = dict(
                total=0, correct=0
            )
        statistics[type_variant]["total"] += 1
        if zero_shot:
            statistics[type_variant]["correct"] += 1 if record["final_choice"].upper() == record[
                "answer"].upper() else 0
        else:
            statistics[type_variant]["correct"] += 1 if record["final_choice"].upper() == record[
                "answer"].upper() else 0
    for type, stat in statistics.items():
        stat["accuracy"] = stat["correct"] / stat["total"]
        total += stat["total"]
        total_correct += stat["correct"]
    return statistics, total, total_correct




for qid in final.keys():
    choice = parse_response_llava(final[qid]["model_response"], True)
    #print(choice)
    final[qid]["final_choice"] = choice

statistics, total, total_correct = accuracy_analysis_llava(final, True)

result = dict(
    total_questions=total, total_correct=total_correct, stats=statistics
)
json.dump(final, open(merged_path, "w"), indent=1)
json.dump(result, open(stat_path, "w"), indent=1)
